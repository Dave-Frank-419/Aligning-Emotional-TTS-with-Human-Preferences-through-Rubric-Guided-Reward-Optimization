import csv
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoFeatureExtractor, WavLMModel

from f5_tts.model.dataset import trim_silence

# second emotion judge, independent of emotion2vec+ and of the conditioning vector: WavLM-large features (upper half of
# the layers and time averaged) with a linear classifier trained on real ESD; the classifier for a speaker never saw it
# usage: python score_wavlm_judge.py <cache dir> <manifest.csv> [more manifests]  -> scores_wavlm.csv next to each manifest
cache, manifests = Path(sys.argv[1]), [Path(m) for m in sys.argv[2:]]
root = Path("/data/chi-gpu1/ge96xah/data/ESD")
model_name = "/nas/student/DavidFrank/models/wavlm-large"  # microsoft/wavlm-large, downloaded by cluster/eval_setup.sh
emotions = ["neutral", "angry", "happy", "sad", "surprise"]
triad = [1, 2, 4]
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = WavLMModel.from_pretrained(model_name).to(device).eval()


def features(path):  # cut at the silent edges like the generated speech, then 16 kHz
    audio, sr = torchaudio.load(str(path))
    audio = torchaudio.functional.resample(trim_silence(audio.mean(0, keepdim=True), sr), sr, 16000)[0]
    x = extractor(audio.numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        hidden = model(x, output_hidden_states=True).hidden_states
    return torch.stack(hidden[len(hidden) // 2 :]).mean(0).mean(1)[0].cpu()


def esd(split, per_class=None):  # (features, speakers, emotion ids) of real recordings
    wavs = [Path(l.split("|")[0]) for l in (root / f"metadata_{split}.csv").read_text(encoding="utf-8").splitlines()[1:]]
    if per_class:  # an even spread of utterances per speaker and emotion
        groups = {}
        for w in sorted(wavs):
            groups.setdefault((w.parts[-4], w.parts[-3]), []).append(w)
        wavs = [w for g in groups.values() for w in g[:: max(1, len(g) // per_class)][:per_class]]
    x = torch.stack([features(w) for w in wavs])
    return x, [w.parts[-4] for w in wavs], torch.tensor([emotions.index(w.parts[-3].lower()) for w in wavs])


feats = cache / "wavlm_esd_features.pt"
if feats.exists():
    data = torch.load(feats)
else:
    data = {"train": esd("train", per_class=100), "evaluation": esd("evaluation")}
    tmp = feats.with_name(f"{feats.stem}.{os.getpid()}.tmp")  # own name per job, then atomic: never a half-written cache
    torch.save(data, tmp)
    os.replace(tmp, feats)
xtr, str_, ytr = data["train"]
speakers = sorted(set(str_))


def fit(x, y, lam):  # multinomial logistic regression with L2, as in probe_emotion_vectors.py
    w = torch.zeros(x.shape[1], len(emotions), requires_grad=True)
    b = torch.zeros(len(emotions), requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x @ w + b, y) + lam * w.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


judges = {}
for held in speakers:  # L2 strength chosen on one more held-out speaker, then refit on all others
    rest = [s for s in speakers if s != held]
    mask = torch.tensor([s in rest for s in str_])
    m, sd = xtr[mask].mean(0), xtr[mask].std(0) + 1e-6
    x = (xtr - m) / sd
    val = torch.tensor([s == rest[0] for s in str_])
    tr = mask & ~val

    def val_acc(lam):
        w, b = fit(x[tr], ytr[tr], lam)
        return ((x[val] @ w + b).argmax(1) == ytr[val]).float().mean().item()

    judges[held] = (m, sd, *fit(x[mask], ytr[mask], max([1e-3, 1e-2, 1e-1], key=val_acc)))


def probs(x, speaker):
    m, sd, w, b = judges[speaker]
    return torch.softmax(((x - m) / sd) @ w + b, -1)


# how well this judge separates the emotions on real recordings of unseen speakers (its ceiling)
xev, sev, yev = data["evaluation"]
pev = torch.stack([probs(x, s) for x, s in zip(xev, sev)])
tri = torch.tensor([int(y) in triad for y in yev])
ceiling = {
    "5-way": (pev.argmax(1) == yev).float().mean().item(),
    "triad": (torch.tensor(triad)[pev[tri][:, triad].argmax(1)] == yev[tri]).float().mean().item(),
}
print("wavlm judge on real evaluation recordings of unseen speakers:", {k: round(v, 3) for k, v in ceiling.items()}, flush=True)
with open(cache / "wavlm_judge_ceiling.json", "w", encoding="utf-8") as f:
    json.dump(ceiling, f, indent=1)

for manifest in manifests:
    with open(manifest, encoding="utf-8", newline="") as f:
        items = list(csv.DictReader(f))
    with open(manifest.parent / "scores_wavlm.csv", "w", encoding="utf-8", newline="") as f:
        out = csv.writer(f)
        out.writerow(["path"] + emotions)
        for it in items:
            out.writerow([it["path"]] + [f"{p:.6g}" for p in probs(features(it["path"]), it["speaker"]).tolist()])  # keeps tiny values
    print("scored", len(items), "->", manifest.parent / "scores_wavlm.csv", flush=True)
