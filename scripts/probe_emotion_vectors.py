import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# step 1 before a longer run of pilot B: does the emotional-vits vector (extract_emotion.py) separate angry, happy and
# surprise, how strong is that difference next to sad vs neutral, and does pilot B's emo_proj pass it on?
# CPU only: reads the stored <wav>.emo.npy files and only the emo_proj weights of the pilot checkpoints
# usage: python probe_emotion_vectors.py <checkpoint dir of the pilot> <output dir>
ckpt_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
root = Path("/data/chi-gpu1/ge96xah/data/ESD")
emotions = ["Neutral", "Angry", "Happy", "Sad", "Surprise"]
tasks = {"5-way": emotions, "triad": ["Angry", "Happy", "Surprise"], "angry-happy": ["Angry", "Happy"]}
pairs = [("Angry", "Happy"), ("Angry", "Surprise"), ("Happy", "Surprise"), ("Sad", "Neutral")]
torch.manual_seed(0)
torch.set_num_threads(8)

# every stored vector with speaker, emotion and split, ESD layout <speaker>/<Emotion>/<split>/<speaker>_<id>.wav
X, spk, emo, split = [], [], [], []
for s in ["train", "evaluation", "test"]:
    for line in (root / f"metadata_{s}.csv").read_text(encoding="utf-8").splitlines()[1:]:
        wav = Path(line.split("|")[0])
        X.append(np.load(f"{wav}.emo.npy"))
        spk.append(wav.parts[-4])
        emo.append(wav.parts[-3])
        split.append(s)
X = torch.from_numpy(np.stack(X)).float()
spk, emo, split = np.array(spk), np.array(emo), np.array(split)
speakers = sorted(set(spk))
assert set(emo) == set(emotions) and len(speakers) == 10, (set(emo), speakers)
print(len(X), "vectors,", len(speakers), "speakers", flush=True)


def fit_linear(xtr, ytr, xva, yva, xte):  # multinomial logistic regression, L2 strength chosen on the validation set
    best = None
    for lam in [1e-3, 1e-2, 1e-1]:
        w = torch.zeros(xtr.shape[1], int(ytr.max()) + 1, requires_grad=True)
        b = torch.zeros(w.shape[1], requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=100, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(xtr @ w + b, ytr) + lam * w.pow(2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            acc = ((xva @ w + b).argmax(1) == yva).float().mean().item()
            if best is None or acc > best[0]:
                best = (acc, (xte @ w + b).argmax(1))
    return best[1]


def fit_mlp(xtr, ytr, xva, yva, xte, epochs=40):  # one hidden layer, the epoch is chosen on the validation set
    net = torch.nn.Sequential(
        torch.nn.Linear(xtr.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(256, int(ytr.max()) + 1)
    )
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    best = None
    for _ in range(epochs):
        net.train()
        for i in torch.randperm(len(xtr)).split(256):
            opt.zero_grad()
            F.cross_entropy(net(xtr[i]), ytr[i]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            acc = (net(xva).argmax(1) == yva).float().mean().item()
            if best is None or acc > best[0]:
                best = (acc, net(xte).argmax(1))
    return best[1]


def probe(feats, labels, model, folds):  # folds: (train, validation, test) masks, test predictions are pooled
    keep = np.isin(emo, labels)
    y = torch.tensor([labels.index(e) if e in labels else -1 for e in emo])
    yt, pred = [], []
    for tr, va, te in folds:
        tr, va, te = (torch.from_numpy(m & keep) for m in (tr, va, te))
        m, s = feats[tr].mean(0), feats[tr].std(0) + 1e-6  # standardised with training statistics only
        x = (feats - m) / s
        pred.append((fit_linear if model == "linear" else fit_mlp)(x[tr], y[tr], x[va], y[va], x[te]))
        yt.append(y[te])
    yt, pred = torch.cat(yt), torch.cat(pred)
    conf = torch.zeros(len(labels), len(labels), dtype=torch.long)
    for a, p in zip(yt.tolist(), pred.tolist()):
        conf[a, p] += 1
    recall = (conf.diag() / conf.sum(1)).tolist()
    return {"acc": (pred == yt).float().mean().item(), "recall": dict(zip(labels, recall)), "confusion": conf.tolist(), "n": len(yt)}


# sentences: the use in pilot B, seen speakers with unseen sentences (train -> select on evaluation -> test)
# speakers: stricter, 5 folds with 2 held-out speakers each
splits = {"sentences": [(split == "train", split == "evaluation", split == "test")]}
splits["speakers"] = []
for i in range(0, 10, 2):
    held = np.isin(spk, speakers[i : i + 2])
    splits["speakers"].append((~held & (split == "train"), ~held & (split == "evaluation"), held))
Xn = X.clone()  # speaker-normalised: minus the mean vector of the speaker over all its clips (uses no labels)
for s in speakers:
    Xn[torch.from_numpy(spk == s)] -= X[torch.from_numpy(spk == s)].mean(0)

res = {"n_vectors": len(X), "probes": {}}
for sname, folds in splits.items():
    for feats, fname, models in [(X, "raw", ["linear", "mlp"]), (Xn, "speaker-normalised", ["linear"])]:
        for model in models:
            for tname, labels in tasks.items():
                r = probe(feats, labels, model, folds)
                res["probes"][f"{sname} | {fname} | {model} | {tname}"] = r
                recall = " ".join(f"{e} {v:.2f}" for e, v in r["recall"].items())
                print(f"{sname:9s} {fname:18s} {model:6s} {tname:11s} acc {r['acc']:.3f} | recall {recall}", flush=True)


def separation(feats):  # per speaker: distance of class means / within-class spread, then mean over speakers
    out = {f"{a}-{b}": [] for a, b in pairs}
    for s in speakers:
        mu, var, n = {}, {}, {}
        for e in emotions:
            x = feats[torch.from_numpy((spk == s) & (emo == e))]
            mu[e], var[e], n[e] = x.mean(0), (x - x.mean(0)).pow(2).sum(1).mean(), len(x)
        for a, b in pairs:
            # the estimated means carry noise of about var/(n-1) each, subtracted so that no difference gives about 0
            d2 = (mu[a] - mu[b]).pow(2).sum() - var[a] / (n[a] - 1) - var[b] / (n[b] - 1)
            out[f"{a}-{b}"].append((d2.clamp(min=0).sqrt() / ((var[a] + var[b]) / 2).sqrt()).item())
    return {k: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))} for k, v in out.items()}


def nearest_centroid(feats, labels):  # per speaker, centroids from the train split, test split classified
    hit = []
    for s in speakers:
        c = torch.stack([feats[torch.from_numpy((spk == s) & (emo == e) & (split == "train"))].mean(0) for e in labels])
        te = (spk == s) & (split == "test") & np.isin(emo, labels)
        pred = torch.cdist(feats[torch.from_numpy(te)], c).argmin(1)
        hit += (pred == torch.tensor([labels.index(e) for e in emo[te]])).tolist()
    return float(np.mean(hit))


# signal strength in the raw vector, the scale emo_proj sees: is angry/happy/surprise much weaker than sad/neutral?
res["separation_raw"] = separation(X)
res["nearest_centroid_raw"] = {t: nearest_centroid(X, labels) for t, labels in tasks.items()}
print("separation raw:", {k: round(v["mean"], 2) for k, v in res["separation_raw"].items()}, flush=True)

# how much of the triad difference lies in the high-variance directions that are learned first
mu = X[torch.from_numpy(split == "train")].mean(0)
_, S, Vh = torch.linalg.svd(X[torch.from_numpy(split == "train")] - mu, full_matrices=False)
share = (S**2).cumsum(0) / (S**2).sum()
res["triad_from_top_components"] = {}
for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]:
    r = probe((X - mu) @ Vh[:k].T, tasks["triad"], "linear", splits["sentences"])
    res["triad_from_top_components"][k] = {"acc": r["acc"], "variance_share": share[k - 1].item()}
    print(f"top {k:4d} components ({share[k - 1].item():.1%} of the variance): triad acc {r['acc']:.3f}", flush=True)

# what pilot B's emo_proj makes of it, per checkpoint: offsets that are added to every text token
res["pilot"] = {}
for p in sorted(ckpt_dir.glob("model_*.pt"), key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0)):
    if p.name == "model_last.pt":
        continue
    ckpt = torch.load(p, map_location="cpu", weights_only=True, mmap=True)  # only the tensors used below are read
    sd = ckpt["model_state_dict"]
    w, b = sd["transformer.text_embed.emo_proj.weight"].float(), sd["transformer.text_embed.emo_proj.bias"].float()
    token = sd["transformer.text_embed.text_embed.weight"].float().norm(dim=1).median().item()  # typical text token
    P = X @ w.T + b
    offset = {e: P[torch.from_numpy(emo == e)].mean(0) for e in emotions}
    r = {
        "update": int(ckpt["update"]),
        "emo_proj_mean_abs": w.abs().mean().item(),
        "token_norm": token,
        "offset_vs_neutral_per_token_norm": {e: ((offset[e] - offset["Neutral"]).norm() / token).item() for e in emotions},
        "separation": separation(P),
        "nearest_centroid": {t: nearest_centroid(P, labels) for t, labels in tasks.items()},
    }
    res["pilot"][p.name] = r
    print(
        f"{p.name}: |W| {r['emo_proj_mean_abs']:.2e}, separation",
        {k: round(v["mean"], 2) for k, v in r["separation"].items()},
        "nearest centroid",
        {k: round(v, 3) for k, v in r["nearest_centroid"].items()},
        flush=True,
    )

# statistics of the vectors, for a later control run with fixed per-emotion codes of the same scale
out.mkdir(parents=True, exist_ok=True)
np.savez(out / "vector_stats.npz", mean=X.mean(0).numpy(), std=X.std(0).numpy())
res["vector_norms"] = {"l1": X.abs().sum(1).mean().item(), "l2": X.norm(dim=1).mean().item()}
with open(out / "probe_emotion_vectors.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
print("done", out / "probe_emotion_vectors.json")
