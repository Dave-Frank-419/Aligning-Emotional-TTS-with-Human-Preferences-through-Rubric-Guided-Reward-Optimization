import csv
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from funasr import AutoModel

# first emotion judge: emotion2vec+ large (9 classes), the categorical model also used in EMORL-TTS
# runs on the CPU in its own environment (funasr, see cluster/eval_setup.sh), so the F5 environment stays untouched
# usage: python score_emotion2vec.py <model dir> <manifest.csv> [more manifests]  -> scores_e2v.csv next to each manifest
model_dir, manifests = sys.argv[1], [Path(m) for m in sys.argv[2:]]
model = AutoModel(model=model_dir, disable_update=True, device="cpu")


def trim_silence(audio, sample_rate, threshold_db=-42.0, chunk_ms=10):  # same as trim_silence in F5-TTS dataset.py
    chunk = sample_rate * chunk_ms // 1000
    n = audio.shape[-1] // chunk
    rms = audio[0, : n * chunk].reshape(n, chunk).pow(2).mean(-1).sqrt()
    loud = torch.nonzero(20 * torch.log10(rms.clamp(min=1e-10)) >= threshold_db).flatten()
    if len(loud) == 0:
        return audio
    return audio[:, loud[0] * chunk : (loud[-1] + 1) * chunk]


def scores(path):  # trimmed like the generated speech and resampled here, funasr assumes 16 kHz for arrays
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = trim_silence(torch.from_numpy(audio.T).mean(0, keepdim=True), sr)
    audio = torchaudio.functional.resample(audio, sr, 16000)[0].numpy()
    res = model.generate(audio, granularity="utterance", extract_embedding=False)[0]
    return {label.split("/")[-1]: s for label, s in zip(res["labels"], res["scores"])}  # e.g. '生气/angry' -> angry


for manifest in manifests:
    with open(manifest, encoding="utf-8", newline="") as f:
        items = list(csv.DictReader(f))
    rows = [{"path": it["path"], **scores(it["path"])} for it in items]
    labels = [k for k in rows[0] if k != "path"]
    assert {"neutral", "angry", "happy", "sad", "surprised"} <= set(labels), labels
    with open(manifest.parent / "scores_e2v.csv", "w", encoding="utf-8", newline="") as f:
        out = csv.DictWriter(f, fieldnames=["path"] + labels)
        out.writeheader()
        out.writerows(rows)
    print("scored", len(rows), "->", manifest.parent / "scores_e2v.csv", "| labels", labels, flush=True)
