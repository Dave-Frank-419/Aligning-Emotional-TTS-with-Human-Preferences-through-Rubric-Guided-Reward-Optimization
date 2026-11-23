"""Word errors of generated clips: a guard of the preference stage (the winner of a pair may not have more word errors
than its loser) and a measure of damage (the word errors of neutral after training).

Recogniser: openai/whisper-base (a small model, so the count is a relative measure between clips of the same
sentences, not a word error rate to report). The reference is the neutral transcription of the clip's sentence (ESD is
parallel; the generation speaks that text). Both sides go through whisper's English normaliser, so that "9" against
"nine" or "Mr." against "Mister" is not an error. Clips may exceed full scale (float WAV): each is scaled to a peak of
at most 1 before recognition. A manifest whose wer.csv covers all of its clips and is newer than each of them and than
this script is skipped.
usage: python score_words.py [--esd /data/chi-gpu1/ge96xah/data/ESD] <manifest.csv> [...]  -> wer.csv next to each
       manifest (path, errors, words, heard)
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor

argv = sys.argv[1:]
esd = Path("/data/chi-gpu1/ge96xah/data/ESD")
if argv and argv[0] == "--esd":
    esd, argv = Path(argv[1]), argv[2:]


def errors(ref, hyp):  # word-level edit distance
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1]


text = {}  # (speaker, sentence) -> the neutral transcription
for split in ("train", "evaluation", "test"):
    with open(esd / "esd_csvs" / f"{split}.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["lang"] == "en" and r["emotion"].lower() == "neutral":
                name = os.path.basename(r["relative_audio_path"])
                text[(int(name[:4]), (int(Path(name).stem.split("_")[1]) - 1) % 350)] = r["transcription"].strip()

proc = WhisperProcessor.from_pretrained("openai/whisper-base")
asr = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to("cuda").eval()
norm = lambda s: proc.tokenizer.normalize(s).split()
for m in argv:
    man, out = pd.read_csv(m), Path(m).parent / "wer.csv"
    if out.exists() and set(pd.read_csv(out).path) >= set(man.path) and out.stat().st_mtime > max([Path(__file__).stat().st_mtime] + [Path(q).stat().st_mtime for q in man.path]):
        print(f"{out}: complete, skipped", flush=True)
        continue
    rows = []
    for r in man.itertuples():
        x, sr = sf.read(r.path, dtype="float32", always_2d=True)
        x = resample_poly(x.mean(1), 16000, sr).astype(np.float32)
        x = x / max(1.0, float(np.abs(x).max()))
        feats = proc(x, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")
        with torch.no_grad():
            heard = proc.batch_decode(asr.generate(feats, language="en", task="transcribe"), skip_special_tokens=True)[0].strip()
        ref = norm(text[(int(r.speaker), int(r.sentence))])
        rows.append(dict(path=r.path, errors=errors(ref, norm(heard)), words=len(ref), heard=heard))
    o = pd.DataFrame(rows)
    o.to_csv(out, index=False)
    print(f"{out}: clips {len(o)}; without a word error {int((o.errors == 0).sum())}; errors per reference word {o.errors.sum() / o.words.sum():.3f}", flush=True)
