"""Level matching of generated clips before they are compared (the evaluation of a trained state).

Every clip of the four emotions is scaled so that its RMS level is that of the start model's clip of the same speaker,
sentence, emotion and seed: the emotion input changes the level by several dB for some speakers, the listener heard a
rise of about 6 dB as not sadder, and level is the one property that can be set after the generation, so it is taken
out of every comparison. Neutral clips are copied as they are. The clips are written as float WAV under the same names
into a new folder with a manifest of the same fields; a folder that holds a manifest is never written again.
usage: python level_match.py <folder of the generated clips> <folder to write> <table of the start model's clips of the
           same set and seed offset (columns speaker, sentence, emotion, path; its manifest will do)>
"""
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

raw, out, ref = Path(sys.argv[1]), Path(sys.argv[2]), pd.read_csv(sys.argv[3])
assert not (out / "manifest.csv").exists(), f"{out} holds a manifest already: a folder is written once"
out.mkdir(parents=True, exist_ok=True)
local = lambda p: Path(str(p).replace("\\", "/"))
start = {(int(r["speaker"]), int(r["sentence"]), r["emotion"]): local(r["path"]) for r in ref.to_dict("records")}
rms = lambda w: float(np.sqrt(np.mean(np.asarray(w, dtype=np.float64) ** 2)))
with open(raw / "manifest.csv", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
gains, new = [], []
for r in rows:
    name = local(r["path"]).name
    if r["emotion"] == "neutral":
        shutil.copyfile(raw / name, out / name)
        g = 1.0
    else:
        w, sr = sf.read(str(raw / name), dtype="float32")
        s, sr0 = sf.read(str(start[(int(r["speaker"]), int(r["sentence"]), r["emotion"])]), dtype="float32")
        assert sr == sr0 and len(w) == len(s), f"{name}: the start model's clip has another length or rate"
        g = rms(s) / max(rms(w), 1e-10)
        sf.write(str(out / name), (w * g).astype(np.float32), sr, subtype="FLOAT")
    gains.append(dict(file=name, speaker=r["speaker"], emotion=r["emotion"], gain_db=round(20 * np.log10(g), 3)))
    new.append(dict(r, path=str(out / name)))
with open(out / "manifest.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(new)
g = pd.DataFrame(gains)
g.to_csv(out / "level_match.csv", index=False)
if (raw / "condition.txt").exists():
    (out / "condition.txt").write_text((raw / "condition.txt").read_text(encoding="utf-8").rstrip("\n") + "; every clip of the four emotions scaled to the RMS level of the "
                                       "start model's clip of the same item and seed (experiments/preference/level_match.py)\n", encoding="utf-8")
x = g[g["emotion"] != "neutral"]
print(f"{out}: {len(rows)} clips; gain applied to the {len(x)} clips of the four emotions: mean {x['gain_db'].mean():+.2f} dB, from {x['gain_db'].min():+.1f} to {x['gain_db'].max():+.1f} dB")
