"""Pitch of generated clips against the speaker's real range: the guard of the preference stage against overdone speech.

The listener's tick "overdone or unnatural" on the listening pages of 2026-11-20 to 2026-11-23 was predicted by the
clip's mean pitch as a z value against the speaker's real recordings of the same emotion (zP): in all 6 pairs with
exactly one ticked clip the ticked clip had the higher zP, area under the curve 0.73 [0.58, 0.86] over 768 clips (ticked
clips +2.3, the others +0.4). In the pair rule a clip with zP above 2 can be no winner.
Method (as the measurements behind that check): mono, resampled to 16 kHz, level as it is, openSMILE eGeMAPSv02
functionals; pitch = F0semitoneFrom27.5Hz_sma3nz_amean (mean over voiced frames, in semitones).
    frame  the six evaluation speakers' real ESD recordings of the splits test and evaluation (50 per speaker and
           emotion, not seen in training): mean, standard deviation and 95th percentile of the pitch per speaker and emotion
    clips  per clip of a manifest: pitch, its variability (F0 stddevNorm), zP = (pitch - mean) / sd of the frame of the
           clip's speaker and emotion, and whether it lies beyond the 95th percentile (neutral clips get their own frame)
One CPU core per process. Needs openSMILE (pip install opensmile).
usage: python score_pitch.py frame <frame.csv> [--esd /data/chi-gpu1/ge96xah/data/ESD]
       python score_pitch.py clips <frame.csv> <manifest.csv> [...]   -> pitch.csv next to each manifest
"""
import math
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import opensmile
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

SPEAKERS = [11, 13, 15, 16, 18, 20]
EMOS = ["neutral", "angry", "happy", "sad", "surprise"]
F0, VAR = "F0semitoneFrom27.5Hz_sma3nz_amean", "F0semitoneFrom27.5Hz_sma3nz_stddevNorm"
fun = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)


def measure(path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1).astype(np.float32)
    g = math.gcd(sr, 16000)
    x16 = x if sr == 16000 else resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    f = fun.process_signal(x16, 16000).iloc[0]
    return float(f[F0]), float(f[VAR])


if __name__ == "__main__":
    mode, frame_file = sys.argv[1], Path(sys.argv[2])
    if mode == "frame":
        esd = Path(sys.argv[sys.argv.index("--esd") + 1]) if "--esd" in sys.argv else Path("/data/chi-gpu1/ge96xah/data/ESD")
        rows = []
        for s in SPEAKERS:
            for e in EMOS:
                pitch = [measure(p)[0] for split in ("test", "evaluation") for p in sorted((esd / f"{s:04d}" / e.capitalize() / split).glob("*.wav"))]
                rows.append(dict(speaker=s, emotion=e, n=len(pitch), mean=float(np.mean(pitch)), sd=float(pd.Series(pitch).std()), p95=float(np.percentile(pitch, 95))))
                print(f"{s:04d} {e}: {len(pitch)} real recordings, pitch mean {rows[-1]['mean']:.2f}, sd {rows[-1]['sd']:.2f}, p95 {rows[-1]['p95']:.2f}", flush=True)
        pd.DataFrame(rows).to_csv(frame_file, index=False)
    elif mode == "clips":
        frame = {(int(r.speaker), r.emotion): r for r in pd.read_csv(frame_file).itertuples()}
        for m in sys.argv[3:]:
            man, out = pd.read_csv(m), Path(m).parent / "pitch.csv"
            rows = []
            for r in man.itertuples():
                pitch, var = measure(r.path)
                fr = frame[(int(r.speaker), r.emotion)]
                rows.append(dict(path=r.path, pitch=pitch, pitch_var=var, zP=(pitch - fr.mean) / fr.sd, beyond=bool(pitch > fr.p95)))
            o = pd.DataFrame(rows)
            o.to_csv(out, index=False)
            print(f"{out}: clips {len(o)}; zP mean {o.zP.mean():+.2f}; above 2: {int((o.zP > 2).sum())}; beyond the 95th percentile: {int(o.beyond.sum())}", flush=True)
    else:
        sys.exit("usage: score_pitch.py frame <frame.csv> [--esd <root>] | clips <frame.csv> <manifest.csv> [...]")
