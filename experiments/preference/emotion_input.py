"""The emotion input of the preference stage.

The fine-tuned F5-TTS takes an utterance-level emotion vector (1024-d, the last hidden layer of
audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim averaged over time, as written by scripts/extract_emotion.py next
to every ESD recording as <file>.wav.emo.npy). At inference the vector of a source recording of the intended emotion is
given. The emotion input moves that vector along what is unique to the intended emotion:

    vector = vector of the source recording + strength(emotion) x push(speaker, emotion)

    own push(s, e)  for speaker s: the direction from s's neutral centroid to s's centroid of e, reduced to the part
                    orthogonal to s's directions of the other three emotions (least squares), and scaled back to the
                    length of the full direction. Centroids are means over s's ESD training recordings.
    push(s, e)      the mean of the own pushes of e over the five OTHER evaluation speakers (the push "knows only the
                    emotion", not the speaker); a speaker outside the six evaluation speakers gets the mean over all six.

Strength 1 is a move as long as the typical distance from neutral to the emotion. The strengths of the preference stage
were chosen by ear (angry 0.5, happy 2, sad 1.5, surprise 1). Neutral items are never moved.
"""
import csv
from pathlib import Path

import numpy as np

SPEAKERS = ["0011", "0013", "0015", "0016", "0018", "0020"]  # the six evaluation speakers of the thesis
FOUR = ["angry", "happy", "sad", "surprise"]


def parse_strengths(text):
    """'angry=0.5,happy=2,sad=1.5,surprise=1' -> dict; '' -> no emotion input."""
    if not text:
        return {}
    out = {k: float(v) for k, v in (part.split("=") for part in text.split(","))}
    assert sorted(out) == sorted(FOUR), f"a strength is given for each of {FOUR}: {text}"
    return out


def centroids(esd):
    """Mean emotion vector (float64) per (speaker, emotion) over the speaker's English ESD training recordings, in the
    order of their file names."""
    files = {}
    with open(Path(esd) / "esd_csvs" / "train.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["lang"] == "en":
                files.setdefault((int(r["relative_audio_path"][:4]), r["emotion"].lower()), []).append(r["relative_audio_path"])
    return {k: np.mean(np.stack([np.load(Path(esd) / f"{p}.emo.npy") for p in sorted(v)]).astype(np.float64), 0) for k, v in files.items()}


def pushes(esd):
    """push[(speaker as int, emotion)] for the six evaluation speakers, and outside[emotion] for any other speaker."""
    cen, ids = centroids(esd), [int(s) for s in SPEAKERS]
    own = {}
    for s in ids:
        d = {o: cen[(s, o)] - cen[(s, "neutral")] for o in FOUR}
        for e in FOUR:
            others = np.stack([d[o] for o in FOUR if o != e], 1)
            coef, *_ = np.linalg.lstsq(others, d[e], rcond=None)
            u = d[e] - others @ coef
            own[(s, e)] = np.linalg.norm(d[e]) * u / np.linalg.norm(u)
    push = {(s, e): np.mean([own[(x, e)] for x in ids if x != s], 0) for s in ids for e in FOUR}
    outside = {e: np.mean([own[(x, e)] for x in ids], 0) for e in FOUR}
    return push, outside


if __name__ == "__main__":  # prints the length of every push, a check of the centroids
    import sys

    push, outside = pushes(sys.argv[1] if len(sys.argv) > 1 else "/data/chi-gpu1/ge96xah/data/ESD")
    for e in FOUR:
        print(f"{e}: length of the push per evaluation speaker " + " ".join(f"{np.linalg.norm(push[(int(s), e)]):.3f}" for s in SPEAKERS)
              + f"; for other speakers {np.linalg.norm(outside[e]):.3f}")
