import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# blind three-way listening test (angry / happy / surprise) on one arm of the triad evaluation
# make:  python make_blind_test.py make <arm dir> <test dir> [clips per emotion, default 15]
#        <test dir>/clips/blind_NN.wav at the same loudness and answers.csv to fill in; only this folder is copied to listen,
#        the solution goes to <parent>/blind_keys/<test name>.csv
# best:  python make_blind_test.py best <arm dir> <test dir> <key of an earlier test> [clips per emotion, default 5] [shortest, default 2 s]
#        as make, but per emotion the clips both judges rate highest, as a reward would pick them; clips of the earlier test
#        and very short ones are left out. Shows whether the arm can render an emotion at all, not whether it does so reliably
# score: python make_blind_test.py score <key.csv> <filled answers.csv>   (answer per clip: angry, happy or surprise)
# with 45 clips, 21 or more correct is above chance (one-sided p < 0.05); with 15 clips, 9 or more
triad = ["angry", "happy", "surprise"]


def rows(path):  # also as saved by Excel: BOM and ';' as separator
    with open(path, encoding="utf-8-sig", newline="") as f:
        text = f.read()
    return list(csv.DictReader(text.splitlines(), delimiter=";" if ";" in text.splitlines()[0] else ","))


def share(scores, emotion, rename={}):  # probability of the emotion among the three
    p = {e: float(scores[rename.get(e, e)]) + 1e-12 for e in triad}
    return p[emotion] / sum(p.values())


if sys.argv[1] in ["make", "best"]:
    arm, test = Path(sys.argv[2]), Path(sys.argv[3])
    items = [r for r in rows(arm / "manifest.csv") if r["emotion"] in triad]
    chosen = []
    if sys.argv[1] == "make":
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 15
        rng = random.Random(0)  # the same items in every test, so tests of different checkpoints stay comparable
        for e in triad:
            c = [r for r in items if r["emotion"] == e]
            rng.shuffle(c)
            chosen += c[:n]
    else:
        heard = {r["path"] for r in rows(sys.argv[4])}
        n = int(sys.argv[5]) if len(sys.argv) > 5 else 5
        shortest = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0
        wavlm = {r["path"]: r for r in rows(arm / "scores_wavlm.csv")}
        e2v = {r["path"]: r for r in rows(arm / "scores_e2v.csv")}
        for e in triad:
            c = [r for r in items if r["emotion"] == e and r["path"] not in heard and float(r["seconds"]) >= shortest]
            print(e, len(c), "clips to choose from")
            c.sort(key=lambda r: share(wavlm[r["path"]], e) + share(e2v[r["path"]], e, {"surprise": "surprised"}), reverse=True)
            chosen += c[:n]
    random.Random(test.name).shuffle(chosen)  # but each test in its own order, nothing to remember from an earlier test
    (test / "clips").mkdir(parents=True, exist_ok=False)  # never overwrite a test that may have been listened to
    key = []
    for i, r in enumerate(chosen, 1):
        x, sr = sf.read(r["path"], dtype="float32")
        x = np.clip(x * 0.05 / max(float(np.sqrt(np.mean(x**2))), 1e-8), -1, 1)  # same RMS for all clips, about -26 dBFS
        sf.write(str(test / "clips" / f"blind_{i:02d}.wav"), x, sr)
        key.append({"clip": i, "emotion": r["emotion"], "path": r["path"]})
    keyfile = test.parent / "blind_keys" / f"{test.name}.csv"
    keyfile.parent.mkdir(exist_ok=True)
    with open(keyfile, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "emotion", "path"])
        w.writeheader()
        w.writerows(key)
    with open(test / "answers.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "answer"])
        w.writerows([[i, ""] for i in range(1, len(chosen) + 1)])
    print(len(chosen), "clips ->", test / "clips", "| fill in", test / "answers.csv", "| solution in", keyfile)

elif sys.argv[1] == "score":
    key = {r["clip"]: r["emotion"] for r in rows(sys.argv[2])}
    given = {r["clip"]: r["answer"].strip().lower() for r in rows(sys.argv[3]) if r["answer"].strip()}
    norm = {"a": "angry", "h": "happy", "s": "surprise", "surprised": "surprise"}
    given = {c: norm.get(a, a) for c, a in given.items()}
    assert all(a in triad for a in given.values()), set(given.values())
    k, n = sum(given[c] == key[c] for c in given), len(given)
    p = sum(math.comb(n, i) * (1 / 3) ** i * (2 / 3) ** (n - i) for i in range(k, n + 1))  # one-sided against guessing
    print(f"{k} of {n} correct ({k / n:.2f}, chance 0.33), p = {p:.4f}")
    for e in triad:
        mine = [c for c in given if key[c] == e]
        print(f"  {e:8s} heard as:", {a: sum(given[c] == a for c in mine) for a in triad})
