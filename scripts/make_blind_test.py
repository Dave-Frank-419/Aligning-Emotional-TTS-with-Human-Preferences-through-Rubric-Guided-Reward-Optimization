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
# score: python make_blind_test.py score <key.csv> <filled answers.csv>   (answer per clip: angry, happy or surprise)
# with 45 clips, 21 or more correct is above chance (one-sided p < 0.05)
triad = ["angry", "happy", "surprise"]


def rows(path):  # also as saved by Excel: BOM and ';' as separator
    with open(path, encoding="utf-8-sig", newline="") as f:
        text = f.read()
    return list(csv.DictReader(text.splitlines(), delimiter=";" if ";" in text.splitlines()[0] else ","))


if sys.argv[1] == "make":
    arm, test = Path(sys.argv[2]), Path(sys.argv[3])
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 15
    with open(arm / "manifest.csv", encoding="utf-8", newline="") as f:
        items = [r for r in csv.DictReader(f) if r["emotion"] in triad]
    rng = random.Random(0)  # the same items in every test, so tests of different checkpoints stay comparable
    chosen = []
    for e in triad:
        c = [r for r in items if r["emotion"] == e]
        rng.shuffle(c)
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
