import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# blind listening test over all five emotions on one arm of the emotion evaluation
# make:  python make_blind_test.py make <arm dir> <test dir> [clips per speaker and emotion, default 2]
#        every speaker equally often per emotion (6 speakers x 2 x 5 emotions = 60 clips); <test dir>/clips/blind_NN.wav at
#        the same loudness and answers.csv to fill in; only this folder is copied to listen,
#        the solution goes to <parent>/blind_keys/<test name>.csv
# best:  python make_blind_test.py best <arm dir> <test dir> <key of an earlier test> [clips per emotion, default 5] [shortest, default 2 s]
#        as make, but per emotion the clips both judges rate highest, as a reward would pick them; clips of the earlier test
#        and very short ones are left out. Shows whether the arm can render an emotion at all, not whether it does so reliably
# score: python make_blind_test.py score <key.csv> <filled answers.csv>
#        answer per clip: neutral, angry, happy, sad or surprise (short: n, a, h, sa, su)
# chance is 1/5: with 60 clips, 18 or more correct is above chance (one-sided p < 0.05); with 25 clips, 9 or more
emotions = ["neutral", "angry", "happy", "sad", "surprise"]


def rows(path):  # also as saved by Excel: BOM and ';' as separator
    with open(path, encoding="utf-8-sig", newline="") as f:
        text = f.read()
    return list(csv.DictReader(text.splitlines(), delimiter=";" if ";" in text.splitlines()[0] else ","))


def share(scores, emotion, rename={}):  # probability of the emotion among the five
    p = {e: float(scores[rename.get(e, e)]) + 1e-12 for e in emotions}
    return p[emotion] / sum(p.values())


if sys.argv[1] in ["make", "best"]:
    arm, test = Path(sys.argv[2]), Path(sys.argv[3])
    items = rows(arm / "manifest.csv")
    chosen = []
    if sys.argv[1] == "make":
        k = int(sys.argv[4]) if len(sys.argv) > 4 else 2
        rng = random.Random(0)  # the same items in every test, so tests of different checkpoints stay comparable
        for e in emotions:
            for s in sorted({r["speaker"] for r in items}):
                c = [r for r in items if r["emotion"] == e and r["speaker"] == s]
                rng.shuffle(c)
                chosen += c[:k]
    else:
        heard = {r["path"] for r in rows(sys.argv[4])}
        n = int(sys.argv[5]) if len(sys.argv) > 5 else 5
        shortest = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0
        wavlm = {r["path"]: r for r in rows(arm / "scores_wavlm.csv")}
        e2v = {r["path"]: r for r in rows(arm / "scores_e2v.csv")}
        for e in emotions:
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
        key.append({"clip": i, "emotion": r["emotion"], "path": r["path"], "seconds": r["seconds"]})
    keyfile = test.parent / "blind_keys" / f"{test.name}.csv"
    keyfile.parent.mkdir(exist_ok=True)
    with open(keyfile, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "emotion", "path", "seconds"])
        w.writeheader()
        w.writerows(key)
    with open(test / "answers.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "answer"])
        w.writerows([[i, ""] for i in range(1, len(chosen) + 1)])
    print(len(chosen), "clips ->", test / "clips", "| fill in", test / "answers.csv", "| solution in", keyfile)

elif sys.argv[1] == "score":
    key = {r["clip"]: r for r in rows(sys.argv[2])}
    labels = [e for e in emotions if e in {r["emotion"] for r in key.values()}]  # earlier tests used only three emotions
    norm = {"n": "neutral", "a": "angry", "h": "happy", "sa": "sad", "su": "surprise", "surprised": "surprise"}
    if "sad" not in labels:  # in the earlier tests s meant surprise
        norm["s"] = "surprise"
    given = {}
    for r in rows(sys.argv[3]):
        if r["answer"].strip():
            given[r["clip"]] = norm.get(r["answer"].strip().lower(), r["answer"].strip().lower())
    assert all(a in labels for a in given.values()), f"unknown answers {set(given.values()) - set(labels)} ('s' is ambiguous: sa or su)"
    hit = {c: given[c] == key[c]["emotion"] for c in given}
    k, n, chance = sum(hit.values()), len(hit), 1 / len(labels)
    p = sum(math.comb(n, i) * chance**i * (1 - chance) ** (n - i) for i in range(k, n + 1))  # one-sided against guessing
    print(f"{k} of {n} correct ({k / n:.2f}, chance {chance:.2f}), p = {p:.4f}")
    for e in labels:
        mine = [c for c in given if key[c]["emotion"] == e]
        print(f"  {e:8s} heard as:", {a: sum(given[c] == a for c in mine) for a in labels})
    if all(r.get("seconds") for r in key.values()):  # keys of earlier tests have no length
        long = [c for c in given if float(key[c]["seconds"]) >= 2.0]
        print(f"  clips of 2 s or more: {sum(hit[c] for c in long)} of {len(long)}")
