import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

# compares the arms of the emotion evaluation (experiments/emotion_a/eval_emotions_generate.py) item by item, for each judge,
# always over all five emotions
#   5-way:    every item decided among the five emotions (chance 0.2), Wilson 95 % interval, recall per emotion
#   pairs:    same speaker and sentence, emotions a and b: does the a item get more "a" than the b item, and vice versa
#             (chance 0.5); compares the raw scores, just as a reward would; mean over the ten pairs, each pair below
#   vs ref:   McNemar on the 5-way hits, b = only this arm right, c = only the reference right
#   rel:      (5-way - 0.2) / (vocoded - 0.2), the share of what the vocoded recordings reach
# usage: python eval_emotions_report.py <evaluation dir> [reference arm, e.g. pilotB_13095]
ev, ref = Path(sys.argv[1]), (sys.argv[2] if len(sys.argv) > 2 else None)
emotions = ["neutral", "angry", "happy", "sad", "surprise"]
judges = {"e2v": ("scores_e2v.csv", {"surprise": "surprised"}), "wavlm": ("scores_wavlm.csv", {}), "llm": ("scores_llm.csv", {})}


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def mcnemar(b, c):  # exact two-sided
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def analyse(items, scores, rename):
    P = np.array([[float(scores[it["path"]][rename.get(e, e)]) for e in emotions] for it in items])
    y = np.array([emotions.index(it["emotion"]) for it in items])
    hit = P.argmax(1) == y
    r = {"n": len(items), "5-way": float(hit.mean()), "5-way_ci": wilson(int(hit.sum()), len(hit))}
    r["recall"] = {e: float(hit[y == i].mean()) for i, e in enumerate(emotions)}
    r["confusion"] = [[int(((y == a) & (P.argmax(1) == b)).sum()) for b in range(5)] for a in range(5)]  # row intended, column heard
    if rename:  # emotion2vec+: plain argmax over all 9 classes, as reported in EMORL-TTS
        full = [max((k for k in scores[it["path"]] if k != "path"), key=lambda k: float(scores[it["path"]][k])) for it in items]
        r["9-way_recall"] = {e: float(np.mean([f == rename.get(e, e) for f, it in zip(full, items) if it["emotion"] == e])) for e in emotions}
    keys = [(it["speaker"], it["sentence"], it["emotion"]) for it in items]
    r["hits"], p = dict(zip(keys, hit.tolist())), dict(zip(keys, P))
    r["pairs"] = {}
    for ia, a in enumerate(emotions):
        for ib in range(ia + 1, len(emotions)):
            b, wins = emotions[ib], []
            for (s, n, e), pa in p.items():
                if e == a and (s, n, b) in p:
                    wins += [pa[ia] > p[(s, n, b)][ia], p[(s, n, b)][ib] > pa[ib]]
            r["pairs"][f"{a}-{b}"] = float(np.mean(wins)) if wins else float("nan")
    r["pairs_mean"] = float(np.mean(list(r["pairs"].values())))
    # the items are not independent (30 per speaker), so the spread over speakers matters more than the interval
    r["per_speaker"] = {s: float(np.mean([h for k, h in r["hits"].items() if k[0] == s])) for s in sorted({k[0] for k in keys})}
    return r


report = {}
arms = sorted(p.name for p in ev.iterdir() if (p / "manifest.csv").exists() and read(p / "manifest.csv"))
kind = {arm: read(ev / arm / "manifest.csv")[0]["arm"] for arm in arms}  # real, vocoded, prompt, ckpt or novec
for judge, (fname, rename) in judges.items():
    res = {}
    for arm in arms:
        if (ev / arm / fname).exists():
            scores = {r["path"]: r for r in read(ev / arm / fname)}
            items = [it for it in read(ev / arm / "manifest.csv") if it["path"] in scores]
            res[arm] = analyse(items, scores, rename)
    if not res:
        continue
    if ref and ref not in res:
        print(f"\nreference arm {ref} not scored by {judge}, no paired test")
    vocoded = next((a for a in res if kind[a] == "vocoded"), None)
    top = res[vocoded]["5-way"] - 0.2 if vocoded else 0.0
    print(f"\n== judge {judge}  (5-way chance 0.20, pairs chance 0.50)")
    print(f"{'arm':20s} {'n':>4s}  {'5-way [95% CI]':20s}" + "".join(f"{e[:5]:6s}" for e in emotions) + f"| pairs | rel   | vs {ref}: b  c  p")
    for arm, r in res.items():
        rel = (r["5-way"] - 0.2) / top if top > 0 else float("nan")
        ci = f"{r['5-way']:.2f} [{r['5-way_ci'][0]:.2f}, {r['5-way_ci'][1]:.2f}]"
        line = f"{arm:20s} {r['n']:4d}  {ci:20s}" + "".join(f"{r['recall'][e]:<6.2f}" for e in emotions)
        line += f"| {r['pairs_mean']:.2f}  | {rel:5.2f} |"
        if ref in res and arm != ref:
            keys = res[ref]["hits"].keys() & r["hits"].keys()
            b = sum(r["hits"][k] and not res[ref]["hits"][k] for k in keys)
            c = sum(res[ref]["hits"][k] and not r["hits"][k] for k in keys)
            line += f" {b:3d} {c:3d} {mcnemar(b, c):.3f}"
            r["vs_ref"] = {"ref": ref, "b": b, "c": c, "p": mcnemar(b, c)}
        r["rel_to_vocoded"] = rel
        print(line)
        print(f"{'':27s}pairs: " + "  ".join(f"{k[:3]}-{k.split('-')[1][:3]} {v:.2f}" for k, v in r["pairs"].items()))
        print(f"{'':27s}per speaker: " + "  ".join(f"{s} {v:.2f}" for s, v in r["per_speaker"].items()))
    report[judge] = {arm: {k: v for k, v in r.items() if k != "hits"} for arm, r in res.items()}
ceiling = ev / "wavlm_judge_ceiling.json"
if ceiling.exists():
    report["wavlm_ceiling_5way"] = json.loads(ceiling.read_text(encoding="utf-8"))["5-way"]
    print(f"\nwavlm judge on real recordings of unseen speakers: 5-way {report['wavlm_ceiling_5way']:.3f}")
with open(ev / "report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=1)
print("\nreport ->", ev / "report.json")
