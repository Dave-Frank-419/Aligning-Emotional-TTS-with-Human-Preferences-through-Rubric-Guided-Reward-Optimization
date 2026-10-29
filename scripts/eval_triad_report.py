import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

# compares the arms of the triad evaluation (experiments/emotion_a/eval_triad_generate.py) item by item, for each judge
#   triad:    angry/happy/surprise items, decided among these three classes only (chance 1/3), Wilson 95 % interval
#   pairs:    same speaker and sentence, does the angry item get more "angry" than the happy item, and vice versa (chance 0.5)
#   vs ref:   McNemar on the triad items, b = only this arm right, c = only the reference right
#   rel:      (triad - 1/3) / (vocoded - 1/3), the share of what the vocoded recordings reach
#   S-N:      control, the same pairwise test for sad vs neutral
# usage: python eval_triad_report.py <evaluation dir> [reference arm, e.g. pilotB_13095]
ev, ref = Path(sys.argv[1]), (sys.argv[2] if len(sys.argv) > 2 else None)
emotions = ["neutral", "angry", "happy", "sad", "surprise"]
triad = ["angry", "happy", "surprise"]
judges = {"e2v": ("scores_e2v.csv", {"surprise": "surprised"}), "wavlm": ("scores_wavlm.csv", {})}


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
    idx = [emotions.index(e) for e in triad]
    t = np.isin(y, idx)
    P3 = (P[t][:, idx] + 1e-12) / (P[t][:, idx] + 1e-12).sum(1, keepdims=True)  # a judge sure of sad can leave all three at 0
    y3 = np.array([idx.index(v) for v in y[t]])
    hit = P3.argmax(1) == y3
    r = {"n": int(t.sum()), "triad": float(hit.mean()), "triad_ci": wilson(int(hit.sum()), len(hit))}
    r["recall"] = {e: float(hit[y3 == i].mean()) for i, e in enumerate(triad)}
    r["confusion"] = [[int(((y3 == a) & (P3.argmax(1) == b)).sum()) for b in range(3)] for a in range(3)]
    sad = emotions.index("sad")  # control: pilot B already renders sad
    r["sad_recall_5way"] = float((P.argmax(1)[y == sad] == sad).mean())
    r["5-way"] = float((P.argmax(1) == y).mean())
    if rename:  # emotion2vec+: plain argmax over all 9 classes, as reported in EMORL-TTS
        full = [max((k for k in scores[it["path"]] if k != "path"), key=lambda k: float(scores[it["path"]][k])) for it in items]
        r["9-way_recall"] = {e: float(np.mean([f == rename.get(e, e) for f, it in zip(full, items) if it["emotion"] == e])) for e in emotions}
    tri_items = [it for it in items if it["emotion"] in triad]
    r["hits"] = {(it["speaker"], it["sentence"], it["emotion"]): bool(h) for it, h in zip(tri_items, hit)}
    p3 = {(it["speaker"], it["sentence"], it["emotion"]): p for it, p in zip(tri_items, P3)}
    r["pairs"] = {}
    for a, b in [("angry", "happy"), ("angry", "surprise"), ("happy", "surprise")]:
        ia, ib = triad.index(a), triad.index(b)
        wins = []
        for (s, n, e), pa in p3.items():
            if e == a and (s, n, b) in p3:
                pb = p3[(s, n, b)]
                wins += [pa[ia] > pb[ia], pb[ib] > pa[ib]]
        r["pairs"][f"{a[0].upper()}-{b[0].upper()}"] = float(np.mean(wins)) if wins else float("nan")
    # control, same speaker and sentence: is the sad item more "sad" and the neutral item more "neutral" (chance 0.5);
    # stricter than sad recall, which a judge that calls everything sad would pass
    neu, p5, wins = emotions.index("neutral"), {(it["speaker"], it["sentence"], it["emotion"]): p for it, p in zip(items, P)}, []
    for (s, n, e), ps in p5.items():
        if e == "sad" and (s, n, "neutral") in p5:
            wins += [ps[sad] > p5[(s, n, "neutral")][sad], p5[(s, n, "neutral")][neu] > ps[neu]]
    r["sad-neutral_pair"] = float(np.mean(wins)) if wins else float("nan")
    # the items are not independent (18 per speaker), so the spread over speakers matters more than the interval
    r["per_speaker"] = {s: float(np.mean([h for k, h in r["hits"].items() if k[0] == s])) for s in sorted({k[0] for k in r["hits"]})}
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
    print(f"\n== judge {judge}  (triad chance 0.33, pairs chance 0.50)")
    print(f"{'arm':20s} {'n':>4s}  triad [95% CI]       angry happy surpr | A-H  A-S  H-S  | S-N  5-way | rel  | vs {ref}: b  c  p")
    for arm, r in res.items():
        top = res[vocoded]["triad"] - 1 / 3 if vocoded else 0.0
        rel = (r["triad"] - 1 / 3) / top if top > 0 else float("nan")
        line = f"{arm:20s} {r['n']:4d}  {r['triad']:.2f} [{r['triad_ci'][0]:.2f}, {r['triad_ci'][1]:.2f}]   "
        line += " ".join(f"{r['recall'][e]:.2f} " for e in triad) + " | "
        line += " ".join(f"{v:.2f}" for v in r["pairs"].values()) + " | "
        line += f"{r['sad-neutral_pair']:.2f} {r['5-way']:.2f}  | {rel:.2f} |"
        if ref in res and arm != ref:
            keys = res[ref]["hits"].keys() & r["hits"].keys()
            b = sum(r["hits"][k] and not res[ref]["hits"][k] for k in keys)
            c = sum(res[ref]["hits"][k] and not r["hits"][k] for k in keys)
            line += f" {b:3d} {c:3d} {mcnemar(b, c):.3f}"
            r["vs_ref"] = {"ref": ref, "b": b, "c": c, "p": mcnemar(b, c)}
        r["rel_to_vocoded"] = rel
        print(line)
        print(f"{'':26s}per speaker: " + "  ".join(f"{s} {v:.2f}" for s, v in r["per_speaker"].items()))
    report[judge] = {arm: {k: v for k, v in r.items() if k != "hits"} for arm, r in res.items()}
ceiling = ev / "wavlm_judge_ceiling.json"
if ceiling.exists():
    report["wavlm_ceiling_real_unseen_speakers"] = json.loads(ceiling.read_text(encoding="utf-8"))
    print("\nwavlm judge on real recordings of unseen speakers:", report["wavlm_ceiling_real_unseen_speakers"])
with open(ev / "report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=1)
print("\nreport ->", ev / "report.json")
