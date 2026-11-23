"""Evaluation of a saved state of the preference stage against a reference, the bars of a state and the stop rules.

Both arms are the evaluation set (six speakers x six evaluation sentences x five emotions) at the same seed offsets,
generated with the same emotion input (generate.py) and level matched (level_match.py); the reference is the start
model. Each clip is paired with the reference's clip of the same item and seed offset. Measures per clip, as in
make_pairs.py: N_j, N_r (intensity of any form, second judge and reward's classifier), J, C (identity of the intended
emotion), word errors (wer.csv), zP (pitch against the speaker's real range, pitch.csv). Intervals: 95 % bootstrap over
the 36 items (an item's clips are averaged first; 2000 draws, numpy seed 0); pooled = the four emotions.
Bars of a state (fixed in the notebook before the large run big1, 2026-11-23):
    E1  size        pooled N_j at least --e1 (1.2, about one strength unit of the emotion input), and pooled N_r with
                    its interval above 0
    E2  identity    pooled J and C not clearly lower (upper end of each interval at least 0)
    E3  neutral     word errors of the neutral clips at most 0.03 above the reference's, upper end at most 0.06
    E4  words       word errors of the four emotions at most 0.03 above the reference's
    E5  overdone    share of the four emotions' clips with zP above --zmax (2) at most 0.10 above the reference's share
    A state that meets E1 to E5 is a candidate for the ear.
Stop rules (written as the file STOP into --run when one fires; the trainer stops before its next update):
    S1  damage      the state fails E3, E4 or E5
    S2  no progress at --step 600 or later, pooled N_j below --s2 (0.45, the gain of the pilot of 2026-11-22)
    S3  one judge   against --prev (the previous state's summary): N_r rises while N_j rises by less than 0.05 and J falls
                    by more than 0.05
usage: python evaluate.py --arm <folder with s2/, s3/> --ref <folder with s2/, s3/> [--offsets 2,3] [--step 600]
           [--prev <summary.json of the previous state>] [--run <run folder>] [--out <summary.json>] [--real <folder of
           the real recordings with manifest.csv and wer.csv>]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

EMOS, FOUR = ["neutral", "angry", "happy", "sad", "surprise"], ["angry", "happy", "sad", "surprise"]
p = argparse.ArgumentParser()
p.add_argument("--arm", required=True)
p.add_argument("--ref", required=True)
p.add_argument("--offsets", default="2,3")
p.add_argument("--step", type=int, default=0)
p.add_argument("--prev", default="")
p.add_argument("--run", default="")
p.add_argument("--out", default="")
p.add_argument("--real", default="")
p.add_argument("--e1", type=float, default=1.2)
p.add_argument("--s2", type=float, default=0.45)
p.add_argument("--zmax", type=float, default=2.0)
args = p.parse_args()
offsets = [int(k) for k in args.offsets.split(",")]
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
key = lambda p: Path(str(p).replace("\\", "/")).name.lower()


def clips(folder, offset):  # one folder of measured clips -> one row per clip
    m = pd.read_csv(folder / "manifest.csv")
    m["key"] = m["path"].map(key)
    for name, f in (("j", "scores_wavlm.csv"), ("r", "scores_jev_lora4.csv")):
        s = pd.read_csv(folder / f)
        s["key"] = s["path"].map(key)
        m = m.merge(s[["key"] + EMOS].rename(columns={e: f"{name}_{e}" for e in EMOS}), on="key", how="left")
    for f, cols in (("wer.csv", ["errors", "words"]), ("pitch.csv", ["zP"])):
        w = pd.read_csv(folder / f)
        w["key"] = w["path"].map(key)
        m = m.merge(w[["key"] + cols], on="key", how="left")
    assert m[["j_neutral", "r_neutral", "errors", "zP"]].notna().all().all(), f"{folder}: a clip lacks a judge's score, its word count or its pitch"
    m["offset"], m["speaker"], m["sentence"] = offset, m["speaker"].astype(int), m["sentence"].astype(int)
    m["N_j"], m["N_r"] = logit(1 - m.j_neutral), logit(1 - m.r_neutral)
    m["J"] = [logit(r[f"j_{r['emotion']}"]) for _, r in m.iterrows()]
    m["C"] = [logit(r[f"r_{r['emotion']}"]) for _, r in m.iterrows()]
    m["named"] = m[[f"j_{e}" for e in EMOS]].values.argmax(1) == m["emotion"].map(EMOS.index).values
    m["level_db"] = [20 * np.log10(max(float(np.sqrt(np.mean(sf.read(str(q).replace("\\", "/"), dtype="float64")[0] ** 2))), 1e-10)) for q in m["path"]]
    return m


arm = pd.concat([clips(Path(args.arm) / f"s{k}", k) for k in offsets], ignore_index=True)
ref = pd.concat([clips(Path(args.ref) / f"s{k}", k) for k in offsets], ignore_index=True)
ITEM = ["speaker", "sentence", "emotion", "offset"]
pr = arm.merge(ref, on=ITEM, suffixes=("", "_ref"))
assert len(pr) == len(arm) == len(ref), f"the arms do not pair clip by clip: {len(arm)}, {len(ref)}, {len(pr)} pairs"
assert (pr["seconds"] == pr["seconds_ref"]).all(), "a clip and its reference differ in length: not the same items"
rng = np.random.default_rng(0)


def diff(x, col):  # mean difference arm - reference with a 95 % bootstrap interval over the items
    d = (x[col] - x[col + "_ref"]).groupby([x.speaker, x.sentence]).mean().values
    boot = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def wer_diff(x):  # word errors per word, arm and reference, difference with an interval over the items
    g = x.groupby([x.speaker, x.sentence])[["errors", "words", "errors_ref", "words_ref"]].sum().values
    idx = rng.integers(0, len(g), (2000, len(g)))
    boot = g[idx, 0].sum(1) / g[idx, 1].sum(1) - g[idx, 2].sum(1) / g[idx, 3].sum(1)
    a, r = g[:, 0].sum() / g[:, 1].sum(), g[:, 2].sum() / g[:, 3].sum()
    return float(a), float(r), float(a - r), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


f = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
S = dict(arm=args.arm, ref=args.ref, step=args.step, offsets=offsets, per_emotion={})
print(f"{args.arm} against {args.ref}, seed offsets {offsets}: {len(pr)} clip pairs ({pr.groupby(['speaker', 'sentence']).ngroups} items)")
print("\nPer emotion, difference arm - reference (95 % interval over the items):")
for e in FOUR:
    x = pr[pr.emotion == e]
    r = {c: diff(x, c) for c in ("N_j", "N_r", "J", "C")}
    S["per_emotion"][e] = {c: v[0] for c, v in r.items()}
    print(f"  {e:9s} " + "  ".join(f"{c} {f(v)}" for c, v in r.items()))
x4 = pr[pr.emotion.isin(FOUR)]
pooled = {c: diff(x4, c) for c in ("N_j", "N_r", "J", "C")}
S["pooled"] = {c: dict(mean=v[0], lo=v[1], hi=v[2]) for c, v in pooled.items()}
print(f"  {'pooled':9s} " + "  ".join(f"{c} {f(v)}" for c, v in pooled.items()) + f"  ({len(x4)} clip pairs)")
xn = pr[pr.emotion == "neutral"]
wn = wer_diff(xn)
w4 = wer_diff(x4)
S["neutral"] = dict(words=wn[0], words_ref=wn[1], diff=wn[2], lo=wn[3], hi=wn[4], N_j=diff(xn, "N_j")[0], N_r=diff(xn, "N_r")[0], level_db=float((xn.level_db - xn.level_db_ref).mean()))
S["words_four"] = dict(words=w4[0], words_ref=w4[1], diff=w4[2], lo=w4[3], hi=w4[4])
share, share_ref = float((x4.zP > args.zmax).mean()), float((x4.zP_ref > args.zmax).mean())
S["overdone"] = dict(share=share, share_ref=share_ref, zP=float(x4.zP.mean()), zP_ref=float(x4.zP_ref.mean()))
S["five_way"] = dict(four=int(x4.named.sum()), four_ref=int(x4.named_ref.sum()), all=int(pr.named.sum()), all_ref=int(pr.named_ref.sum()), clips=len(pr))
print(f"\nNeutral: word errors {wn[0]:.3f} against {wn[1]:.3f}, difference {wn[2]:+.3f} [{wn[3]:+.3f}, {wn[4]:+.3f}]; N_j {S['neutral']['N_j']:+.3f}, N_r {S['neutral']['N_r']:+.3f}; "
      f"level {S['neutral']['level_db']:+.2f} dB (neutral clips are not level matched)")
print(f"Word errors of the four emotions: {w4[0]:.3f} against {w4[1]:.3f}, difference {w4[2]:+.3f} [{w4[3]:+.3f}, {w4[4]:+.3f}]")
for e in EMOS:
    a = wer_diff(pr[pr.emotion == e])
    print(f"   {e:9s} {a[0]:.3f} against {a[1]:.3f}")
if args.real:
    real = pd.read_csv(Path(args.real) / "manifest.csv")
    rw = pd.read_csv(Path(args.real) / "wer.csv")
    real["key"], rw["key"] = real["path"].map(key), rw["path"].map(key)
    real = real.merge(rw[["key", "errors", "words"]], on="key")
    print("   real recordings of the same items: " + ", ".join(f"{e} {real[real.emotion == e].errors.sum() / real[real.emotion == e].words.sum():.3f}" for e in EMOS))
print(f"Overdone: clips of the four emotions with zP above {args.zmax:g}: {share:.3f} against {share_ref:.3f}; mean zP {S['overdone']['zP']:+.2f} against {S['overdone']['zP_ref']:+.2f}")
print(f"Five-way (second judge names the clip by its emotion): the four {S['five_way']['four']} against {S['five_way']['four_ref']} of {len(x4)}; all {S['five_way']['all']} against {S['five_way']['all_ref']} of {len(pr)}")

E = dict(E1=pooled["N_j"][0] >= args.e1 and pooled["N_r"][1] > 0, E2=pooled["J"][2] >= 0 and pooled["C"][2] >= 0, E3=wn[2] <= 0.03 and wn[4] <= 0.06,
         E4=w4[2] <= 0.03, E5=share <= share_ref + 0.10)
S["bars"], S["candidate"] = E, all(E.values())
stops = []
if not (E["E3"] and E["E4"] and E["E5"]):
    stops.append("S1 damage: " + ", ".join(k for k in ("E3", "E4", "E5") if not E[k]) + " failed")
if args.step >= 600 and pooled["N_j"][0] < args.s2:
    stops.append(f"S2 no progress: pooled N_j {pooled['N_j'][0]:+.3f} below {args.s2} after {args.step} updates")
if args.prev:
    q = json.load(open(args.prev, encoding="utf-8"))["pooled"]
    dr, dj, dJ = pooled["N_r"][0] - q["N_r"]["mean"], pooled["N_j"][0] - q["N_j"]["mean"], pooled["J"][0] - q["J"]["mean"]
    if dr > 0 and dj < 0.05 and dJ < -0.05:
        stops.append(f"S3 one judge: N_r {dr:+.3f}, N_j {dj:+.3f}, J {dJ:+.3f} since the previous state")
S["stop"] = stops
print("\nBars: " + ", ".join(f"{k} {'met' if v else 'NOT met'}" for k, v in E.items()) + f"; candidate for the ear: {'YES' if S['candidate'] else 'no'}")
print("Stop rules: " + ("; ".join(stops) if stops else "none fires"))
if args.out:
    Path(args.out).write_text(json.dumps(S, indent=1), encoding="utf-8")
if stops and args.run:
    (Path(args.run) / "STOP").write_text(f"after update {args.step}: " + "; ".join(stops) + "\n", encoding="utf-8")
    print(f"written {Path(args.run) / 'STOP'}")
