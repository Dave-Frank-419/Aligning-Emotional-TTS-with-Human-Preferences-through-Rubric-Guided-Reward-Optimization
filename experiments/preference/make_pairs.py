"""Pairs of the preference stage from a pool of the model's own samples (generate.py), and a control of random pairs.

Per clip: N_j, N_r = logit of (1 - probability of neutral) by the second judge (scripts/score_wavlm_judge.py,
scores_wavlm.csv) and by the reward's classifier (scripts/score_audiojev.py with the adapter lora4,
scores_jev_lora4.csv): the intensity of any form; J, C = logit of the probability of the intended emotion by the same
two: its identity; each standardised over the emotion's clips of the pool. The word errors (score_words.py, wer.csv)
and the pitch against the speaker's real range (score_pitch.py, pitch.csv).
Inside an item (one speaker, sentence and emotion, several seeds) a pair (winner, loser) qualifies if
    angry, happy, surprise   both judges give the winner the higher N, neither gives it a lower J or C
    sad                      both judges give the winner the higher J and C, neither gives it a lower N (the intensity
                             of any form agreed least with the ear for sad)
    always                   the winner has no more word errors than the loser, and its zP is at most --zmax (2: a clip
                             far above the speaker's real pitch for the emotion is what the listener ticked as overdone)
    margin                   the smaller of the two standardised differences (of N, or of J and C for sad)
Per item the qualifying pair with the largest margin is kept if the margin is at least --margin. The rule was checked
against the ear on the model's own samples (listening pages V1, 2026-11-22: the reward's winner heard as more emotional
in 70 of 96 pairs). An item is held out (split val, never trained on) if its sentence number is divisible by --holdout.
The control: for the same items two seeds drawn at random (random.Random(2211), in the order of the items), whatever
the reward says; same columns, kind "random".
Output in <out folder>: pairs_<emotion>.csv and pairs_random_<emotion>.csv (the columns train_dpo.py reads, emo_vec =
the vector the pool was generated with) and pairs_all.csv (every item's verdict).
usage: python make_pairs.py <pool folder with s<offset> subfolders> <out folder> [--offsets 30,31,32,33,34,35]
           [--margin 0.3] [--holdout 10] [--zmax 2 | --no-guard] [--esd /data/chi-gpu1/ge96xah/data/ESD]
"""
import argparse
import csv
import os
import random
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

EMOS, FOUR = ["neutral", "angry", "happy", "sad", "surprise"], ["angry", "happy", "sad", "surprise"]
p = argparse.ArgumentParser()
p.add_argument("pool")
p.add_argument("out")
p.add_argument("--offsets", default="30,31,32,33,34,35")
p.add_argument("--margin", type=float, default=0.3)
p.add_argument("--holdout", type=int, default=10)
p.add_argument("--zmax", type=float, default=2.0)
p.add_argument("--no-guard", action="store_true")  # without the pitch guard (the pilot's rule of 2026-11-22)
p.add_argument("--esd", default="/data/chi-gpu1/ge96xah/data/ESD")
args = p.parse_args()
POOL, OUT, ESD = Path(args.pool), Path(args.out), Path(args.esd)
offsets = [int(k) for k in args.offsets.split(",")]
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
norm = lambda p: str(p).replace("\\", "/").lower()

parts = []
for k in offsets:
    m = pd.read_csv(POOL / f"s{k}" / "manifest.csv")
    m["key"], m["offset"] = m["path"].map(norm), k
    for name, f in (("j", "scores_wavlm.csv"), ("r", "scores_jev_lora4.csv")):
        s = pd.read_csv(POOL / f"s{k}" / f)
        s["key"] = s["path"].map(norm)
        m = m.merge(s[["key"] + EMOS].rename(columns={e: f"{name}_{e}" for e in EMOS}), on="key", how="left")
    w = pd.read_csv(POOL / f"s{k}" / "wer.csv")
    w["key"] = w["path"].map(norm)
    m = m.merge(w[["key", "errors", "words"]], on="key", how="left")
    if not args.no_guard:
        z = pd.read_csv(POOL / f"s{k}" / "pitch.csv")
        z["key"] = z["path"].map(norm)
        m = m.merge(z[["key", "zP"]], on="key", how="left")
    parts.append(m)
d = pd.concat(parts, ignore_index=True)
need = ["j_neutral", "r_neutral", "errors"] + ([] if args.no_guard else ["zP"])
assert d[need].notna().all().all(), "the pool is not fully measured (both judges, words" + ("" if args.no_guard else ", pitch") + ")"
d["N_j"], d["N_r"] = logit(1 - d.j_neutral), logit(1 - d.r_neutral)
d["J"] = [logit(r[f"j_{r['emotion']}"]) for _, r in d.iterrows()]
d["C"] = [logit(r[f"r_{r['emotion']}"]) for _, r in d.iterrows()]
for c in ("N_j", "N_r", "J", "C"):
    d["z_" + c] = d.groupby("emotion")[c].transform(lambda v: (v - v.mean()) / v.std())
assert d.groupby(["speaker", "sentence", "emotion"]).size().eq(len(offsets)).all(), f"not {len(offsets)} seeds for every item"


def text_of(split):  # file name -> transcription
    out = {}
    with open(ESD / "esd_csvs" / f"{split}.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["lang"] == "en":
                out[os.path.basename(r["relative_audio_path"])] = r["transcription"].strip()
    return out


prompt_text = text_of("test")
target_text = {}  # (speaker, sentence) -> the neutral transcription of a training sentence, which the generation speaks
with open(ESD / "esd_csvs" / "train.csv", encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        if r["lang"] == "en" and r["emotion"].lower() == "neutral":
            name = os.path.basename(r["relative_audio_path"])
            target_text[(int(name[:4]), (int(Path(name).stem.split("_")[1]) - 1) % 350)] = r["transcription"].strip()


def row(w, l, kind, margin):
    assert w["prompt"] == l["prompt"] and w["source"] == l["source"] and w["seconds"] == l["seconds"], "the two members of a pair differ in prompt, vector source or length"
    s, n, e = int(w["speaker"]), int(w["sentence"]), w["emotion"]
    vec = POOL / f"s{int(w['offset'])}" / "cond" / f"{s:04d}_{n:03d}_{e}.npy"  # the same vector for every seed of the item
    assert vec.exists(), f"the stored condition is missing: {vec}"
    return dict(speaker=s, sentence=n, emotion=e, split="val" if n % args.holdout == 0 else "train", winner=w["path"], loser=l["path"], seconds=w["seconds"],
                prompt_wav=str(POOL / f"s{int(w['offset'])}" / "prompts" / Path(w["prompt"]).name), prompt_text=prompt_text[Path(w["prompt"]).name],
                text=target_text[(s, n)], source=w["source"], emo_vec=str(vec), kind=kind, margin=round(margin, 4),
                dN_j=round(w["N_j"] - l["N_j"], 3), dN_r=round(w["N_r"] - l["N_r"], 3), dJ=round(w["J"] - l["J"], 3), dC=round(w["C"] - l["C"], 3),
                errors_w=int(w["errors"]), errors_l=int(l["errors"]), offset_w=int(w["offset"]), offset_l=int(l["offset"]),
                **({} if args.no_guard else dict(zP_w=round(w["zP"], 3), zP_l=round(l["zP"], 3))))


rng, verdicts, keep, rand, guarded = random.Random(2211), [], [], [], 0
for (s, n, e), g in d.groupby(["speaker", "sentence", "emotion"]):
    clips = g.sort_values("offset").to_dict("records")
    best, best_unguarded = None, None
    for a, b in combinations(clips, 2):
        for w, l in ((a, b), (b, a)):
            if w["errors"] > l["errors"]:
                continue
            if e == "sad":
                m1, m2 = w["z_J"] - l["z_J"], w["z_C"] - l["z_C"]
                ok = m1 > 0 and m2 > 0 and w["N_j"] >= l["N_j"] and w["N_r"] >= l["N_r"]
            else:
                m1, m2 = w["z_N_j"] - l["z_N_j"], w["z_N_r"] - l["z_N_r"]
                ok = m1 > 0 and m2 > 0 and w["J"] >= l["J"] and w["C"] >= l["C"]
            if ok and (best_unguarded is None or min(m1, m2) > best_unguarded[2]):
                best_unguarded = (w, l, min(m1, m2))
            if ok and (args.no_guard or w["zP"] <= args.zmax) and (best is None or min(m1, m2) > best[2]):
                best = (w, l, min(m1, m2))
    guarded += best_unguarded is not None and best_unguarded[2] >= args.margin and (best is None or best[2] < args.margin)
    verdicts.append(dict(speaker=s, sentence=n, emotion=e, qualifying=best is not None, margin=None if best is None else round(best[2], 4)))
    if best is not None and best[2] >= args.margin:
        keep.append(row(best[0], best[1], "reward", best[2]))
        a, b = rng.sample(clips, 2)  # the control: two seeds of the same item at random
        rand.append(row(a, b, "random", 0.0))
OUT.mkdir(parents=True, exist_ok=True)
v = pd.DataFrame(verdicts)
v.to_csv(OUT / "pairs_all.csv", index=False)
k, r = pd.DataFrame(keep), pd.DataFrame(rand)
for e in FOUR:
    ke, re_ = k[k.emotion == e], r[r.emotion == e]
    ke.to_csv(OUT / f"pairs_{e}.csv", index=False)
    re_.to_csv(OUT / f"pairs_random_{e}.csv", index=False)
    ve = v[v.emotion == e]
    print(f"{e}: {int(ve.qualifying.sum())} of {len(ve)} items have a qualifying pair, {len(ke)} kept with margin >= {args.margin} (train {int((ke.split == 'train').sum())}, "
          f"held out {int((ke.split == 'val').sum())}); margin median {ke.margin.median():.2f}; winner - loser: N_j {ke.dN_j.mean():+.2f}, N_r {ke.dN_r.mean():+.2f}, "
          f"J {ke.dJ.mean():+.2f}, C {ke.dC.mean():+.2f}; word errors winner {ke.errors_w.sum()} / loser {ke.errors_l.sum()}")
print(f"pairs kept: {len(k)}; control pairs on the same items: {len(r)}" + ("" if args.no_guard else f"; items that lost their pair to the pitch guard (zP above {args.zmax:g}): {guarded}"))
