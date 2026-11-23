"""Table of the start model's own neutral clips for the anchor of the preference stage (train_dpo.py --anchor).

The anchor holds the velocity of the policy at the start model's on these clips, so that neutral speech is kept while
the pairs move the emotions (it kept neutral, the level and the word errors in the run e4lman1000 of 2026-11-18). The
clips come from generate.py with the start model on training sentences, neutral only, the source's vector as it is, at
the seed offsets given (one subfolder s<offset> each; the first pilot used the 90 positions 0, 3, ..., 267 and the
offsets 0 to 7: 4,320 clips). A row holds everything needed to lay a clip out as at its generation: the prompt in
front, the clip behind it, the text, the vector source. Clips of sentences whose number is divisible by --holdout are
marked val: never trained on, the drift monitor reads those of the first offset.
usage: python make_anchor_table.py <folder with s<offset> subfolders> <out table.csv> --ckpt <the start weights as named
           in the manifests> [--offsets 0,1,2,3,4,5,6,7] [--holdout 10] [--esd /data/chi-gpu1/ge96xah/data/ESD]
"""
import argparse
import csv
import os
from pathlib import Path

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("folder")
p.add_argument("out")
p.add_argument("--ckpt", required=True)
p.add_argument("--offsets", default="0,1,2,3,4,5,6,7")
p.add_argument("--holdout", type=int, default=10)
p.add_argument("--esd", default="/data/chi-gpu1/ge96xah/data/ESD")
args = p.parse_args()
CAND, ESD = Path(args.folder), Path(args.esd)
slash = lambda s: str(s).replace("\\", "/")


def text_of(split, emotion=None):  # file name -> transcription
    out = {}
    with open(ESD / "esd_csvs" / f"{split}.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["lang"] == "en" and (emotion is None or r["emotion"].lower() == emotion):
                out[os.path.basename(r["relative_audio_path"])] = r["transcription"].strip()
    return out


prompt_text = text_of("test")
target_text = {(int(n[:4]), (int(Path(n).stem.split("_")[1]) - 1) % 350): t for n, t in text_of("train", "neutral").items()}  # the generation speaks the neutral transcription
rows = []
for k in [int(x) for x in args.offsets.split(",")]:
    m = pd.read_csv(CAND / f"s{k}" / "manifest.csv")
    m = m[m.emotion == "neutral"]
    assert (m.checkpoint.map(slash) == slash(args.ckpt)).all() and (m.seed // 1000 == k).all(), f"offset {k}: not clips of the start model at this seed offset"
    for r in m.itertuples():
        rows.append(dict(speaker=r.speaker, sentence=r.sentence, emotion="neutral", split="val" if r.sentence % args.holdout == 0 else "train", offset=k, clip=r.path,
                         seconds=r.seconds, prompt_wav=str(CAND / f"s{k}" / "prompts" / Path(r.prompt).name), prompt_text=prompt_text[Path(r.prompt).name],
                         text=target_text[(int(r.speaker), int(r.sentence))], source=r.source))
o = pd.DataFrame(rows)
assert not o.duplicated(["speaker", "sentence", "offset"]).any() and all(os.path.exists(q) for q in o.prompt_wav.unique())
o.to_csv(args.out, index=False)
print(f"{args.out}: {len(o)} clips of the start model ({int((o.split == 'train').sum())} on training sentences, {int((o.split == 'val').sum())} on held-out sentences), "
      f"{o.groupby(['speaker', 'sentence']).ngroups} items x {o.offset.nunique()} seed offsets")
