import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

# cross-emotion prompt pairs for ESD, read by F5-TTS dataset.py from pairs.json
# for every training utterance (target):
#   prompt: same speaker, emotion drawn uniformly from all five, other sentence
#           -> in 4 of 5 cases the prompt carries a different emotion, so only the embedding reliably gives the target emotion
#   emotion sources: 10 other utterances of the same speaker and emotion, one drawn per sample (cross-utterance embedding)
# usage: python make_xprompt_pairs.py <metadata_train.csv> <F5 data dir> <new F5 data dir>
meta, src, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
emotions = ["Neutral", "Angry", "Happy", "Sad", "Surprise"]

utts = []  # (audio_path as stored in raw.arrow, speaker, emotion, sentence number)
with open(meta, encoding="utf-8-sig", newline="") as f:
    for row in list(csv.reader(f, delimiter="|"))[1:]:
        path = Path(row[0].strip()).expanduser().as_posix()  # same normalisation as prepare_csv_wavs.py
        parts = Path(path).parts  # .../ESD/<speaker>/<Emotion>/<split>/<speaker>_<id>.wav
        assert parts[-3] in emotions and parts[-4] == Path(path).stem[:4], path
        sentence = (int(Path(path).stem.split("_")[1]) - 1) % 350  # ESD is parallel: same number = same sentence
        utts.append((path, parts[-4], parts[-3], sentence))

groups = {}  # (speaker, emotion) -> utterances
for u in utts:
    groups.setdefault((u[1], u[2]), []).append(u)

rng = random.Random(0)
pairs = {}
for path, spk, emo, sentence in utts:
    prompt_emo = rng.choice(emotions)
    prompts = [u[0] for u in groups[(spk, prompt_emo)] if u[3] != sentence]
    sources = [u[0] for u in groups[(spk, emo)] if u[0] != path]
    pairs[path] = [rng.choice(prompts), rng.sample(sources, min(10, len(sources)))]

info = {u[0]: u for u in utts}
assert all(info[p][1] == info[t][1] and info[p][3] != info[t][3] for t, (p, _) in pairs.items())
assert all(info[s][1:3] == info[t][1:3] and s != t for t, (_, ss) in pairs.items() for s in ss)
missing = [s for _, ss in pairs.values() for s in ss if not os.path.exists(s + ".emo.npy")]
assert not missing, f"{len(missing)} emotion sources without .emo.npy, e.g. {missing[:3]}"

dst.mkdir(parents=True)  # fails if it exists, nothing is overwritten
for name in ["raw.arrow", "duration.json", "vocab.txt"]:
    os.symlink((src / name).resolve(), dst / name)
with open(dst / "pairs.json", "w", encoding="utf-8") as f:
    json.dump(pairs, f)

same = sum(info[p][2] == info[t][2] for t, (p, _) in pairs.items())
print(len(pairs), "targets,", len({u[1] for u in utts}), "speakers,", f"prompt emotion = target emotion in {same / len(pairs):.1%}")
print("prompt emotions:", dict(Counter(info[p][2] for p, _ in pairs.values())))
