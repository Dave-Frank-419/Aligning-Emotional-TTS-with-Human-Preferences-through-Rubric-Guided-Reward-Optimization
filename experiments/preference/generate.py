"""Clips of a checkpoint with the emotion input, for the preference stage: the pool of own samples (training
sentences), the anchor's own neutral clips, and the evaluation of a trained state (evaluation sentences).

The items follow experiments/emotion_a/eval_emotions_generate.py (the fixed evaluation set and its seeds):
    targets   the six evaluation speakers' sentences of --split (train for pools and anchor clips, evaluation for the
              evaluation) at the positions --slots (indices into the speaker's recordings of the split, sorted by file
              name); --every n keeps every n-th position of that list (the seed and prompt of an item stay those of its
              place in the full list)
    prompt    a neutral recording of the speaker from the test split (index = place in --slots, wrapping around)
    source    the recording whose emotion vector is given: the speaker's test recording of the intended emotion at
              index (place + 10)
    text      the neutral transcription of the target sentence (ESD is parallel); length = prompt + the real target's
    seed      100 x (place of the speaker) + (place in --slots) + 1000 x --seed-offset, shared by the item's emotions
The emotion vector is that of the source, moved by the emotion input (emotion_input.py) where --strengths is given;
neutral items are never moved. The vector given to the model is stored in cond/<speaker>_<sentence>_<emotion>.npy, so
that a training lays the pair out under the same condition. Inference in float32 (the weights of the training), clips
as 32-bit float WAV, resumable (items already in the manifest are skipped, if they come from the same setting).
usage: python generate.py <output folder> --ckpt <weights .pt> [--split train] [--slots 0,3,...,267] [--every 1]
           [--emotions angry,happy,sad,surprise] [--strengths angry=0.5,happy=2,sad=1.5,surprise=1] [--seed-offset 30]
           [--speakers 0011,...] [--esd /data/chi-gpu1/ge96xah/data/ESD] [--nfe 32] [--guidance 3]
"""
import argparse
import csv
import os
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

import f5_tts.infer.utils_infer as ui
from f5_tts.model.dataset import trim_silence

from emotion_input import FOUR, SPEAKERS, parse_strengths, pushes

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--ckpt", required=True)  # online weights of a training run ({"model_state_dict": ...})
p.add_argument("--split", choices=["train", "evaluation"], default="evaluation")
p.add_argument("--slots", default="0,3,6,9,12,15")  # the default is the fixed evaluation set
p.add_argument("--every", type=int, default=1)
p.add_argument("--emotions", default="neutral,angry,happy,sad,surprise")
p.add_argument("--strengths", default="")  # the emotion input; empty = the source's vector as it is
p.add_argument("--seed-offset", type=int, default=0)
p.add_argument("--speakers", default=",".join(SPEAKERS))  # a subset of the six (tests); an item keeps the seed of its place among all six
p.add_argument("--esd", default="/data/chi-gpu1/ge96xah/data/ESD")
p.add_argument("--nfe", type=int, default=32)
p.add_argument("--guidance", type=float, default=3.0)  # the emotion guidance of the evaluation
args = p.parse_args()
root, slots, emotions, strength = Path(args.esd), [int(x) for x in args.slots.split(",")], args.emotions.split(","), parse_strengths(args.strengths)
assert set(emotions) <= {"neutral", *FOUR} and args.every >= 1


def rows(split):  # (speaker, emotion) -> rows sorted by file name
    out = {}
    with open(root / "esd_csvs" / f"{split}.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["lang"] == "en":
                out.setdefault((os.path.basename(r["relative_audio_path"])[:4], r["emotion"].lower()), []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["relative_audio_path"])
    return out


def sentence(r):  # ESD is parallel: the same number is the same sentence in every emotion
    return (int(Path(r["relative_audio_path"]).stem.split("_")[1]) - 1) % 350


def trimmed(r, pad_ms=0):  # the preparation of dataset.py for cross-prompt training
    audio, sr = torchaudio.load(str(root / r["relative_audio_path"]))
    return F.pad(trim_silence(audio, sr), (0, sr * pad_ms // 1000)), sr


ev, te = rows(args.split), rows("test")
items = []
for si, spk in enumerate(SPEAKERS):
    if spk not in args.speakers.split(","):
        continue
    for k, j in enumerate(slots):
        if k % args.every:
            continue
        for e in emotions:
            target, neutral = ev[(spk, e)][j], ev[(spk, "neutral")][j]
            prompt = te[(spk, "neutral")][k % len(te[(spk, "neutral")])]
            source = te[(spk, e)][(k + 10) % len(te[(spk, e)])]
            assert sentence(target) == sentence(neutral), (target, neutral)
            assert sentence(prompt) != sentence(target) and sentence(source) != sentence(target)
            items.append(dict(speaker=spk, sentence=sentence(target), emotion=e, text=neutral["transcription"].strip(),
                              target=target, prompt=prompt, source=source, seed=100 * si + k + 1000 * args.seed_offset))

push, outside = pushes(root) if strength else ({}, {})
out = Path(args.out)
(out / "prompts").mkdir(parents=True, exist_ok=True)
(out / "cond").mkdir(exist_ok=True)
device = "cuda"
_load = ui.load_checkpoint
ui.load_checkpoint = lambda model, path, device, dtype=None, use_ema=True: _load(model, path, device, dtype=torch.float32, use_ema=use_ema)  # float32, as in training
cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")))
model = ui.load_model(get_class(f"f5_tts.model.{cfg.model.backbone}"), cfg.model.arch, str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors")))
model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True, mmap=True)["model_state_dict"])  # online weights
vocoder = ui.load_vocoder("vocos")
with open(out / "condition.txt", "a", encoding="utf-8") as f:  # one line per call: a folder may be filled by several calls (one per emotion)
    f.write(f"{args.split} split, slots {args.slots}, every {args.every}; emotions {args.emotions}; weights {args.ckpt}; seed offset {args.seed_offset}; "
            + (f"emotion input: vector of the source + strength x push ({args.strengths}; experiments/preference/emotion_input.py)" if strength else "the vector of the source as it is")
            + "; the vector given is in cond/\n")

fields = ["path", "arm", "checkpoint", "speaker", "sentence", "emotion", "prompt", "source", "seconds", "seed", "nfe", "guidance", "strength"]
manifest = out / "manifest.csv"
setting = ("ckpt", args.ckpt, str(args.nfe), str(args.guidance), args.seed_offset)
done = set()
if manifest.exists() and manifest.stat().st_size:
    with open(manifest, encoding="utf-8", newline="") as f:
        old = list(csv.DictReader(f))
    assert all((r["arm"], r["checkpoint"], r["nfe"], r["guidance"], int(r["seed"]) // 1000) == setting for r in old), f"{out} holds items of another setting, use a new folder"
    done = {r["path"] for r in old}
log = open(manifest, "a", encoding="utf-8", newline="")
writer = csv.DictWriter(log, fieldnames=fields)
if not done:
    writer.writeheader()
print(f"{len(items)} items ({len(done)} done before), {args.split} split, emotions {emotions}, strengths {strength or 'none'}, seed offset {args.seed_offset}, float32", flush=True)
for n, it in enumerate(items):
    path = out / f"{it['speaker']}_{it['sentence']:03d}_{it['emotion']}.wav"
    if str(path) in done:
        continue
    audio, sr = trimmed(it["target"])
    seconds = audio.shape[-1] / sr
    prompt_audio, prompt_sr = trimmed(it["prompt"], pad_ms=50)
    ref_audio = out / "prompts" / Path(it["prompt"]["relative_audio_path"]).name
    sf.write(str(ref_audio), prompt_audio[0].numpy(), prompt_sr)
    emo = torch.from_numpy(np.load(f"{root / it['source']['relative_audio_path']}.emo.npy")).float()[None]
    d = strength.get(it["emotion"], 0.0)
    if d:  # the emotion input; never for neutral
        s = int(it["speaker"])
        emo = emo + d * torch.from_numpy(push[(s, it["emotion"])] if (s, it["emotion"]) in push else outside[it["emotion"]]).float()[None]
    np.save(out / "cond" / f"{it['speaker']}_{it['sentence']:03d}_{it['emotion']}.npy", emo[0].numpy().astype(np.float32))
    torch.manual_seed(it["seed"])
    wave, wave_sr, _ = ui.infer_process(
        str(ref_audio), it["prompt"]["transcription"].strip(), it["text"], model, vocoder, emo=emo,
        emo_cfg_strength=args.guidance, nfe_step=args.nfe,
        fix_duration=prompt_audio.shape[-1] / prompt_sr + seconds,  # prompt plus the real length of the target
        target_rms=0.0,  # training sees the original level
    )
    sf.write(str(path), wave, wave_sr, subtype="FLOAT")
    writer.writerow(dict(path=str(path), arm="ckpt", checkpoint=args.ckpt, speaker=it["speaker"], sentence=it["sentence"], emotion=it["emotion"],
                         prompt=it["prompt"]["relative_audio_path"], source=it["source"]["relative_audio_path"], seconds=f"{seconds:.3f}",
                         seed=it["seed"], nfe=args.nfe, guidance=args.guidance, strength=d))
    log.flush()
    if n % 50 == 0:
        print(n, "/", len(items), flush=True)
log.close()
print("done", flush=True)
