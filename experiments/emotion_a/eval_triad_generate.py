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

from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder
from f5_tts.model.dataset import trim_silence
from f5_tts.model.modules import MelSpec

# fixed evaluation set for angry/happy/surprise: the same items in every arm, so that arms compare item by item
# items: 6 speakers x 6 evaluation sentences x 5 emotions; prompts and vector sources come from the test split
# arms: real     the trimmed recording of the item
#       vocoded  that recording through F5's mel and Vocos, the ceiling for generated speech
#       prompt   unmodified F5 with a prompt in the target emotion, no vector
#       ckpt     a training checkpoint (online weights) with a neutral prompt and a vector of the target emotion
#       novec    the same checkpoint without vector: what the emotion-specific length alone gives (lower bound)
# generated arms get the length of the real recording and one seed per item, shared by all emotions and arms
# usage: python eval_triad_generate.py <output dir> <real|vocoded|prompt|ckpt|novec> [checkpoint .pt] [--nfe 32] [--guidance 3]
p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("arm", choices=["real", "vocoded", "prompt", "ckpt", "novec"])
p.add_argument("ckpt", nargs="?")
p.add_argument("--nfe", type=int, default=32)
p.add_argument("--guidance", type=float, default=3.0)  # emotion guidance, only for the ckpt arm
args = p.parse_args()
assert (args.arm in ["ckpt", "novec"]) == (args.ckpt is not None), "a checkpoint is needed for the ckpt and novec arms only"
guidance = args.guidance if args.arm == "ckpt" else 0.0
root = Path("/data/chi-gpu1/ge96xah/data/ESD")
speakers = ["0011", "0013", "0015", "0016", "0018", "0020"]
slots = [0, 3, 6, 9, 12, 15]  # positions in the evaluation split, 6 sentences per speaker
emotions = ["neutral", "angry", "happy", "sad", "surprise"]


def rows(split):  # (speaker, emotion) -> rows sorted by utterance id
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


def trimmed(r, pad_ms=0):  # same preparation as dataset.py for cross-prompt training
    audio, sr = torchaudio.load(str(root / r["relative_audio_path"]))
    return F.pad(trim_silence(audio, sr), (0, sr * pad_ms // 1000)), sr


ev, te = rows("evaluation"), rows("test")
items = []
for si, spk in enumerate(speakers):
    for k, j in enumerate(slots):
        for e in emotions:
            target, neutral = ev[(spk, e)][j], ev[(spk, "neutral")][j]
            prompt = te[(spk, e if args.arm == "prompt" else "neutral")][k]
            source = te[(spk, e)][(k + 10) % len(te[(spk, e)])]  # another utterance than the prompt arm's prompt
            assert sentence(target) == sentence(neutral), (target, neutral)
            assert sentence(prompt) != sentence(target) and sentence(source) != sentence(target)
            items.append(dict(speaker=spk, sentence=sentence(target), emotion=e, text=neutral["transcription"].strip(),
                              target=target, prompt=prompt, source=source, seed=100 * si + k))

out = Path(args.out)
(out / "prompts").mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
vocoder = load_vocoder("vocos") if args.arm != "real" else None
model = None
if args.arm in ["prompt", "ckpt", "novec"]:
    cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")))
    backbone = get_class(f"f5_tts.model.{cfg.model.backbone}")
    model = load_model(backbone, cfg.model.arch, str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors")))
    if args.ckpt:  # online weights only, the EMA has barely moved in these runs; mmap reads only these tensors
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True, mmap=True)["model_state_dict"])
mel_spec = MelSpec(mel_spec_type="vocos").to(device) if args.arm == "vocoded" else None

fields = ["path", "arm", "checkpoint", "speaker", "sentence", "emotion", "prompt", "source", "seconds", "seed", "nfe", "guidance"]
manifest = out / "manifest.csv"
new = not manifest.exists() or manifest.stat().st_size == 0
done = set()
if not new:  # resumable: items already in the manifest are skipped, but only if they come from this very setting
    with open(manifest, encoding="utf-8", newline="") as f:
        old = list(csv.DictReader(f))
    same = [(r["arm"], r["checkpoint"], r["nfe"], r["guidance"]) == (args.arm, args.ckpt or "", str(args.nfe), str(guidance)) for r in old]
    assert all(same), f"{out} holds items of another arm, checkpoint or setting, use a new name"
    done = {r["path"] for r in old}
log = open(manifest, "a", encoding="utf-8", newline="")
writer = csv.DictWriter(log, fieldnames=fields)
if new:
    writer.writeheader()
for n, it in enumerate(items):
    path = out / f"{it['speaker']}_{it['sentence']:03d}_{it['emotion']}.wav"
    if str(path) in done:
        continue
    audio, sr = trimmed(it["target"])
    seconds = audio.shape[-1] / sr
    if args.arm == "real":
        sf.write(str(path), audio[0].numpy(), sr)
    elif args.arm == "vocoded":
        with torch.inference_mode():
            wave = vocoder.decode(mel_spec(torchaudio.functional.resample(audio, sr, 24000).to(device)))
        sf.write(str(path), wave.squeeze().cpu().numpy(), 24000)
    else:
        prompt_audio, prompt_sr = trimmed(it["prompt"], pad_ms=50)
        ref_audio = out / "prompts" / Path(it["prompt"]["relative_audio_path"]).name
        sf.write(str(ref_audio), prompt_audio[0].numpy(), prompt_sr)
        emo = None
        if args.arm == "ckpt":
            emo = torch.from_numpy(np.load(f"{root / it['source']['relative_audio_path']}.emo.npy")).float()[None]
        torch.manual_seed(it["seed"])
        wave, wave_sr, _ = infer_process(
            str(ref_audio), it["prompt"]["transcription"].strip(), it["text"], model, vocoder, emo=emo,
            emo_cfg_strength=guidance, nfe_step=args.nfe,
            fix_duration=prompt_audio.shape[-1] / prompt_sr + seconds,  # prompt plus the real length of the target
            target_rms=0.0,  # training sees the original level
        )
        sf.write(str(path), wave, wave_sr)
    writer.writerow(dict(path=str(path), arm=args.arm, checkpoint=args.ckpt or "", speaker=it["speaker"], sentence=it["sentence"],
                         emotion=it["emotion"], prompt=it["prompt"]["relative_audio_path"], source=it["source"]["relative_audio_path"],
                         seconds=f"{seconds:.3f}", seed=it["seed"], nfe=args.nfe, guidance=guidance))
    log.flush()
    if n % 30 == 0:
        print(n, "/", len(items), flush=True)
log.close()
print("done", out)
