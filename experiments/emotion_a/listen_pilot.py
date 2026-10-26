import csv
import os
import shutil
import sys
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text

# listening test for the pilot: same prompt, same unseen sentence and same seed, only model and emotion vector change
# usage: python listen_pilot.py <checkpoint .pt> <output dir> [emotion guidance strengths, e.g. 0 or 3,6] [duration: auto|neutral|oracle]
#   auto:    standard F5-TTS inference, trimmed prompt and duration estimated from the prompt's speaking rate
#   neutral: untrimmed prompt and plain prompt text as in cross-prompt training, length of the real neutral recording
#   oracle:  as neutral, but the length of the real recording in the emotion of the vector (ESD is parallel)
ckpt, out = sys.argv[1], Path(sys.argv[2])
strengths = [float(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0.0]
mode = sys.argv[4] if len(sys.argv) > 4 else "auto"
assert mode in ["auto", "neutral", "oracle"], mode
root = Path("/data/chi-gpu1/ge96xah/data/ESD")
speakers = ["0011", "0016"]
emotions = ["neutral", "angry", "happy", "sad", "surprise"]
prompts = ["neutral", "angry"]  # angry comes through worst in plain F5

# ESD evaluation split (not seen in training), per speaker and emotion sorted by utterance id
utts = {}
with open(root / "esd_csvs" / "evaluation.csv", encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        if r["lang"] == "en":
            utts.setdefault((os.path.basename(r["relative_audio_path"])[:4], r["emotion"].lower()), []).append(r)
for v in utts.values():
    v.sort(key=lambda r: r["relative_audio_path"])
assert all((s, e) in utts for s in speakers for e in emotions), sorted({e for _, e in utts})

cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")))
backbone = get_class(f"f5_tts.model.{cfg.model.backbone}")
f5 = load_model(backbone, cfg.model.arch, str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors")))
pilot = load_model(backbone, cfg.model.arch, ckpt, use_ema=False)  # online weights, the EMA has barely moved in a short run
vocoder = load_vocoder("vocos")
out.mkdir(parents=True, exist_ok=True)
log = open(out / "manifest.txt", "w", encoding="utf-8")
log.write(f"checkpoint {ckpt} | duration {mode}\n")


def seconds(row):
    return sf.info(str(root / row["relative_audio_path"])).duration


def save(name, model, emo, ref_audio, ref_text, gen_text, emo_cfg_strength=0.0, fix_duration=None):
    torch.manual_seed(0)
    wave, sr, _ = infer_process(
        ref_audio, ref_text, gen_text, model, vocoder, emo=emo, emo_cfg_strength=emo_cfg_strength, fix_duration=fix_duration
    )
    sf.write(str(out / name), wave, sr)


for spk in speakers:
    target = {e: utts[(spk, e)][5] for e in emotions}  # one sentence (index 5) in every emotion, differs from the prompts (index 0)
    gen_text = target["neutral"]["transcription"].strip()
    # vectors from yet another utterance (index 10) of the same speaker, never the prompt or the target
    vec_rows = {e: utts[(spk, e)][10] for e in emotions}
    vecs = {e: torch.from_numpy(np.load(f"{root / r['relative_audio_path']}.emo.npy")).float()[None] for e, r in vec_rows.items()}
    for p in prompts:
        prompt = utts[(spk, p)][0]
        prompt_wav, prompt_text = str(root / prompt["relative_audio_path"]), prompt["transcription"].strip()
        if mode == "auto":
            ref_audio, ref_text = preprocess_ref_audio_text(prompt_wav, prompt_text)
            dur = {e: None for e in emotions}
        else:  # infer_process adds one space after the prompt text, as the training separator
            ref_audio, ref_text = prompt_wav, prompt_text
            dur = {e: seconds(prompt) + seconds(target[e if mode == "oracle" else "neutral"]) for e in emotions}
        tag = f"{spk}_{p}prompt"
        shutil.copy(root / prompt["relative_audio_path"], out / f"{tag}_0_prompt.wav")
        save(f"{tag}_1_f5.wav", f5, None, ref_audio, ref_text, gen_text, fix_duration=dur["neutral"])
        save(f"{tag}_2_pilot_novec.wav", pilot, None, ref_audio, ref_text, gen_text, fix_duration=dur["neutral"])
        for w in strengths:  # files sort per guidance strength, then emotion
            for i, e in enumerate(emotions):
                save(f"{tag}_g{w:g}_{i + 3}_pilot_{e}vec.wav", pilot, vecs[e], ref_audio, ref_text, gen_text, w, dur[e])
        log.write(f"{tag}: prompt {prompt['relative_audio_path']} | text: {gen_text} | seconds: {dur}\n")
    log.write(f"{spk} vectors: " + ", ".join(f"{e} {r['relative_audio_path']}" for e, r in vec_rows.items()) + "\n")
log.close()
print("done", out)
