from importlib.resources import files

import numpy as np
import torch
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text

cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")))
model = load_model(
    get_class(f"f5_tts.model.{cfg.model.backbone}"),
    cfg.model.arch,
    str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors")),
    mel_spec_type="vocos",
    vocab_file="",
)
vocoder = load_vocoder("vocos")
ref_audio, ref_text = preprocess_ref_audio_text(
    str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")),
    "Some call me nature, others call me mother nature.",
)
gen_text = "The meeting has been moved to Thursday afternoon."
emo = torch.randn(1, 1024)


def generate(e):
    torch.manual_seed(0)
    wave, _, _ = infer_process(ref_audio, ref_text, gen_text, model, vocoder, emo=e)
    return wave


base = generate(None)
zero = generate(emo)
print("emo_proj starts at zero:", bool((model.transformer.text_embed.emo_proj.weight == 0).all()))
print("output identical with zero emo_proj:", base.shape == zero.shape and np.array_equal(base, zero))

torch.nn.init.normal_(model.transformer.text_embed.emo_proj.weight, std=0.05)
moved = generate(emo)
n = min(len(base), len(moved))
print("max difference with random emo_proj:", float(np.abs(base[:n] - moved[:n]).max()))
