import csv
import inspect
import sys
import time
from pathlib import Path

import librosa
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# third emotion judge in the style of EmoSURA: an audio LLM verifies one unit at a time ("Does the speaker sound angry?"),
# the probability of "yes" is the score of the unit; tests whether the planned reward can hear the emotions at all
# the 7B model does not fit on the 12 GB GPU, so part of it stays in CPU memory: slow, but enough for offline scoring
# usage: python score_audio_llm.py <model dir> <manifest.csv> [more manifests]  -> scores_llm.csv next to each manifest
model_dir, manifests = sys.argv[1], [Path(m) for m in sys.argv[2:]]
units = {"neutral": "emotionally neutral", "angry": "angry", "happy": "happy", "sad": "sad", "surprise": "surprised"}
processor = AutoProcessor.from_pretrained(model_dir)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=torch.float16, device_map="auto", max_memory={0: "10GiB", "cpu": "48GiB"}
).eval()
tok = processor.tokenizer
yes = [tok.encode(w, add_special_tokens=False)[0] for w in ["Yes", "yes"]]
no = [tok.encode(w, add_special_tokens=False)[0] for w in ["No", "no"]]
audio_arg = "audio" if "audio" in inspect.signature(processor.__call__).parameters else "audios"  # renamed in newer versions
prompts = [
    processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "audio", "audio_url": "clip.wav"},
                                      {"type": "text", "text": f"Does the speaker sound {u}? Answer yes or no."}]}],
        add_generation_prompt=True, tokenize=False)
    for u in units.values()
]
print(prompts[1], flush=True)
sr = processor.feature_extractor.sampling_rate


def scores(path):  # one batch: the same clip with one question per unit
    wave, _ = librosa.load(path, sr=sr)
    inputs = processor(text=prompts, **{audio_arg: [wave] * len(prompts)}, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)
    inputs["input_features"] = inputs["input_features"].to(model.dtype)
    with torch.no_grad():
        logits = model(**inputs).logits.float()
    mask = inputs["attention_mask"]
    last = mask.shape[1] - 1 - mask.flip(1).argmax(1)  # last real token, whatever the padding side
    z = logits[torch.arange(len(prompts)), last]
    p = torch.sigmoid(torch.logsumexp(z[:, yes], -1) - torch.logsumexp(z[:, no], -1))
    assert torch.isfinite(p).all(), f"non-finite scores for {path}, fp16 overflow?"
    return p.tolist()


for manifest in manifests:
    with open(manifest, encoding="utf-8", newline="") as f:
        items = list(csv.DictReader(f))
    start = time.time()
    with open(manifest.parent / "scores_llm.csv", "w", encoding="utf-8", newline="") as f:
        out = csv.writer(f)
        out.writerow(["path"] + list(units))
        for it in items:
            out.writerow([it["path"]] + [f"{p:.6g}" for p in scores(it["path"])])
    print(f"scored {len(items)} in {time.time() - start:.0f} s ->", manifest.parent / "scores_llm.csv", flush=True)
