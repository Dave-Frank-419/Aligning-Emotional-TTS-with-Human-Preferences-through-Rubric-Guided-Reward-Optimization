from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# emotion embedding as in emotional-vits (emotion_extract.py): wav2vec2 fine-tuned on MSP-Podcast
# for arousal/dominance/valence, last hidden layer averaged over time, one 1024-d vector per utterance
model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
root = Path("/data/chi-gpu1/ge96xah/data/ESD")

extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name).to("cuda").eval()

wavs = []
for split in ["train", "evaluation", "test"]:
    lines = (root / f"metadata_{split}.csv").read_text(encoding="utf-8").splitlines()[1:]
    wavs += [line.split("|")[0] for line in lines]

for i, wav in enumerate(wavs):
    out = wav + ".emo.npy"  # next to the wav, where dataset.py looks for it
    if Path(out).exists():
        continue
    audio, sr = torchaudio.load(wav)
    audio = torchaudio.functional.resample(audio.mean(0), sr, 16000)
    inputs = extractor(audio.numpy(), sampling_rate=16000, return_tensors="pt").input_values.to("cuda")
    with torch.no_grad():
        emb = model(inputs).last_hidden_state.mean(dim=1)[0]
    np.save(out, emb.cpu().numpy())
    if i % 1000 == 0:
        print(i, "/", len(wavs))

print("done", len(wavs))
