import sys
from pathlib import Path

import torch

# start a new run from the online weights of a training checkpoint (emo_proj included), with a fresh optimizer and
# learning-rate schedule; resuming the old run instead would restore its schedule, which has decayed to almost zero
# the output holds only ema_model_state_dict, so trainer.load_checkpoint takes its pretrained branch and starts at update 0
# usage: python warmstart_from.py <training checkpoint model_N.pt> <checkpoint dir of the new run>
src, dst_dir = Path(sys.argv[1]), Path(sys.argv[2])
taken = [p.name for p in dst_dir.iterdir() if p.name.startswith(("model_", "pretrained_"))]
assert not taken, f"{dst_dir} already holds {taken}, the trainer would load that instead"  # trainer.py load_checkpoint
ckpt = torch.load(src, map_location="cpu", weights_only=True, mmap=True)
online, ema = ckpt["model_state_dict"], ckpt["ema_model_state_dict"]

# same keys and shapes as the EMA copy the trainer itself wrote, only the values are the online weights
keys = [k for k in ema if k.startswith("ema_model.")]
assert sorted(k.removeprefix("ema_model.") for k in keys) == sorted(online), "online and EMA keys differ"
state = {k: online[k.removeprefix("ema_model.")].clone() for k in keys}
assert all(state[k].shape == ema[k].shape for k in keys)
dst = dst_dir / f"pretrained_{src.parent.name}_{src.stem}_online.pt"
torch.save({"ema_model_state_dict": state}, dst)  # no update/step key, else the trainer would resume the old schedule

check = torch.load(dst, map_location="cpu", weights_only=True)["ema_model_state_dict"]
assert all(torch.equal(check[k], online[k.removeprefix("ema_model.")]) for k in keys)
print(f"update {ckpt['update']} of {src}, {len(state)} tensors -> {dst}")
print("emo_proj mean abs weight", online["transformer.text_embed.emo_proj.weight"].abs().mean().item())  # as check_ckpt.py
