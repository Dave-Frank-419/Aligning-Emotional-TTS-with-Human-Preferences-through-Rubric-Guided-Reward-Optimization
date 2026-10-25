import sys

import torch

# quick look at a training checkpoint: has emo_proj moved away from its zero init, and which lr per param group
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
w = ckpt["model_state_dict"]["transformer.text_embed.emo_proj.weight"]
print("update", ckpt["update"])
print("emo_proj mean abs weight", w.abs().mean().item())  # 0 = emotion input not trained
print("lr per param group", [g["lr"] for g in ckpt["optimizer_state_dict"]["param_groups"]])
