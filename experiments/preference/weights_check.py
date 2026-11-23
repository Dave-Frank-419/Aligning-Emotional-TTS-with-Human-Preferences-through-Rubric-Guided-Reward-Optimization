"""The start weights of the preference stage on another machine: extract them, and check that they are the same tensors.

The preference stage starts from the online weights of the SFT checkpoint (the continuation run's model_36666.pt):
generate.py and train_dpo.py read {"model_state_dict": ...}. A file written by torch.save can differ in its bytes on
two machines while holding the same tensors, so the check is a SHA-256 over the tensors themselves (sorted by name:
name, dtype, shape and the raw bytes of each), which is the same wherever the weights are the same.
usage: python weights_check.py extract <training checkpoint model_N.pt> <out .pt>   (CPU only)
       python weights_check.py checksum <weights .pt>                              -> one line: the SHA-256 and the tensor count
"""
import hashlib
import sys

import torch

mode = sys.argv[1]
if mode == "extract":
    ckpt = torch.load(sys.argv[2], map_location="cpu", weights_only=True, mmap=True)
    torch.save({"model_state_dict": ckpt["model_state_dict"]}, sys.argv[3])  # the online weights, as the trainer wrote them
    print(f"written {sys.argv[3]}: the online weights of {sys.argv[2]} ({len(ckpt['model_state_dict'])} tensors)")
elif mode == "checksum":
    w = torch.load(sys.argv[2], map_location="cpu", weights_only=True, mmap=True)["model_state_dict"]
    h = hashlib.sha256()
    for k in sorted(w):
        t = w[k].detach().contiguous().cpu()
        h.update(k.encode() + str(t.dtype).encode() + str(tuple(t.shape)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    print(f"{h.hexdigest()}  {len(w)} tensors  {sys.argv[2]}")
else:
    sys.exit("usage: weights_check.py extract <checkpoint> <out> | checksum <weights>")
