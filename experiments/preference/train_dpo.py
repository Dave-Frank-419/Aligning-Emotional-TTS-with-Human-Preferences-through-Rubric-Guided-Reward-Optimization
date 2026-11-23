"""Preference stage of the emotional F5-TTS: Flow-DPO on pairs of the model's own samples, with an anchor on neutral.

What it follows
  Ye et al., "Refining Emphasis Control in Flow-Matching TTS via Preference Alignment and Reinforcement Learning",
  Interspeech 2026 (Sec. 2.4 and 3.1): Flow-DPO on F5-TTS with pairs of the SFT model's own samples,
  all parameters trained, start and frozen reference = the SFT checkpoint, loss = L_DPO + 0.5 x L_SFT, AdamW, learning
  rate 1e-5, time step uniform in (0, 1):
      err(x)  = mean squared error of the predicted flow on the target frames of x (the frames after the prompt)
      D       = (err_policy(winner) - err_ref(winner)) - (err_policy(loser) - err_ref(loser))
      L_DPO   = -log sigmoid(-beta x D)
  Diffusion-DPO: noise and time step shared inside a pair, error averaged per sample, reference without gradient, a
  linear warm-up of the learning rate.
  The anchor (a deviation from Ye et al.; it kept neutral speech, the level and the word errors in the thesis run e4lman1000):
      drift(clip) = mean over target frames and mel bins of (G_policy - G_ref)^2, G = the velocity the sampler
                    integrates, 6 x full condition - 3 x without the vector - 2 x unconditional
      L = L_DPO + sft_weight x L_SFT + anchor_weight x mean drift over --anchor-items of the start model's own
          neutral clips per update (make_anchor_table.py)
  It is 0 with a gradient of 0 at the start, and only resists movement (the form of GROW's velocity anchor,
  arXiv:2608.03215, and of the KL penalty of Flow-GRPO, arXiv:2505.05470).
CHOICES no paper states: the error is a mean over target frames and mel bins, beta is given for that mean; dropout is
off in the pair and anchor passes (so D is exactly 0 while the policy equals the reference); the pair passes use the full
condition; the SFT term runs on the real ESD training items of the baseline (cross-emotion prompts) with the baseline's
dropout; one learning rate; gradient clipping at 1.0 as in the baseline.
Pairs (make_pairs.py): a pair is laid out as the model saw it at its generation: the prompt in front (the condition),
the clip behind it, prompt text and text, and the vector the pool was generated with (column emo_vec); --level-match
sets both clips of a pair to their common level first, so that "louder" cannot be learned from a pair. Every update
takes --pairs-per-step pairs from each table (one table per emotion).
One update = the pairs, the anchor clips and --sft-items real items, one sequence at a time (gradients add up, nothing is
padded), then one clipped AdamW update. Monitors every --val-every updates, with fixed noise and time steps: D and the
share of ordered pairs on the held-out pairs (per table) and on 24 watched training pairs; the flow error on 24 real
training items in the three branches of inference; the drift on the held-out own clips.
Safety: the start checks stop the run unless D = 0 and the loss = ln 2 for the first pairs, every monitor is 0, the
anchor and its gradient are 0, and the GPU memory suffices (--mem-gb is a hard limit of the process: an error instead
of a silent spill into system memory). An update with a loss or gradient that is not finite is not made; the run stops.
Saved: the weights after the updates in --save (model_<n>.pt, online weights as the generation reads them); a full
state every --state-every updates (state.pt: weights, optimiser, step and every random generator, written to a
temporary file first), from which --resume continues where it was, on the same draws (up to the rounding of the card). A file STOP stops the run
before its next update (the evaluation writes it when a stop rule fires); the run then saves its state and ends. The
run stops as well before a save would leave less than --min-free-gb on the disk. log.jsonl holds the options, the code
version, every update and every event.
usage: python train_dpo.py --run <run folder> --start <SFT weights .pt> --dataset <F5 data set of the SFT term, e.g.
           ESD_en_xprompt1_cont> --pairs a.csv,b.csv,... --beta 100 --steps 1500 --save 300,600 [--level-match]
           [--anchor <table> --anchor-weight 800 --anchor-items 8] [--resume] [--check]
"""
import argparse
import json
import math
import random
import shutil
import subprocess
import time
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

import f5_tts.infer.utils_infer as ui
from f5_tts.model.dataset import collate_fn, load_dataset
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import convert_char_to_pinyin, list_str_to_idx

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)  # the run's folder; a run folder is used by one run (and its resumptions)
p.add_argument("--start", required=True)  # the SFT checkpoint: start of the policy and frozen reference
p.add_argument("--dataset", required=True)  # the F5 data set of the SFT term (needs cross-emotion prompts)
p.add_argument("--esd", default="/data/chi-gpu1/ge96xah/data/ESD")  # for a pair without emo_vec and for the anchor clips' vectors
p.add_argument("--beta", type=float, default=100.0)
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--save", default="")  # updates after which the weights are saved, e.g. 300,600,900
p.add_argument("--state-every", type=int, default=100)
p.add_argument("--pairs-per-step", type=int, default=8)
p.add_argument("--sft-items", type=int, default=4)
p.add_argument("--sft-weight", type=float, default=0.5)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--warmup", type=int, default=75)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--val-every", type=int, default=50)
p.add_argument("--val-max", type=int, default=0)  # at most this many held-out pairs per table in the monitors; 0 = all
p.add_argument("--mem-gb", type=float, default=10.0)
p.add_argument("--min-free-gb", type=float, default=5.0)
p.add_argument("--check", action="store_true")  # the start checks only: no update, nothing saved
p.add_argument("--resume", action="store_true")
p.add_argument("--level-match", action="store_true")
p.add_argument("--pairs", required=True)
p.add_argument("--anchor", default="")
p.add_argument("--anchor-weight", type=float, default=0.0)
p.add_argument("--anchor-items", type=int, default=0)
args = p.parse_args()
assert not args.anchor_items or args.anchor, "the anchor term needs a table of own clips (--anchor)"
dev = "cuda"
torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory))
random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
run, ESD = Path(args.run), Path(args.esd)
assert args.check or args.resume or not (run / "log.jsonl").exists(), f"the run folder {run} holds a run already: use --resume, or a new folder"
assert not args.resume or (run / "state.pt").exists(), f"--resume: no state.pt in {run}"
save_at = {int(s) for s in args.save.split(",") if s}
assert all(0 < s <= args.steps for s in save_at), "a save step lies outside the run"

_load_checkpoint = ui.load_checkpoint
ui.load_checkpoint = lambda model, path, device, dtype=None, use_ema=True: _load_checkpoint(model, path, device, dtype=torch.float32, use_ema=use_ema)  # float32 as in training
cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")))


def build():
    m = ui.load_model(get_class(f"f5_tts.model.{cfg.model.backbone}"), OmegaConf.to_container(cfg.model.arch),
                      str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors")))
    m.load_state_dict(torch.load(args.start, map_location="cpu", weights_only=True, mmap=True)["model_state_dict"])
    return m.to(dev).float()


model, ref = build(), build()
ref.eval().requires_grad_(False)
assert all(q.requires_grad for q in model.parameters()), "a parameter of the policy is frozen"  # Ye et al.: all parameters are trained
model.emo_drop_prob = 0.1  # the baseline runs trained with --emo_drop_prob 0.1; the other dropout rates are the class defaults
assert (model.audio_drop_prob, model.cond_drop_prob) == (0.3, 0.2)
mel_spec = MelSpec(**OmegaConf.to_container(cfg.model.mel_spec))


def wave_of(path):  # as dataset.py: mono, 24 kHz
    audio, sr = torchaudio.load(str(path))
    audio = audio.mean(0, keepdim=True)
    if sr != mel_spec.target_sample_rate:
        audio = torchaudio.functional.resample(audio, sr, mel_spec.target_sample_rate)
    return audio


def mel_of(path, gain=1.0):  # mel of a file (d x t); gain scales the waveform first
    return mel_spec(wave_of(path) * gain).squeeze(0).cpu()


def level_gains(winner, loser):
    """Gains that set both clips of a pair to their common level (the geometric mean of their RMS values); returns the
    two gains and the level gap in dB before."""
    rw, rl = (float(wave_of(q).pow(2).mean().sqrt()) for q in (winner, loser))
    common = math.sqrt(rw * rl)
    return common / rw, common / rl, 20 * math.log10(rw / rl)


# ---------------------------------------------------------------------------------------------------------- the pairs
names = [Path(f).stem for f in args.pairs.split(",")]  # a group = one table (one emotion)
table = pd.concat([pd.read_csv(f).assign(group=g) for g, f in enumerate(args.pairs.split(","))], ignore_index=True)
pairs, gaps = [], []
for r in table.itertuples():
    gw, gl, gap = level_gains(r.winner, r.loser) if args.level_match else (1.0, 1.0, 0.0)
    gaps.append(gap)
    prompt = mel_of(r.prompt_wav)
    # the generation vocodes the frames from the last prompt frame on (utils_infer.py: generated[:, ref_audio_len:]), so the
    # first mel frame of a saved clip is that prompt frame again; it is dropped to get the sequence of the generation
    w, l = mel_of(r.winner, gw)[:, 1:], mel_of(r.loser, gl)[:, 1:]
    assert w.shape == l.shape, f"winner and loser differ in length: {r.winner}"
    info = sf.info(str(r.prompt_wav))  # the number of frames the generation used: int((prompt seconds + target seconds) x 24000 / 256)
    frames = int((info.frames / info.samplerate + float(r.seconds)) * mel_spec.target_sample_rate / mel_spec.hop_length)
    assert prompt.shape[-1] + w.shape[-1] == frames, f"{r.winner}: {prompt.shape[-1]} + {w.shape[-1]} frames, the generation had {frames}"
    ref_text = r.prompt_text + (" " if len(r.prompt_text[-1].encode("utf-8")) == 1 else "")  # as infer_batch_process
    vec = getattr(r, "emo_vec", "")  # the vector the pair was generated with; empty = the source clip's vector
    pairs.append(dict(w=torch.cat([prompt, w], -1).T, l=torch.cat([prompt, l], -1).T, pf=prompt.shape[-1], text=convert_char_to_pinyin([ref_text + r.text])[0],
                      emo=torch.from_numpy(np.load(vec) if isinstance(vec, str) and vec else np.load(str(ESD / r.source) + ".emo.npy")).float(), split=r.split, group=int(r.group)))
train_pairs, val_pairs = [x for x in pairs if x["split"] == "train"], [x for x in pairs if x["split"] == "val"]
if args.val_max:  # the monitors on at most this many held-out pairs per table, one fixed draw
    pick, held = random.Random(11), len(val_pairs)
    val_pairs = [x for g in range(len(names)) for v in [[x for x in val_pairs if x["group"] == g]] for x in (v if len(v) <= args.val_max else pick.sample(v, args.val_max))]
    print(f"held-out pairs: {held}; the monitors use {len(val_pairs)} of them (at most {args.val_max} per table, a fixed draw)", flush=True)
watch_pairs = random.Random(7).sample(train_pairs, min(24, len(train_pairs)))  # training pairs that are measured like the held-out ones
if args.level_match:
    print(f"level matched inside every pair: the winner was louder by {np.mean(gaps):+.2f} dB on average (SD {np.std(gaps):.2f}); now 0", flush=True)
groups = [[i for i, x in enumerate(train_pairs) if x["group"] == g] for g in range(len(names))]  # indices of the training pairs of each table
print(f"pairs: {len(train_pairs)} for training, {len(val_pairs)} held out" + ("" if len(names) == 1 else "; per table: " + ", ".join(f"{n} {len(i)}" for n, i in zip(names, groups))), flush=True)
assert train_pairs and val_pairs and all(groups), "every table needs pairs for training, and held-out pairs are needed"


def pair_batch(q, gen, t=None):
    """The two sequences of one pair (winner, loser) as one batch of two, with the same noise and the same time step."""
    x1 = torch.stack([q["w"], q["l"]]).to(dev)  # 2 n d
    n = x1.shape[1]
    x0 = torch.randn(1, n, x1.shape[2], device=dev, generator=gen).expand(2, -1, -1)
    time_ = (torch.rand(1, device=dev, generator=gen) if t is None else torch.full((1,), t, device=dev)).expand(2)
    span = (torch.arange(n, device=dev) >= q["pf"])[None].expand(2, -1)  # the target frames; the prompt is the condition
    tb = time_[:, None, None]
    return dict(x=(1 - tb) * x0 + tb * x1, flow=x1 - x0, cond=torch.where(span[..., None], torch.zeros_like(x1), x1),
                text=list_str_to_idx([q["text"], q["text"]], model.vocab_char_map).to(dev), time=time_,
                mask=torch.ones(2, n, dtype=torch.bool, device=dev), emo=q["emo"][None].expand(2, -1).to(dev), span=span)


def err(m, b):  # per sequence: mean squared error of the predicted flow on the target frames (CHOICE: mean, not sum)
    pred = m.transformer(x=b["x"], cond=b["cond"], text=b["text"], time=b["time"], drop_audio_cond=False, drop_text=False, mask=b["mask"], emo=b["emo"])
    return (((pred - b["flow"]) ** 2).mean(-1) * b["span"]).sum(1) / b["span"].sum(1)


def margin(q, gen, t=None):
    """D of one pair, the policy's errors (winner, loser) with gradient and the reference's errors without."""
    b = pair_batch(q, gen, t)
    with torch.no_grad():
        er = err(ref, b)
    em = err(model, b)
    return (em[0] - er[0]) - (em[1] - er[1]), em, er


# --------------------------------------------------------------------------- the anchor: the start model's own clips
BRANCHES = dict(full=dict(drop_audio_cond=False, drop_text=False), novec=dict(drop_audio_cond=False, drop_text=False), uncond=dict(drop_audio_cond=True, drop_text=True))
GUIDED = dict(full=6.0, novec=-3.0, uncond=-2.0)  # the sampler: pred + 2 (pred - uncond) + 3 (pred - novec), with the guidance values of the evaluation
own = pd.read_csv(args.anchor) if args.anchor else None
own_cache = {}


def own_item(i):  # one own clip laid out as a pair member: prompt in front, the clip behind it (its first frame is the last prompt frame, see above)
    if i not in own_cache:
        r = own.iloc[i].to_dict()  # a dict: "clip" as an attribute of a pandas row would be the method of that name
        prompt, clip = mel_of(r["prompt_wav"]), mel_of(r["clip"])[:, 1:]
        info = sf.info(str(r["prompt_wav"]))
        frames = int((info.frames / info.samplerate + float(r["seconds"])) * mel_spec.target_sample_rate / mel_spec.hop_length)
        assert prompt.shape[-1] + clip.shape[-1] == frames, f"{r['clip']}: {prompt.shape[-1]} + {clip.shape[-1]} frames, the generation had {frames}"
        ref_text = r["prompt_text"] + (" " if len(r["prompt_text"][-1].encode("utf-8")) == 1 else "")
        own_cache[i] = dict(x=torch.cat([prompt, clip], -1).T, pf=prompt.shape[-1], text=convert_char_to_pinyin([ref_text + r["text"]])[0],
                            emo=torch.from_numpy(np.load(str(ESD / r["source"]) + ".emo.npy")).float())
    return own_cache[i]


def own_batch(q, gen, t=None):
    """One own clip at a noise sample and a time step, as pair_batch does for the two clips of a pair."""
    x1 = q["x"][None].to(dev)  # 1 n d
    n = x1.shape[1]
    x0 = torch.randn(1, n, x1.shape[2], device=dev, generator=gen)
    time_ = torch.rand(1, device=dev, generator=gen) if t is None else torch.full((1,), t, device=dev)
    span = (torch.arange(n, device=dev) >= q["pf"])[None]
    tb = time_[:, None, None]
    return dict(x=(1 - tb) * x0 + tb * x1, cond=torch.where(span[..., None], torch.zeros_like(x1), x1), text=list_str_to_idx([q["text"]], model.vocab_char_map).to(dev),
                time=time_, mask=torch.ones(1, n, dtype=torch.bool, device=dev), emo=q["emo"][None].to(dev), span=span)


def velocity(m, b, branch):  # the model's prediction in one of the three branches the sampler uses
    return m.transformer(x=b["x"], cond=b["cond"], text=b["text"], time=b["time"], mask=b["mask"], emo=None if branch == "novec" else b["emo"], **BRANCHES[branch])


def drift_step(b, scale):
    """The drift of one own clip on the guided velocity: adds scale x its gradient to the policy's gradients and returns
    the drift. The drift is quadratic in the weighted sum of the three branches, so the gradient through branch k is
    2 x coefficient_k x residual; the branches are passed one at a time (the same gradient, one graph in memory)."""
    w = b["span"][..., None] / (b["span"].sum() * b["x"].shape[-1])  # weights that make the sum below the mean over target frames and bins
    with torch.no_grad():
        res = sum(c * (velocity(model, b, k) - velocity(ref, b, k)) for k, c in GUIDED.items())
    for k, c in GUIDED.items():
        (scale * (2 * c * res * w * velocity(model, b, k)).sum()).backward()
    return float(((res ** 2) * w).sum())


own_train = [i for i in range(len(own)) if own["split"].iloc[i] == "train"] if args.anchor else []
own_val = [i for i in range(len(own)) if own["split"].iloc[i] == "val" and own["offset"].iloc[i] == own["offset"].min()] if args.anchor else []  # the held-out clips of the first offset: monitoring only


@torch.no_grad()
def own_drift():
    """Monitor: the drift on the held-out own clips with fixed noise and time steps, per branch and for the guided
    velocity, and the share of the guided drift that the two branches without the full condition carry (rest)."""
    was = model.training
    model.eval()
    gen, out = torch.Generator(device=dev).manual_seed(24680), dict(full=[], novec=[], uncond=[], guided=[], rest=[])
    for i in own_val:
        q = own_item(i)
        for t in (0.2, 0.5, 0.8):
            b = own_batch(q, gen, t)
            d = {k: velocity(model, b, k) - velocity(ref, b, k) for k in BRANCHES}
            sq = lambda v: float((((v ** 2).mean(-1) * b["span"]).sum(1) / b["span"].sum(1))[0])
            for k in BRANCHES:
                out[k].append(sq(d[k]))
            out["guided"].append(sq(sum(c * d[k] for k, c in GUIDED.items())))
            out["rest"].append(sq(GUIDED["novec"] * d["novec"] + GUIDED["uncond"] * d["uncond"]))
    model.train(was)
    return {f"drift_{k}": float(np.mean(v)) for k, v in out.items()}


@torch.no_grad()
def monitor(items):
    """The same noise and time steps at every call, so that the numbers of two calls can be compared."""
    was = model.training
    model.eval()
    gen, D, dw, dl, P = torch.Generator(device=dev).manual_seed(12345), [], [], [], []
    for q in items:
        d3, p3 = [], []
        for t in (0.2, 0.5, 0.8):
            d, em, er = margin(q, gen, t)
            d3.append(float(d)), dw.append(float(em[0] / er[0] - 1)), dl.append(float(em[1] / er[1] - 1)), p3.append(float(em[0] - em[1]))
        D.append(float(np.mean(d3)))  # one number per pair
        P.append(float(np.mean(p3)))  # the policy's error on the winner minus its error on the loser
    model.train(was)
    D = np.array(D)
    if not len(D):
        return dict(D=0.0, ordered=0.0, err_winner=0.0, err_loser=0.0, pref=0.0)
    return dict(D=float(D.mean()), ordered=float((D < 0).mean()), err_winner=float(np.mean(dw)), err_loser=float(np.mean(dl)), pref=float((np.array(P) < 0).mean()))


# ------------------------------------------------------------------------------- the real items of the SFT term
data = load_dataset(args.dataset, "pinyin", mel_spec_kwargs=OmegaConf.to_container(cfg.model.mel_spec))
assert data.prompt_index is not None, "the data set has no cross-emotion prompts"
real_rng = random.Random(args.seed + 1)  # its own generator


def real_item(i):  # one training item of the baseline: prompt and target of the same speaker, vector of the target's emotion
    b = collate_fn([data[i]])
    return dict(inp=b["mel"].permute(0, 2, 1).to(dev), text=b["text"], lens=b["mel_lengths"].to(dev), emo=b["emo"].to(dev), prompt_frames=b["prompt_frames"])


random.seed(999)
fixed_real = [real_item(i) for i in random.Random(999).sample(range(len(data)), 24)]  # for the monitor below


@torch.no_grad()
def real_error():
    """Monitor: flow error on 24 fixed real training items, relative to the reference, with fixed noise and time steps,
    in the three branches of inference (full condition, without the vector, unconditional)."""
    was = model.training
    model.eval()
    gen, rel = torch.Generator(device=dev).manual_seed(54321), dict(real=[], real_novec=[], real_uncond=[])
    for b in fixed_real:
        x1 = b["inp"]
        n = x1.shape[1]
        span = torch.arange(n, device=dev)[None] >= b["prompt_frames"].to(dev)[:, None]
        text = list_str_to_idx(b["text"], model.vocab_char_map).to(dev)
        for t in (0.2, 0.5, 0.8):
            x0 = torch.randn(1, n, x1.shape[2], device=dev, generator=gen)
            kw = dict(x=(1 - t) * x0 + t * x1, cond=torch.where(span[..., None], torch.zeros_like(x1), x1), text=text, time=torch.full((1,), t, device=dev),
                      mask=torch.ones(1, n, dtype=torch.bool, device=dev))
            for name, branch in (("real", dict(drop_audio_cond=False, drop_text=False, emo=b["emo"])), ("real_novec", dict(drop_audio_cond=False, drop_text=False, emo=None)),
                                 ("real_uncond", dict(drop_audio_cond=True, drop_text=True, emo=b["emo"]))):
                e = [float((((m.transformer(**kw, **branch) - (x1 - x0)) ** 2).mean(-1) * span).sum() / span.sum()) for m in (model, ref)]
                rel[name].append(e[0] / e[1] - 1)
    model.train(was)
    return {k: float(np.mean(v)) for k, v in rel.items()}


def measure():
    out = dict(**{f"val_{k}": v for k, v in monitor(val_pairs).items()}, **{f"train_{k}": v for k, v in monitor(watch_pairs).items()}, **real_error(),
               gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), gpu_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2))
    if len(names) > 1:  # the held-out pairs of each table (emotion) on their own
        for g, n in enumerate(names):
            m = monitor([q for q in val_pairs if q["group"] == g])
            out[f"val_D_{n}"], out[f"val_ordered_{n}"], out[f"val_pref_{n}"] = m["D"], m["ordered"], m["pref"]
    if args.anchor:
        out.update(own_drift())
    return out


# ------------------------------------------------------------------------------------------------- start checks
gen = torch.Generator(device=dev).manual_seed(args.seed + 1000)  # the pairs' noise: not the stream of the global generator
model.eval()
with torch.no_grad():
    first = [margin(q, gen) for q in train_pairs[:3]]
D0 = torch.stack([d for d, _, _ in first])
loss0 = float(-F.logsigmoid(-args.beta * D0).mean())
assert float(D0.abs().max()) == 0.0 and abs(loss0 - math.log(2)) < 1e-6, f"start check failed: D {D0.tolist()}, loss {loss0}"
start = dict(step=0, **measure())
assert start["val_D"] == 0.0 and start["train_D"] == 0.0 and start["real"] == 0.0 and start["real_novec"] == 0.0 and start["real_uncond"] == 0.0, f"a monitor is not 0 at the start: {start}"
if args.anchor:  # the anchor at the start: no drift in any branch, and no gradient from the term
    assert all(start[f"drift_{k}"] == 0.0 for k in ("full", "novec", "uncond", "guided", "rest")), f"the drift is not 0 at the start: {start}"
    d0 = drift_step(own_batch(own_item(own_train[0]), torch.Generator(device=dev).manual_seed(1)), 1.0)
    g0 = float(torch.sqrt(sum((q.grad.detach() ** 2).sum() for q in model.parameters() if q.grad is not None)))
    model.zero_grad(set_to_none=True)
    assert d0 == 0.0 and g0 == 0.0, f"anchor check failed: drift {d0}, gradient norm {g0}"
    print(f"anchor check passed: on the {len(own_val)} held-out own clips the drift is 0 in every branch; the term (guided) and its gradient are 0 at the start; "
          f"{len(own_train)} own clips for the term, {args.anchor_items} per update, weight {args.anchor_weight}", flush=True)
# memory: one pass with gradient through the longest training pair (no update), plus the state AdamW will hold
longest = max(train_pairs, key=lambda q: q["w"].shape[0] + q["l"].shape[0])
torch.cuda.reset_peak_memory_stats()
margin(longest, gen)[0].backward()
model.zero_grad(set_to_none=True)
need = (torch.cuda.max_memory_reserved() + 2 * sum(q.numel() * q.element_size() for q in model.parameters())) / 2**30
free = torch.cuda.mem_get_info()[0] / 2**30 + torch.cuda.memory_reserved() / 2**30
print(f"memory: longest pair {longest['w'].shape[0]} frames; this process will need about {need:.1f} GB (limit {args.mem_gb} GB); free on the card for it {free:.1f} GB", flush=True)
assert need < args.mem_gb and need < free, "not enough GPU memory: another job holds the card, or the limit is too low"
print(f"start check passed: D = 0 for every pair, loss = ln 2 = {loss0:.6f}; flow error of the first pairs: winner {[round(float(em[0]), 4) for _, em, _ in first]}, "
      f"loser {[round(float(em[1]), 4) for _, em, _ in first]}", flush=True)
print(f"at the start the flow error is lower on the winner than on the loser in {start['val_pref']:.2f} of the held-out pairs and in {start['train_pref']:.2f} of the watched training pairs"
      + ("" if len(names) == 1 else "; held-out pairs per table: " + ", ".join(f"{n} {start[f'val_pref_{n}']:.2f}" for n in names)), flush=True)
if args.check:
    raise SystemExit

# ----------------------------------------------------------------------------------------------------- training
run.mkdir(parents=True, exist_ok=True)
log = open(run / "log.jsonl", "a", encoding="utf-8", buffering=1)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # as the baseline's trainer (default weight decay), one learning rate (CHOICE)
grad_norm = lambda: float(torch.sqrt(sum((q.grad.detach() ** 2).sum() for q in model.parameters() if q.grad is not None)))
pair_rng, order, t0 = random.Random(args.seed + 2), [[] for _ in names], time.time()
own_rng, own_gen, own_order = random.Random(args.seed + 3), torch.Generator(device=dev).manual_seed(args.seed + 3000), []  # the anchor's own generators
first_step = 1


def code_version():  # the commit of the repository this file belongs to, and whether it has uncommitted changes
    try:
        here = Path(__file__).resolve().parent
        head = subprocess.run(["git", "-C", str(here), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--", "."], capture_output=True, text=True, timeout=20).stdout.strip()
        return f"{head}{' with uncommitted changes in experiments/preference' if dirty else ''}"
    except Exception as e:  # no git: the run still goes on
        return f"unknown ({e})"


def free_gb():
    return shutil.disk_usage(run).free / 2**30


def save_state(step):
    """Everything a resumed run needs to continue exactly: weights, optimiser and every random generator."""
    tmp = run / "state.tmp"
    torch.save(dict(step=step, model=model.state_dict(), optimizer=optimizer.state_dict(), pair_rng=pair_rng.getstate(), order=order,
                    own_rng=own_rng.getstate(), own_gen=own_gen.get_state(), own_order=own_order, real_rng=real_rng.getstate(), gen=gen.get_state(),
                    py_random=random.getstate(), np_random=np.random.get_state(), torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all(),
                    args=vars(args)), tmp)
    tmp.replace(run / "state.pt")


if args.resume:
    st = torch.load(run / "state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(st["model"])
    optimizer.load_state_dict(st["optimizer"])
    pair_rng.setstate(st["pair_rng"]), own_rng.setstate(st["own_rng"]), real_rng.setstate(st["real_rng"]), random.setstate(st["py_random"]), np.random.set_state(st["np_random"])
    own_gen.set_state(st["own_gen"]), gen.set_state(st["gen"]), torch.set_rng_state(st["torch_cpu"]), torch.cuda.set_rng_state_all(st["torch_cuda"])
    order, own_order, first_step = st["order"], st["own_order"], st["step"] + 1
    log.write(json.dumps(dict(event="resumed", step=st["step"], code=code_version(), time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
    print(f"resumed after update {st['step']} from {run / 'state.pt'}", flush=True)
    del st
else:
    log.write(json.dumps(dict(event="start", args=vars(args), code=code_version(), pairs_train=len(train_pairs), pairs_val=len(val_pairs), tables=dict(zip(names, map(len, groups))),
                              time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
    log.write(json.dumps(start) + "\n")
for step in range(first_step, args.steps + 1):
    if (run / "STOP").exists():  # written by the evaluation when a stop rule fires
        save_state(step - 1)
        log.write(json.dumps(dict(event="stopped by the file STOP", before_step=step, reason=(run / "STOP").read_text(encoding="utf-8").strip())) + "\n")
        print(f"stopped by {run / 'STOP'} before update {step}", flush=True)
        raise SystemExit
    rec = dict(step=step)
    for g in optimizer.param_groups:
        g["lr"] = args.lr * min(1.0, step / max(1, args.warmup))
    chosen = []
    for g, idx in enumerate(groups):  # the same number of pairs from every table
        if len(order[g]) < args.pairs_per_step:  # the pairs left over from the last pass first, then a new pass in a new order
            more = list(idx)
            pair_rng.shuffle(more)
            order[g] += more
        chosen += order[g][: args.pairs_per_step]
        order[g] = order[g][args.pairs_per_step:]
    model.eval()  # CHOICE: no dropout in the pair passes, so that D is exactly 0 while the policy equals the reference
    Ds, losses, rw, rl = [], [], [], []
    for i in chosen:
        D, em, er = margin(train_pairs[i], gen)
        loss = -F.logsigmoid(-args.beta * D)
        (loss / len(chosen)).backward()  # the mean over the pairs of this update
        em = em.detach()
        Ds.append(float(D.detach())), losses.append(float(loss.detach())), rw.append(float(em[0] / er[0] - 1)), rl.append(float(em[1] / er[1] - 1))
    rec.update(loss_dpo=float(np.mean(losses)), D=float(np.mean(Ds)), ordered=float(np.mean(np.array(Ds) < 0)), err_winner=float(np.mean(rw)),
               err_loser=float(np.mean(rl)), grad_dpo=grad_norm())
    if args.anchor_items:  # the anchor term: hold the guided velocity on the start model's own clips at the start model's
        model.eval()
        if len(own_order) < args.anchor_items:  # a new pass over the own clips in a new order
            more = list(own_train)
            own_rng.shuffle(more)
            own_order += more
        chosen_own, own_order = own_order[: args.anchor_items], own_order[args.anchor_items:]
        dr = [drift_step(own_batch(own_item(i), own_gen), args.anchor_weight / len(chosen_own)) for i in chosen_own]  # the mean over the anchor clips of this update
        rec.update(anchor_drift=float(np.mean(dr)), grad_with_anchor=grad_norm())
    sft = []
    model.train()  # the SFT term as in the baseline: dropout and the classifier-free dropout of prompt, text and emotion vector
    random.seed(real_rng.random())  # the data set draws the vector source and the model its dropout branch with the global generator
    for i in real_rng.sample(range(len(data)), args.sft_items):
        b = real_item(i)
        loss_sft, _, _ = model(b["inp"], text=b["text"], lens=b["lens"], emo=b["emo"], prompt_frames=b["prompt_frames"])
        (args.sft_weight * loss_sft / args.sft_items).backward()
        sft.append(float(loss_sft))
    rec.update(loss_sft=float(np.mean(sft)) if sft else 0.0, grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)))  # the norm before clipping
    if not all(math.isfinite(v) for v in rec.values()):  # no update with a broken loss or gradient
        log.write(json.dumps(dict(event="stopped: a loss or gradient is not finite", **rec)) + "\n")
        raise SystemExit(f"update {step}: a loss or gradient is not finite, stopped before the update: {rec}")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if step % args.val_every == 0 or step == args.steps:
        rec.update(measure())
        print(f"step {step:>4} ({(time.time() - t0) / (step - first_step + 1):.2f} s per step): " + ", ".join(f"{k} {v:+.5f}" for k, v in rec.items() if k != "step"), flush=True)
    log.write(json.dumps(rec) + "\n")
    if step in save_at or step % args.state_every == 0 or step == args.steps:
        if free_gb() < args.min_free_gb + 4:  # the state takes about 4 GB, the weights 1.4 GB
            log.write(json.dumps(dict(event=f"stopped: less than {args.min_free_gb + 4:.0f} GB free on the disk", step=step, free_gb=round(free_gb(), 1))) + "\n")
            raise SystemExit(f"update {step}: less than {args.min_free_gb + 4:.0f} GB free on the disk, stopped without saving")
    if step in save_at:
        torch.save({"model_state_dict": model.state_dict(), "step": step, "args": vars(args)}, run / f"model_{step}.pt")
        log.write(json.dumps(dict(event="saved", step=step, file=f"model_{step}.pt")) + "\n")
        print(f"saved {run / f'model_{step}.pt'}", flush=True)
    if step % args.state_every == 0 or step == args.steps:
        save_state(step)
log.write(json.dumps(dict(event="end", steps=args.steps, seconds=round(time.time() - t0, 1))) + "\n")
print("done", flush=True)
