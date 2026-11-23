import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from audiojev_inference import AudioInput, AudioJev, Choice, Noul
from scipy.signal import butter, resample_poly, sosfiltfilt

# the verifier AudioJev (arXiv 2610.01293) on the items of the emotion evaluation, in two readouts:
#   noul:   one statement per emotion ("The speaker sounds happy."), P(true) as score (a yes/no verification)
#   choice: one question with the five emotions as options, the probability of each option as score
# --units: factual statements instead (pitch, pitch range, loudness, rate, pitch rise at the end, gender as control), one
#          P(true) per statement; checked against eGeMAPS measurements by units_report.py (history_m2 step 1)
# --relative <ESD root>: the same units relative to the item's neutral prompt (the clip F5 is given): the prompt is played
#          first, then 0.5 s of silence, then the clip; duration as control
# --tone, --lowpass, --tone-lowpass: does AudioJev go by the words, which are neutral in ESD? (history_m2 step 1, option B test 0)
#          the emotion questions, prefixed with "Ignore the words and judge only the tone of voice.", and/or the clip low-passed
#          at 500 Hz, so that the words are hard to make out while pitch stays (ESD women up to about 450 Hz); the filtered clip
#          gets the clip's RMS level back, so that level differences between clips stay (unmatched, it loses 4 to 6 dB, more
#          for angry than for sad); the spectral cues above 500 Hz are gone either way
# --adapter <dir>: AudioJev with a LoRA adapter from train_audiojev_lora.py (option B test 2); "_" and the adapter's folder
#          name are appended to the score file names, e.g. scores_jevchoice_lora4.csv
# AudioJev needs BF16 and its own venv (torch 2.8, transformers 4.57.6), so it runs on cards with BF16 (the local RTX
# 4070, or the cluster's RTX PRO 4000 Blackwell in the environment jev of cluster/pref_setup.sh), not on the TITAN X
# the manifests may hold the paths of another machine: the clips are read with <cluster prefix> replaced by <local
# prefix>, the scores keep the manifest path for the report (where the clips lie where the manifest says, both are /nas)
# usage: python score_audiojev.py [--adapter <dir>] [--units | --relative <ESD root> | --tone | --lowpass | --tone-lowpass] <model dir>
#        <cluster prefix> <local prefix> <manifest.csv> [...] -> next to each manifest: scores_jev.csv and scores_jevchoice.csv
#        (with the variant's name appended, e.g. scores_jevchoicetone.csv), scores_jevunits.csv or scores_jevrelative.csv
argv = sys.argv[1:]
adapter = Path(argv[1]) if argv[0] == "--adapter" else None
argv = argv[2:] if adapter else argv
mode = argv[0] if argv[0] in ("--units", "--relative", "--tone", "--lowpass", "--tone-lowpass") else ""
args = argv[2 if mode == "--relative" else 1 if mode else 0:]
esd = Path(argv[1]) if mode == "--relative" else None
tone, low = mode in ("--tone", "--tone-lowpass"), mode in ("--lowpass", "--tone-lowpass")
variant = (mode[2:].replace("-", "") if tone or low else "") + ("_" + adapter.resolve().name if adapter else "")  # names the score files
lowpass = butter(8, 500, fs=16000, output="sos")
model_dir, remote, local, manifests = args[0], args[1], args[2], [Path(m) for m in args[3:]]
emotions = {"neutral": "emotionally neutral", "angry": "angry", "happy": "happy", "sad": "sad", "surprise": "surprised"}
statements = {  # in pairs, so that a unit's score can be P(first) - P(second)
    "--units": {
        "pitch_high": "The speaker's voice has a high pitch.", "pitch_low": "The speaker's voice has a low pitch.",
        "range_wide": "The speaker's pitch varies a lot.", "range_flat": "The speaker's pitch stays almost flat.",
        "loud": "The speaker speaks loudly.", "quiet": "The speaker speaks quietly.",
        "fast": "The speaker speaks fast.", "slow": "The speaker speaks slowly.",
        "rise_end": "The pitch rises at the end of the utterance.",
        "female": "The speaker is female.", "male": "The speaker is male.",
    },
    "--relative": {
        "higher": "The pitch in the second part is higher than in the first part.",
        "lower": "The pitch in the second part is lower than in the first part.",
        "varies_more": "The pitch in the second part varies more than in the first part.",
        "varies_less": "The pitch in the second part varies less than in the first part.",
        "louder": "The speaker speaks louder in the second part than in the first part.",
        "quieter": "The speaker speaks more quietly in the second part than in the first part.",
        "faster": "The speaker speaks faster in the second part than in the first part.",
        "slower": "The speaker speaks more slowly in the second part than in the first part.",
        "rise_end": "The pitch rises at the end of the second part.",
        "longer": "The second part is longer than the first part.", "shorter": "The second part is shorter than the first part.",
    },
}.get(mode)
outfile = {"--units": f"scores_jevunits{variant}.csv", "--relative": f"scores_jevrelative{variant}.csv"}.get(mode)
if statements:
    questions = {k: Noul(s) for k, s in statements.items()}
else:
    ask = "Ignore the words and judge only the tone of voice. " if tone else ""
    questions = {e: Noul(f"{ask}The speaker sounds {u}.") for e, u in emotions.items()}
    questions["choice"] = Choice(f"{ask}Which emotion does the speaker express?", emotions)
jev = AudioJev(model_dir, adapter=adapter, device="cuda:0", local_files_only=True)


def mono16k(path):  # both parts of a relative item must share one rate; AudioJev works at 16 kHz
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    g = math.gcd(sr, 16000)
    return resample_poly(x.mean(1), 16000 // g, sr // g).astype(np.float32)


for manifest in manifests:
    with open(manifest, encoding="utf-8", newline="") as f:
        items = list(csv.DictReader(f))
    start = time.time()
    noul, choice = f"scores_jev{variant}.csv", f"scores_jevchoice{variant}.csv"
    out = {outfile: []} if statements else {noul: [], choice: []}
    for it in items:
        clip = it["path"].replace(remote, local, 1)
        assert Path(clip).is_file(), f"{clip} not found: check the cluster and local prefixes"
        state = clip
        if mode == "--relative":  # the levels stay as recorded, so that loudness can be compared
            parts = [mono16k(esd / it["prompt"]), np.zeros(8000, np.float32), mono16k(clip)]
            state = AudioInput(waveform=np.concatenate(parts), sample_rate=16000)
        if low:  # zero phase, so the pitch movement keeps its timing; then back to the clip's RMS level
            x = mono16k(clip)
            y = sosfiltfilt(lowpass, x)
            state = AudioInput(waveform=(y * np.sqrt(np.mean(x**2) / np.mean(y**2))).astype(np.float32), sample_rate=16000)
        answers = jev.system_one(state=state, questions=questions)["answers"]
        if statements:
            out[outfile].append([it["path"]] + [f"{answers[k]['noul']:.6g}" for k in statements])
        else:
            out[noul].append([it["path"]] + [f"{answers[e]['noul']:.6g}" for e in emotions])
            out[choice].append([it["path"]] + [f"{answers['choice']['probabilities'][e]:.6g}" for e in emotions])
    for name, rows in out.items():
        with open(manifest.parent / name, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path"] + list(statements or emotions))
            w.writerows(rows)
    took = time.time() - start
    print(f"scored {len(items)} in {took:.0f} s ({took / len(items):.2f} s per clip) ->", manifest.parent, flush=True)
