#!/usr/bin/env bash
#SBATCH --job-name=pref_smoke
#SBATCH --time=01:30:00
#SBATCH --partition=students
#SBATCH --gres=gpu:rtx4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --chdir=/data/chi-gpu1/ge96xah/f5tts
#SBATCH --output=/nas/student/DavidFrank/f5tts/logs/%x_%A.out

# Test of the preference stage on a Blackwell card after pref_setup.sh, before any run (nothing is kept but the log):
#   card         both environments see the card and compute on it
#   generation   speaker 0011, the first evaluation sentence in all five emotions, seed offset 2, with the emotion input
#   scoring      the reward's classifier (AudioJev with lora4, environment jev), the second judge and the word errors
#                (f5bw), the pitch against the frame (jev)
#   trainer      train_dpo.py --check, then two updates with saving and resuming, on the pairs and the anchor table of the
#                first archive (pack_pairs.py: the pilot's pairs and the start model's neutral clips, unpacked into $PREF)
# usage: sbatch thesis/cluster/pref_smoke.sh
[ $# -eq 0 ] || { echo "pref_smoke.sh takes no arguments"; exit 1; }
N=/nas/student/DavidFrank
M=$N/models
PREF=$N/pref
CODE=/data/chi-gpu1/ge96xah/f5tts/thesis
P=$CODE/experiments/preference
ESD=/data/chi-gpu1/ge96xah/data/ESD
F5=$N/envs/f5bw/bin/python
JEV=$N/envs/jev/bin/python
S=$PREF/smoke_$SLURM_JOB_ID
umask 022  # files of a job readable by the next: the NAS reports another owner for them, so a private umask locks them out
unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE  # a cache folder of the login shell (sbatch passes its environment on) must not win over the NAS
export HF_HOME=$N/hf_cache HF_HUB_CACHE=$N/hf_cache/hub HF_XET_CACHE=$N/hf_cache/xet HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$CODE/F5-TTS/src  # the F5 code of this repository, and nothing of the login shell's environment
set -ex
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
git -C $CODE log -1 --oneline
$F5 -c "import torch; x = torch.randn(512, 512, device='cuda'); print('f5bw', torch.cuda.get_device_name(0), float((x @ x).sum()))"
$F5 -c "import torch, f5_tts; print('f5bw imports torch', torch.__version__, 'from', torch.__file__, 'and F5 from', list(f5_tts.__path__))"
$JEV -I -c "import torch; x = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16); print('jev', torch.cuda.get_device_name(0), float((x @ x).float().sum()))"
$F5 $P/pack_pairs.py --check $PREF  # the unpacked archive is complete and unchanged

$F5 $P/generate.py $S/gen --ckpt $PREF/start_36666_online.pt --strengths angry=0.5,happy=2,sad=1.5,surprise=1 --seed-offset 2 --speakers 0011 --slots 0 --esd $ESD
$JEV -I $CODE/scripts/score_audiojev.py --adapter $M/adapters/lora4 $M/AudioJev /nas /nas $S/gen/manifest.csv
$F5 $CODE/scripts/score_wavlm_judge.py $N/f5tts/eval/emotions $S/gen/manifest.csv
$F5 $P/score_words.py --esd $ESD $S/gen/manifest.csv
$JEV -I $P/score_pitch.py clips $PREF/frame.csv $S/gen/manifest.csv
head -6 $S/gen/scores_jev_lora4.csv $S/gen/scores_wavlm.csv $S/gen/wer.csv $S/gen/pitch.csv

T=$PREF/tables
PAIRS=$T/pairs_pool_angry.csv,$T/pairs_pool_happy.csv,$T/pairs_pool_sad.csv,$T/pairs_pool_surprise.csv
COMMON="--start $PREF/start_36666_online.pt --dataset ESD_en_xprompt1_cont --esd $ESD --pairs $PAIRS --beta 100 --warmup 75 --level-match --anchor $T/own_clips_neutral.csv --anchor-weight 800 --anchor-items 8 --val-every 2 --val-max 2 --mem-gb 20"
$F5 $P/train_dpo.py --run $S/run $COMMON --check
$F5 $P/train_dpo.py --run $S/run $COMMON --steps 1 --save 1
$F5 $P/train_dpo.py --run $S/run $COMMON --steps 2 --save 2 --resume
grep -E '"step": 2|resumed' $S/run/log.jsonl | cut -c1-300
rm -rf $S
echo "smoke test passed"
