#!/usr/bin/env bash
#SBATCH --job-name=pref_ref
#SBATCH --time=03:00:00
#SBATCH --partition=students
#SBATCH --gres=gpu:rtx4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --chdir=/data/chi-gpu1/ge96xah/f5tts
#SBATCH --output=/nas/student/DavidFrank/f5tts/logs/%x_%A.out

# The reference of the preference stage on the cluster's cards (after pref_setup.sh): the evaluation set (six speakers,
# six evaluation sentences, five emotions) at the seed offsets 2 and 3,
#   start     the start model without the emotion input: the level every evaluated clip is scaled to
#   labelcal  the start model with the emotion input at the strengths of the pool, level matched and measured as every
#             evaluated state (both judges, words, pitch): what pref_train.sh reads every state against
# The trained states are generated on the same cards, so a state and its reference differ by the training alone.
# A folder that is complete is not generated again.
# usage: sbatch thesis/cluster/pref_ref.sh
[ $# -eq 0 ] || { echo "pref_ref.sh takes no arguments"; exit 1; }
N=/nas/student/DavidFrank
M=$N/models
PREF=$N/pref
CODE=/data/chi-gpu1/ge96xah/f5tts/thesis
P=$CODE/experiments/preference
ESD=/data/chi-gpu1/ge96xah/data/ESD
F5=$N/envs/f5bw/bin/python
JEV=$N/envs/jev/bin/python
STRENGTHS=angry=0.5,happy=2,sad=1.5,surprise=1
umask 022  # files of a job readable by the next: the NAS reports another owner for them, so a private umask locks them out
unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE  # a cache folder of the login shell (sbatch passes its environment on) must not win over the NAS
export HF_HOME=$N/hf_cache HF_HUB_CACHE=$N/hf_cache/hub HF_XET_CACHE=$N/hf_cache/xet HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$CODE/F5-TTS/src  # the F5 code of this repository, and nothing of the login shell's environment
set -ex
nvidia-smi --query-gpu=name,memory.total --format=csv
git -C $CODE log -1 --oneline
E=$PREF/eval
for k in 2 3; do
  $F5 $P/generate.py $E/start/s$k --ckpt $PREF/start_36666_online.pt --seed-offset $k --esd $ESD
  $F5 $P/generate.py $E/labelcal_raw/s$k --ckpt $PREF/start_36666_online.pt --strengths $STRENGTHS --seed-offset $k --esd $ESD
  [ -s $E/labelcal/s$k/manifest.csv ] || $F5 $P/level_match.py $E/labelcal_raw/s$k $E/labelcal/s$k $E/start/s$k/manifest.csv
  MF=$E/labelcal/s$k/manifest.csv
  [ -s $E/labelcal/s$k/scores_jev_lora4.csv ] || $JEV -I $CODE/scripts/score_audiojev.py --adapter $M/adapters/lora4 $M/AudioJev /nas /nas $MF
  [ -s $E/labelcal/s$k/scores_wavlm.csv ] || $F5 $CODE/scripts/score_wavlm_judge.py $N/f5tts/eval/emotions $MF
  [ -s $E/labelcal/s$k/wer.csv ] || $F5 $P/score_words.py --esd $ESD $MF
  [ -s $E/labelcal/s$k/pitch.csv ] || $JEV -I $P/score_pitch.py clips $PREF/frame.csv $MF
done
echo "reference done"
