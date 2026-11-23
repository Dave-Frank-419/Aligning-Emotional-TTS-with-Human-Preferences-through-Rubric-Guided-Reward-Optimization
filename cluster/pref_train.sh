#!/usr/bin/env bash
#SBATCH --job-name=pref_train
#SBATCH --time=24:00:00
#SBATCH --partition=students
#SBATCH --gres=gpu:rtx4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --chdir=/data/chi-gpu1/ge96xah/f5tts
#SBATCH --output=/nas/student/DavidFrank/f5tts/logs/%x_%A.out

# One run of the preference stage on a Blackwell card (after pref_setup.sh, pref_ref.sh and the unpacked pairs of the
# pool, pack_pairs.py), in parts of 300 updates up to 1,500. After each part the saved state is evaluated against the
# reference labelcal of pref_ref.sh: generate.py on the evaluation set at the seed offsets 2 and 3 with the emotion
# input, level_match.py, both judges, the words and the pitch, and evaluate.py (bars E1 to E5, stop rules S1 to S3).
#   reward   the pairs of the reward (make_pairs.py: pairs_<emotion>.csv); a stop rule that fires writes STOP into the
#            run folder and the run ends after that part
#   random   the control on random pairs of the same items (pairs_random_<emotion>.csv); it is evaluated the same way but
#            not stopped by the rules; with a third argument (the name of the reward's run) it ends as soon as that run
#            has ended or stopped
# Training as the large run big1 (notebook, 2026-11-23): beta 100, levels matched in a pair, the anchor on the start
# model's own neutral clips (lambda 800, 8 per update), the SFT term, learning rate 1e-5, warm-up 75, 8 pairs per emotion
# and update. A part that is done (its weights and its evaluation exist) is not done again; the trainer resumes from
# its state, so the job can be submitted again after an interruption.
# usage: sbatch thesis/cluster/pref_train.sh <run name> <reward|random> [<name of the reward's run, for random>]
set -o pipefail  # a failing step before a pipe stops the job
[ $# -ge 2 ] || { echo "usage: sbatch pref_train.sh <run name> <reward|random> [<reward run to follow>]"; exit 1; }
NAME=$1; KIND=$2; FOLLOW=${3:-}
case "$KIND" in reward) SUF="" ;; random) SUF="_random" ;; *) echo "the kind is reward or random"; exit 1 ;; esac
N=/nas/student/DavidFrank
M=$N/models
PREF=$N/pref
CODE=/data/chi-gpu1/ge96xah/f5tts/thesis
P=$CODE/experiments/preference
ESD=/data/chi-gpu1/ge96xah/data/ESD
F5=$N/envs/f5bw/bin/python
JEV=$N/envs/jev/bin/python
STRENGTHS=angry=0.5,happy=2,sad=1.5,surprise=1
T=$PREF/tables
E=$PREF/eval
RUN=$PREF/runs/$NAME
unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE  # a cache folder of the login shell (sbatch passes its environment on) must not win over the NAS
export HF_HOME=$N/hf_cache HF_HUB_CACHE=$N/hf_cache/hub HF_XET_CACHE=$N/hf_cache/xet HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$CODE/F5-TTS/src  # the F5 code of this repository, and nothing of the login shell's environment
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:128  # as every local training
nvidia-smi --query-gpu=name,memory.total --format=csv
git -C $CODE log -1 --oneline
git -C $CODE status --short --untracked-files=no  # uncommitted changes to the code of this run (no line = none)
TABLES=$T/pairs${SUF}_angry.csv,$T/pairs${SUF}_happy.csv,$T/pairs${SUF}_sad.csv,$T/pairs${SUF}_surprise.csv
for f in ${TABLES//,/ } $T/anchor_neutral.csv $PREF/start_36666_online.pt $PREF/frame.csv $E/labelcal/s3/pitch.csv; do
  [ -s $f ] || { echo "missing: $f (pref_setup.sh, pref_ref.sh, and the unpacked pairs come first)"; exit 1; }
done
mkdir -p $RUN
PREV=""
for STEP in 300 600 900 1200 1500; do
  [ -e $RUN/STOP ] && { echo "$NAME: stopped ($(cat $RUN/STOP))"; break; }
  if [ -n "$FOLLOW" ] && { [ -e $PREF/runs/$FOLLOW/STOP ] || [ -e $PREF/runs/$FOLLOW/done ]; }; then
    echo "$NAME: the run $FOLLOW has ended, so does the control"; break
  fi
  if [ ! -s $RUN/model_$STEP.pt ]; then
    RES=""
    [ -s $RUN/state.pt ] && RES=--resume
    $F5 $P/train_dpo.py --run $RUN --start $PREF/start_36666_online.pt --dataset ESD_en_xprompt1_cont --esd $ESD --pairs $TABLES --beta 100 \
      --steps $STEP --save $STEP --state-every 100 --warmup 75 --pairs-per-step 8 --level-match --anchor $T/anchor_neutral.csv --anchor-weight 800 \
      --anchor-items 8 --val-every 50 --val-max 12 --mem-gb 20 $RES || { echo "$NAME: training failed at the part up to $STEP"; exit 1; }
    [ -s $RUN/model_$STEP.pt ] || { echo "$NAME: no weights after the part up to $STEP (stopped?)"; exit 1; }
  fi
  TAG=${NAME}_$STEP
  if [ ! -s $RUN/eval_$STEP.json ]; then
    for k in 2 3; do
      if [ ! -s $E/$TAG/s$k/manifest.csv ]; then
        $F5 $P/generate.py $E/${TAG}_raw/s$k --ckpt $RUN/model_$STEP.pt --strengths $STRENGTHS --seed-offset $k --esd $ESD || exit 1
        $F5 $P/level_match.py $E/${TAG}_raw/s$k $E/$TAG/s$k $E/start/s$k/manifest.csv || exit 1
        rm -rf $E/${TAG}_raw/s$k
      fi
      MF=$E/$TAG/s$k/manifest.csv
      [ -s $E/$TAG/s$k/scores_jev_lora4.csv ] || $JEV -I $CODE/scripts/score_audiojev.py --adapter $M/adapters/lora4 $M/AudioJev /nas /nas $MF || exit 1
      [ -s $E/$TAG/s$k/scores_wavlm.csv ] || $F5 $CODE/scripts/score_wavlm_judge.py $N/f5tts/eval/emotions $MF || exit 1
      [ -s $E/$TAG/s$k/wer.csv ] || $F5 $P/score_words.py --esd $ESD $MF || exit 1
      [ -s $E/$TAG/s$k/pitch.csv ] || $JEV -I $P/score_pitch.py clips $PREF/frame.csv $MF || exit 1
    done
    STOPS=""
    [ $KIND = reward ] && STOPS="--run $RUN"
    $F5 $P/evaluate.py --arm $E/$TAG --ref $E/labelcal --step $STEP ${PREV:+--prev $PREV} $STOPS --out $RUN/eval_$STEP.json | tee $RUN/eval_$STEP.txt || exit 1
  fi
  PREV=$RUN/eval_$STEP.json
done
date > $RUN/done
echo "$NAME: done"
