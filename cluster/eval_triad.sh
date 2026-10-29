#!/usr/bin/env bash
#SBATCH --job-name=eval_triad
#SBATCH --time=03:00:00
#SBATCH --partition=students
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir=/data/chi-gpu1/ge96xah/f5tts
#SBATCH --output=/nas/student/DavidFrank/f5tts/logs/%x_%A.out

# triad evaluation of one arm (history_m1 section 14): generate the fixed items, score them with both judges, update
# the report over all arms evaluated so far; for checkpoints with vector also a blind listening test
# run eval_setup.sh once before; run the arm "real" first and alone, it builds the WavLM features that the other jobs reuse
# usage: sbatch eval_triad.sh <arm name> <real|vocoded|prompt|ckpt|novec> [checkpoint .pt] [--nfe 32]
NAME=$1
ARM=$2
shift 2
EV=/nas/student/DavidFrank/f5tts/eval/triad
E2V_PY=/nas/student/DavidFrank/envs/e2v/bin/python  # own environment for emotion2vec+ (funasr)
E2V_MODEL=/nas/student/DavidFrank/models/emotion2vec_plus_large
WAVLM_MODEL=/nas/student/DavidFrank/models/wavlm-large  # read by score_wavlm_judge.py
direnv allow . && eval "$(direnv export bash)" || exit 1
export ACCELERATE_MIXED_PRECISION=no
# both judges are needed; without its weights funasr would not fail but score with a random network
[ -x $E2V_PY ] && [ -s $E2V_MODEL/model.pt ] && [ -s $WAVLM_MODEL/pytorch_model.bin ] || { echo "run eval_setup.sh first"; exit 1; }
mkdir -p $EV
python thesis/experiments/emotion_a/eval_triad_generate.py $EV/$NAME $ARM "$@" || exit 1
rm -f $EV/$NAME/scores_wavlm.csv $EV/$NAME/scores_e2v.csv  # never report scores of earlier audio
python thesis/scripts/score_wavlm_judge.py $EV $EV/$NAME/manifest.csv || exit 1
$E2V_PY -I thesis/scripts/score_emotion2vec.py $E2V_MODEL $EV/$NAME/manifest.csv || exit 1  # -I: ignore the F5 PYTHONPATH
if [ "$ARM" = ckpt ] && [ ! -d $EV/blind_$NAME ]; then
    python thesis/scripts/make_blind_test.py make $EV/$NAME $EV/blind_$NAME
fi
python thesis/scripts/eval_triad_report.py $EV pilotB_13095
