#!/usr/bin/env bash
#SBATCH --job-name=listen_pilot
#SBATCH --time=00:45:00
#SBATCH --partition=students
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir=/data/chi-gpu1/ge96xah/f5tts
#SBATCH --output=/nas/student/DavidFrank/f5tts/logs/%x_%A.out

# listening test on the last pilot checkpoint, wavs go to the NAS
# usage: sbatch listen_pilot.sh [emotion guidance strengths, e.g. 3,6] [output dir name]
G=${1:-0}
OUT=/nas/student/DavidFrank/f5tts/listen/${2:-pilot1}
direnv allow . && eval "$(direnv export bash)" || exit 1
python thesis/experiments/emotion_a/listen_pilot.py /nas/student/DavidFrank/f5tts/ckpts/ESD_en_pilot1/model_6270.pt $OUT $G
ls -la $OUT
