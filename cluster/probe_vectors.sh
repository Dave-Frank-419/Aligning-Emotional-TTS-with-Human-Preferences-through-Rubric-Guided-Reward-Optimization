#!/usr/bin/env bash
#SBATCH --job-name=probe_vectors
#SBATCH --time=01:00:00
#SBATCH --partition=students
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --chdir=/data/chi-gpu1/ge96xah/f5tts
#SBATCH --output=/nas/student/DavidFrank/f5tts/logs/%x_%A.out

# step 1, CPU only: does the emotional-vits vector separate angry, happy and surprise, and does pilot B's emo_proj pass it on
# usage: sbatch probe_vectors.sh [checkpoint dir of the pilot]
CK=${1:-/nas/student/DavidFrank/f5tts/ckpts/ESD_en_xprompt1}
direnv allow . && eval "$(direnv export bash)" || exit 1
python thesis/scripts/probe_emotion_vectors.py $CK /nas/student/DavidFrank/f5tts/probe
