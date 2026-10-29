#!/usr/bin/env bash
#SBATCH --job-name=warmstart
#SBATCH --time=00:30:00
#SBATCH --partition=students
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --chdir=/data/chi-gpu1/ge96xah/f5tts
#SBATCH --output=/nas/student/DavidFrank/f5tts/logs/%x_%A.out

# new experiment that starts from the online weights of a training checkpoint, with fresh optimizer and schedule (CPU only)
# the checkpoint dir of the new experiment (symlink to the NAS) has to exist and must not hold model_* or pretrained_* files
# usage: sbatch warmstart.sh <training checkpoint model_N.pt> <new experiment name>
direnv allow . && eval "$(direnv export bash)" || exit 1
python thesis/scripts/warmstart_from.py $1 thesis/F5-TTS/ckpts/$2 || exit 1  # else SLURM reports the ls below as success
ls -la thesis/F5-TTS/ckpts/$2/
