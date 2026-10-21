#!/usr/bin/env bash
#SBATCH --job-name=extract_emotion
#SBATCH --time=01:00:00
#SBATCH --partition=students
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=./%x_%A.out

direnv allow . && eval "$(direnv export bash)"
python thesis/scripts/extract_emotion.py
