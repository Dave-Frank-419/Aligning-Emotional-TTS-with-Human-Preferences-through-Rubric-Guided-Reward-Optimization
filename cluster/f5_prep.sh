#!/usr/bin/env bash
#SBATCH --job-name=f5_prep
#SBATCH --time=00:30:00
#SBATCH --partition=students
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=./%x_%A.out

direnv allow . && eval "$(direnv export bash)"
python thesis/F5-TTS/src/f5_tts/train/datasets/prepare_csv_wavs.py /data/chi-gpu1/ge96xah/data/ESD/metadata_train.csv /data/chi-gpu1/ge96xah/f5tts/thesis/F5-TTS/data/ESD_en_pinyin --workers 8
ls -la /data/chi-gpu1/ge96xah/f5tts/thesis/F5-TTS/data/ESD_en_pinyin
