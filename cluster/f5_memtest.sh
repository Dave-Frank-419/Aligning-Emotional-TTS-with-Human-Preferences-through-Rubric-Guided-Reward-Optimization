#!/usr/bin/env bash
#SBATCH --job-name=f5_memtest
#SBATCH --time=00:25:00
#SBATCH --partition=students
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=./%x_%A.out

direnv allow . && eval "$(direnv export bash)"
export ACCELERATE_MIXED_PRECISION=no
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 5 > gpu_mem_${SLURM_JOB_ID}.csv &
SMI=$!
timeout 600 python F5-TTS/src/f5_tts/train/finetune_cli.py --exp_name F5TTS_v1_Base --dataset_name ESD_en --finetune --tokenizer pinyin --batch_size_per_gpu 3200 --batch_size_type frame --max_samples 64 --epochs 1 --num_warmup_updates 200 --save_per_updates 100000 --last_per_updates 100000 --keep_last_n_checkpoints 0
kill $SMI
sort -t, -k2 -n gpu_mem_${SLURM_JOB_ID}.csv | tail -n 3
