#!/usr/bin/env bash
#SBATCH --job-name=f5_pilot
#SBATCH --time=10:00:00
#SBATCH --partition=students
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --chdir=/data/chi-gpu1/ge96xah/f5tts
#SBATCH --output=/nas/student/DavidFrank/f5tts/logs/%x_%A.out

# pilot: 5 epochs on ESD English, one checkpoint per epoch
# checkpoints, log and tensorboard go to the NAS, /data is full
# usage: sbatch f5_pilot.sh [experiment name] [updates per epoch] [extra finetune_cli options, e.g. --emo_drop_prob 0.1]
EXP=${1:-ESD_en_pilot1}
SAVE=${2:-1254}
CK=/data/chi-gpu1/ge96xah/f5tts/thesis/F5-TTS/ckpts/$EXP  # symlink to the NAS

direnv allow . && eval "$(direnv export bash)" || exit 1
export ACCELERATE_MIXED_PRECISION=no
python -c "import f5_tts.model.backbones.dit as d; print(d.__file__); assert '/thesis/F5-TTS/src/' in d.__file__" || exit 1
python -c "import tensorboard" || exit 1
ls -la $CK/ || exit 1
cd /nas/student/DavidFrank/f5tts/runs/$EXP || exit 1  # tensorboard writes to ./runs
python /data/chi-gpu1/ge96xah/f5tts/thesis/F5-TTS/src/f5_tts/train/finetune_cli.py --exp_name F5TTS_v1_Base --dataset_name $EXP --finetune --pretrain /data/chi-gpu1/ge96xah/f5tts/thesis/F5-TTS/ckpts/ESD_en/pretrained_model_1250000.safetensors --tokenizer pinyin --batch_size_per_gpu 3200 --batch_size_type frame --max_samples 64 --learning_rate 1e-5 --emo_lr_scale 10 --num_warmup_updates 500 --epochs 5 --save_per_updates $SAVE --last_per_updates 100000 --keep_last_n_checkpoints -1 --log_samples --logger tensorboard "${@:3}"
