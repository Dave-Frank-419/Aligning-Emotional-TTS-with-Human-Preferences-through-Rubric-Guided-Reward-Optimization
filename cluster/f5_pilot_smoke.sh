#!/usr/bin/env bash
#SBATCH --job-name=f5_pilot_smoke
#SBATCH --time=00:45:00
#SBATCH --partition=students
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --chdir=/data/chi-gpu1/ge96xah/f5tts
#SBATCH --output=/nas/student/DavidFrank/f5tts/logs/%x_%A.out

# same run as f5_pilot.sh, stopped after the first save at update 100
# checks saving to the NAS, sample generation, tensorboard and that emo_proj trains
# usage: sbatch f5_pilot_smoke.sh [experiment name] [extra finetune_cli options, e.g. --emo_drop_prob 0.1]
EXP=${1:-ESD_en_smoke}
CK=/data/chi-gpu1/ge96xah/f5tts/thesis/F5-TTS/ckpts/$EXP  # symlink to the NAS

direnv allow . && eval "$(direnv export bash)" || exit 1
export ACCELERATE_MIXED_PRECISION=no
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # varying batch shapes otherwise fragment the 12 GB
python -c "import f5_tts.model.backbones.dit as d; print(d.__file__); assert '/thesis/F5-TTS/src/' in d.__file__" || exit 1
python -c "import tensorboard" || exit 1
ls -la $CK/ || exit 1
cd /nas/student/DavidFrank/f5tts/runs/$EXP || exit 1  # tensorboard writes to ./runs

( while sleep 15; do echo "$(date +%T) $(ls -l $CK/ | grep '\.pt' | awk '{print $9, $5}' | xargs)"; done ) &  # save speed to the NAS
MON=$!
python /data/chi-gpu1/ge96xah/f5tts/thesis/F5-TTS/src/f5_tts/train/finetune_cli.py --exp_name F5TTS_v1_Base --dataset_name $EXP --finetune --pretrain /data/chi-gpu1/ge96xah/f5tts/thesis/F5-TTS/ckpts/ESD_en/pretrained_model_1250000.safetensors --tokenizer pinyin --batch_size_per_gpu 3200 --batch_size_type frame --max_samples 64 --learning_rate 1e-5 --emo_lr_scale 10 --num_warmup_updates 500 --epochs 5 --save_per_updates 100 --last_per_updates 100000 --keep_last_n_checkpoints -1 --log_samples --logger tensorboard "${@:2}" &
PID=$!
while kill -0 $PID 2>/dev/null && [ ! -f $CK/samples/update_100_ref.wav ]; do sleep 5; done
sleep 10
kill $PID $MON

ls -la $CK/ $CK/samples/ runs/F5TTS_v1_Base/
unzip -tq $CK/model_100.pt
python /data/chi-gpu1/ge96xah/f5tts/thesis/scripts/check_ckpt.py $CK/model_100.pt
