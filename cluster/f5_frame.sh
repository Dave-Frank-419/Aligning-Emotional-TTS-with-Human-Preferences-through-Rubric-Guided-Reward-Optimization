#!/usr/bin/env bash
#SBATCH --job-name=f5_frame
#SBATCH --time=10:00:00
#SBATCH --partition=students
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --chdir=/data/chi-gpu1/ge96xah/f5tts
#SBATCH --output=/nas/student/DavidFrank/f5tts/logs/%x_%A.out

# ablation of the emotion input: 7 more epochs from ESD_en_xprompt1_cont/model_36666.pt with the second projection
# (emo_proj_frame in dit.py, the vector at every frame)
# control: model_54999.pt, the same 7 epochs without it (a plain resume: optimizer and EMA kept, no second warm-up)
# same data, pairs and options as the continuation run; --emo_lr_scale 10 covers both projections (1e-4, backbone 1e-5)
# every option is written out here, nothing is typed after the script name
# before, once (N=/nas/student/DavidFrank/f5tts, F5 as below):
#   mkdir -p $N/ckpts/ESD_en_xprompt1_frame $N/runs/ESD_en_xprompt1_frame
#   ln -s $N/ckpts/ESD_en_xprompt1_frame $F5/ckpts/ESD_en_xprompt1_frame
#   ln -s $F5/data/ESD_en_xprompt1_pinyin $F5/data/ESD_en_xprompt1_frame_pinyin
#   sbatch warmstart.sh $N/ckpts/ESD_en_xprompt1_cont/model_36666.pt ESD_en_xprompt1_frame
# usage: sbatch f5_frame.sh
[ $# -eq 0 ] || { echo "f5_frame.sh takes no arguments"; exit 1; }
EXP=ESD_en_xprompt1_frame
F5=/data/chi-gpu1/ge96xah/f5tts/thesis/F5-TTS
CK=$F5/ckpts/$EXP  # symlink to the NAS
START=$CK/pretrained_ESD_en_xprompt1_cont_model_36666_online.pt  # online weights of model_36666.pt, written by warmstart.sh

direnv allow . && eval "$(direnv export bash)" || exit 1
export ACCELERATE_MIXED_PRECISION=no
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # varying batch shapes otherwise fragment the 12 GB
python -c "import f5_tts.model.backbones.dit as d; print(d.__file__); assert '/thesis/F5-TTS/src/' in d.__file__; assert hasattr(d.TextEmbedding(8, 8), 'emo_proj_frame'), 'code without the second projection'" || exit 1
python -c "import tensorboard" || exit 1
git -C /data/chi-gpu1/ge96xah/f5tts/thesis log -1 --oneline  # the code of this run ...
git -C /data/chi-gpu1/ge96xah/f5tts/thesis status --short --untracked-files=no  # ... and uncommitted changes to it (no line = none)
readlink -f $CK  # has to lie on the NAS, /data is full
ls -la $CK/ || exit 1
[ -s $START ] || { echo "start file $START missing: remove any other pretrained_* file listed above, then run warmstart.sh on ESD_en_xprompt1_cont/model_36666.pt"; exit 1; }
[ -s $F5/data/${EXP}_pinyin/pairs.json ] || { echo "no pairs.json in the data folder: training would run without cross-emotion prompts"; exit 1; }
[ $F5/data/${EXP}_pinyin/pairs.json -ef $F5/data/ESD_en_xprompt1_cont_pinyin/pairs.json ] || { echo "pairs.json is not the file of the continuation run"; exit 1; }
cd /nas/student/DavidFrank/f5tts/runs/$EXP || exit 1  # tensorboard writes to ./runs
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv -l 30 > /nas/student/DavidFrank/f5tts/logs/gpu_mem_$SLURM_JOB_ID.csv &
SMI=$!
python $F5/src/f5_tts/train/finetune_cli.py --exp_name F5TTS_v1_Base --dataset_name $EXP --finetune --pretrain $START --tokenizer pinyin --batch_size_per_gpu 3200 --batch_size_type frame --max_samples 64 --learning_rate 1e-5 --emo_lr_scale 10 --num_warmup_updates 500 --constant_lr --emo_drop_prob 0.1 --ema_on_cpu --epochs 7 --save_per_updates 2619 --last_per_updates 100000 --keep_last_n_checkpoints -1 --log_samples --logger tensorboard
RC=$?  # status of the training; else SLURM reports the kill below as the result of the job
kill $SMI
exit $RC
