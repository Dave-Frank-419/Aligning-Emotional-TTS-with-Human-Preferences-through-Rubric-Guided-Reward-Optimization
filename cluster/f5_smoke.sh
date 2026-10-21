#!/usr/bin/env bash
#SBATCH --job-name=f5_smoke
#SBATCH --time=00:20:00
#SBATCH --partition=students
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=./%x_%A.out

direnv allow . && eval "$(direnv export bash)"
export HF_HOME=/data/chi-gpu1/ge96xah/hf_cache

nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
python -c "from f5_tts.infer.utils_infer import load_vocoder, load_model; from f5_tts.model import CFM, DiT; print('f5 imports ok')"
f5-tts_infer-cli -r /data/chi-gpu1/ge96xah/data/ESD/0011/Neutral/train/0011_000280.wav -s "Your path now goes south." -t "The meeting has been moved to Thursday afternoon." -o /data/chi-gpu1/ge96xah/f5tts/outputs -w smoke_esd_0011.wav
