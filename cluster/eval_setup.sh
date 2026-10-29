#!/usr/bin/env bash
#SBATCH --job-name=eval_setup
#SBATCH --time=01:30:00
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=/data/chi-gpu1/ge96xah/f5tts
#SBATCH --output=/nas/student/DavidFrank/f5tts/logs/%x_%A.out

# one-time preparation of the triad evaluation (CPU only), everything on the NAS because /data is nearly full:
# an own environment for emotion2vec+, so that funasr cannot touch the F5 environment, and the weights of both judges
# can be repeated after a failure, finished steps are skipped
# usage: sbatch eval_setup.sh
E2V=/nas/student/DavidFrank/envs/e2v
MODELS=/nas/student/DavidFrank/models
direnv allow . && eval "$(direnv export bash)" || exit 1
export UV_CACHE_DIR=/nas/student/DavidFrank/envs/uv_cache
set -ex  # stop at the first error, every step is visible in the log

[ -e $E2V/bin/python ] || uv venv $E2V --python $VIRTUAL_ENV/bin/python  # same interpreter as the F5 environment
grep -H '^home' $E2V/pyvenv.cfg $VIRTUAL_ENV/pyvenv.cfg  # the two lines must match
# CPU torch: fast enough for a few hundred short clips, and no CUDA libraries that could clash with the old GPUs
uv pip install --python $E2V/bin/python torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python $E2V/bin/python funasr soundfile huggingface_hub torch==2.6.0 torchaudio==2.6.0
$E2V/bin/python -I -c "import torch, torchaudio, funasr; print('e2v', torch.__version__, torchaudio.__version__, funasr.__version__)"

$E2V/bin/python -I -c "
from huggingface_hub import snapshot_download
snapshot_download('emotion2vec/emotion2vec_plus_large', local_dir='$MODELS/emotion2vec_plus_large',
                  allow_patterns=['config.yaml', 'configuration.json', 'model.pt', 'tokens.txt'])
snapshot_download('microsoft/wavlm-large', local_dir='$MODELS/wavlm-large',
                  allow_patterns=['config.json', 'preprocessor_config.json', 'pytorch_model.bin'])
"
ls -la $MODELS/emotion2vec_plus_large $MODELS/wavlm-large
[ -s $MODELS/emotion2vec_plus_large/model.pt ] && [ -s $MODELS/wavlm-large/pytorch_model.bin ]

# both judges load with their real weights, emotion2vec+ also scores a clip of noise (9 labels and scores expected)
$E2V/bin/python -I -c "
import numpy as np
from funasr import AutoModel
model = AutoModel(model='$MODELS/emotion2vec_plus_large', disable_update=True, device='cpu')
noise = np.random.default_rng(0).normal(0, 0.05, 32000).astype('float32')
print(model.generate(noise, granularity='utterance', extract_embedding=False)[0])
"
python -c "
from transformers import AutoFeatureExtractor, WavLMModel
AutoFeatureExtractor.from_pretrained('$MODELS/wavlm-large')
print('wavlm parameters', sum(p.numel() for p in WavLMModel.from_pretrained('$MODELS/wavlm-large').parameters()))
"
echo "setup done"
