#!/usr/bin/env bash
#SBATCH --job-name=pref_setup
#SBATCH --time=06:00:00
#SBATCH --partition=students
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --chdir=/data/chi-gpu1/ge96xah/f5tts
#SBATCH --output=/nas/student/DavidFrank/f5tts/logs/%x_%A.out

# One-time setup of the preference stage (experiments/preference/) for the RTX PRO 4000 Blackwell cards of the
# partition (gres gpu:rtx4000, compute capability 12.0). The F5 environment (torch 2.6 for CUDA 12.4) cannot drive
# these cards, and AudioJev needs BF16, which the TITAN X lack. So two environments on the NAS, torch 2.8 for CUDA 12.8:
#   f5bw   F5-TTS of this repository (editable): generation, training, the second judge, the word errors
#   jev    AudioJev's inference package (pinned commit), transformers 4.57.6, peft, openSMILE: the reward's classifier
#          and the pitch
# On the NAS as well, since /data is full (HF_HOME moves there for these environments): AudioJev at the revision every
# local score used, whisper-base, the F5-TTS base model and the vocoder; the start weights (the online weights of the
# continuation run's model_36666.pt), checked tensor by tensor against the copy of the local runs; the pitch frame of
# the real recordings. Before: copy the adapter from the PC (scp, see the notes of this run) to $M/adapters/lora4.
# Finished steps are skipped; CPU only.
# usage: sbatch thesis/cluster/pref_setup.sh
[ $# -eq 0 ] || { echo "pref_setup.sh takes no arguments"; exit 1; }
N=/nas/student/DavidFrank
ENVS=$N/envs
M=$N/models
PREF=$N/pref
CODE=/data/chi-gpu1/ge96xah/f5tts/thesis
ESD=/data/chi-gpu1/ge96xah/data/ESD
direnv allow . && eval "$(direnv export bash)" || exit 1  # uv and the interpreter of the F5 environment
PYV=$VIRTUAL_ENV/bin/python
unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE  # a cache folder of the login shell (sbatch passes its environment on) must not win over the NAS
export UV_CACHE_DIR=$ENVS/uv_cache HF_HOME=$N/hf_cache HF_HUB_CACHE=$N/hf_cache/hub HF_XET_CACHE=$N/hf_cache/xet
export PYTHONPATH=$CODE/F5-TTS/src  # the F5 code of this repository, and nothing of the F5 environment of the login shell
set -ex  # stop at the first error, every step is visible in the log
mkdir -p $PREF $HF_HOME
git -C $CODE log -1 --oneline

# f5bw: torch for CUDA 12.8 first, then F5-TTS of this repository with its dependencies (torch is kept)
[ -e $ENVS/f5bw/bin/python ] || uv venv $ENVS/f5bw --python $PYV
uv pip install --python $ENVS/f5bw/bin/python torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $ENVS/f5bw/bin/python -e $CODE/F5-TTS
# the versions of the local environment in which every part of the stage was checked (2026-11-23)
uv pip install --python $ENVS/f5bw/bin/python transformers==5.19.0 x-transformers==2.31.7 vocos==0.1.0 accelerate==1.15.0 torchdiffeq==0.2.5 \
  ema-pytorch==0.8.3 hydra-core==1.3.7 omegaconf==2.3.1 cached_path==1.8.10 huggingface_hub==1.33.0 safetensors==0.8.0 einops==0.8.2 \
  numpy==2.4.6 scipy==1.17.1 pandas==3.0.6 soundfile==0.14.0 librosa==0.11.0
$ENVS/f5bw/bin/python -c "import torch, torchaudio, transformers, f5_tts; print('f5bw', torch.__version__, torch.version.cuda, torchaudio.__version__, transformers.__version__, f5_tts.__file__)"

# jev: the versions of the local environment that scored every clip so far
[ -e $ENVS/jev/bin/python ] || uv venv $ENVS/jev --python $PYV
uv pip install --python $ENVS/jev/bin/python torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
# the inference package as the archive of the pinned commit, not by git: git refuses the NAS cache folder, whose owner
# the NAS reports as another user ("dubious ownership"), and the archive needs no git and no change of its config
uv pip install --python $ENVS/jev/bin/python transformers==4.57.6 peft==0.17.1 opensmile==2.6.0 soundfile scipy pandas \
  "audiojev-inference @ https://github.com/SihanLv/AudioJev-Inference/archive/8eeeab5e77d2c410fd6058e7bdf2b4125f11903e.tar.gz"
$ENVS/jev/bin/python -I -c "import torch, transformers, peft, opensmile, audiojev_inference; print('jev', torch.__version__, transformers.__version__, peft.__version__)"

# models: AudioJev at the local revision, and what F5 and the word count load from Hugging Face
[ -s $M/AudioJev/model.safetensors.index.json ] || $ENVS/jev/bin/python -I -c "from huggingface_hub import snapshot_download; snapshot_download('shlv/AudioJev', revision='d23dddb9669021bb2ef8cae5c07fe24b3e35702d', local_dir='$M/AudioJev')"
$ENVS/f5bw/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/whisper-base'); snapshot_download('charactr/vocos-mel-24khz'); from cached_path import cached_path; print(cached_path('hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors'))"
du -sh $M/AudioJev $HF_HOME

# the adapter of the reward's classifier, copied from the PC: the file every local score used
echo "ae9e3484c22b929fdb4a78d5d026c35d03fc4169fe907ae6af342ae9beb30e02  $M/adapters/lora4/adapter_model.safetensors" | sha256sum -c

# the start weights: the online weights of model_36666.pt, the same tensors as the local runs' copy (checksum of the tensors)
[ -s $PREF/start_36666_online.pt ] || $ENVS/f5bw/bin/python $CODE/experiments/preference/weights_check.py extract $N/f5tts/ckpts/ESD_en_xprompt1_cont/model_36666.pt $PREF/start_36666_online.pt
$ENVS/f5bw/bin/python $CODE/experiments/preference/weights_check.py checksum $PREF/start_36666_online.pt | tee $PREF/start_checksum.txt
grep -q "^606d2e3f2cabc19a139ce45dfbe7f799471c031ce4de1fe1284bf54f912586ce " $PREF/start_checksum.txt || { echo "the start weights are not the tensors of the local runs"; exit 1; }

# the pitch frame of the real recordings (score_pitch.py frame), for the guard of the pairs and the evaluation
[ -s $PREF/frame.csv ] || $ENVS/jev/bin/python -I $CODE/experiments/preference/score_pitch.py frame $PREF/frame.csv --esd $ESD
echo "setup done"
