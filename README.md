# Aligning Emotional TTS with Human Preferences through Rubric-Guided Reward Optimization

Code for my bachelor's thesis at the Chair of Health Informatics, Technical University of Munich.

## Layout

- `F5-TTS/` modified copy of [F5-TTS](https://github.com/SWivid/F5-TTS)
- `scripts/` data preparation and checkpoint checks
- `experiments/` local experiments and tests
- `cluster/` SLURM job scripts

## F5-TTS

`F5-TTS/` is based on [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) at commit `d852486` (v1.1.22), released under the MIT License (see `F5-TTS/LICENSE`). The BigVGAN submodule is not included.

Changes: an utterance-level emotion embedding (1024-d) is projected to the text dimension and added to the text embedding before the ConvNeXt blocks, following [emotional-vits](https://github.com/innnky/emotional-vits). The projection can be trained with its own learning rate (`--emo_lr_scale`). At inference, an optional guidance term (`emo_cfg_strength`) amplifies the effect of the emotion embedding. Modified files:

- `src/f5_tts/model/backbones/dit.py`
- `src/f5_tts/model/cfm.py`
- `src/f5_tts/model/dataset.py`
- `src/f5_tts/model/trainer.py`
- `src/f5_tts/infer/utils_infer.py`
- `src/f5_tts/train/finetune_cli.py`

## Not included

Model checkpoints and datasets are not part of this repository. The pretrained F5-TTS models are licensed under CC-BY-NC; ESD is available for research use from its authors.
