# Aligning Emotional TTS with Human Preferences through Rubric-Guided Reward Optimization

Code for my bachelor's thesis at the Chair of Health Informatics, Technical University of Munich.

## Layout

- `F5-TTS/` modified copy of [F5-TTS](https://github.com/SWivid/F5-TTS)
- `scripts/` data preparation, analyses and evaluation
- `experiments/` experiments: the emotion evaluation (`emotion_a/`) and the preference stage (`preference/`)
- `cluster/` SLURM job scripts

## F5-TTS

`F5-TTS/` is based on [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) at commit `d852486` (v1.1.22), released under the MIT License (see `F5-TTS/LICENSE`). The BigVGAN submodule is not included.

Changes: an utterance-level emotion embedding (1024-d) is projected to the text dimension and added to the text embedding before the ConvNeXt blocks, following [emotional-vits](https://github.com/innnky/emotional-vits). The projection can be trained with its own learning rate (`--emo_lr_scale`). At inference, an optional guidance term (`emo_cfg_strength`) amplifies the effect of the emotion embedding. For training with cross-emotion prompts, `scripts/make_xprompt_pairs.py` writes a `pairs.json` next to the prepared data: each target utterance is preceded by another utterance of the same speaker as prompt, only the target is masked, and the embedding comes from another utterance of the target emotion; `--emo_drop_prob` drops the embedding alone. Prompt and target are cut at their silent edges as F5-TTS does with a reference at inference. Modified files:

- `src/f5_tts/model/backbones/dit.py`
- `src/f5_tts/model/cfm.py`
- `src/f5_tts/model/dataset.py`
- `src/f5_tts/model/trainer.py`
- `src/f5_tts/infer/utils_infer.py`
- `src/f5_tts/train/finetune_cli.py`

## Preference stage

`experiments/preference/` trains the fine-tuned model on pairs of its own samples (Flow-DPO), with a reward that two independent emotion classifiers must agree on. The order of a run:

1. `generate.py` writes a pool of the model's own clips on training sentences, several seeds per item, with the emotion input of `emotion_input.py`: the emotion vector of a source recording moved along what is unique to the intended emotion, by a strength per emotion chosen by ear.
2. `scripts/score_audiojev.py` with a LoRA adapter, `scripts/score_wavlm_judge.py`, `score_words.py` and `score_pitch.py` measure every clip: the intensity and identity of the emotion by both classifiers, the word errors (whisper-base) and the mean pitch against the speaker's real recordings.
3. `make_pairs.py` picks per item the pair whose winner both classifiers hear as more intense, with an identity that is not weaker, no more word errors and a pitch not far above the speaker's real range, and writes a control of random pairs of the same items. `make_anchor_table.py` lists the start model's own neutral clips for the anchor.
4. `train_dpo.py` trains on the pairs, with the SFT term and an anchor that holds the velocity on neutral clips at the start model's. It saves weights and a resumable state, and stops when a file `STOP` appears in its run folder.
5. `generate.py`, `level_match.py` and the scorers produce a saved state's evaluation set, and `evaluate.py` reads it against the start model with the same emotion input: the effect in units of the emotion input's strength, identity, the word errors of neutral and of the emotions, and the share of clips with an overdone pitch. It writes `STOP` when a stop rule fires.

Every script describes its method and its choices in its head comment.

## Not included

Model checkpoints and datasets are not part of this repository. The pretrained F5-TTS models are licensed under CC-BY-NC; ESD is available for research use from its authors.
