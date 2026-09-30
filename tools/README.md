# Tools

Small repo-owned command-line helpers for repeatable local workflows.

Run tools through Pixi from the repository root so they use the locked AIC
environment:

```bash
pixi run <task-name> [tool arguments]
```

## Download Hugging Face Dataset

Download the default AIC dataset to `~/datasets/aic-dagger-data`, using
`~/hf_cache` as the Hugging Face cache:

```bash
pixi run download-hf-dataset
```

Override the Hugging Face repo, output directory, or cache root:

```bash
pixi run download-hf-dataset \
  --repo-id bha-51/aic-dagger-data \
  --local-dir /media/mbed/T7/datasets/aic/aic-dagger-data \
  --hf-home /media/mbed/T7/hf_cache
```

Useful flags:

- `--dry-run`: show what would be downloaded without writing dataset files.
- `--revision <ref>`: download a specific branch, tag, or commit.
- `--allow-pattern <glob>` / `--ignore-pattern <glob>`: filter files.
- `--force-download`: refresh files even when they already exist locally.
- `--local-files-only`: avoid network access and use cached files only.

By default, the downloader sets `HF_HUB_DISABLE_XET=1`. Use `--enable-xet` to
leave Xet enabled.

## Train ACTTrainedPolicy

Run a short ACT training smoke job on the default local dataset:

```bash
pixi run TrainACTTrainedPolicy
```

Defaults:

- dataset repo: `bha-51/aic-dagger-data`
- dataset root: `~/datasets/aic-dagger-data`
- output dir: `aic-dagger-data-outputs/train/act/act_aic_dagger_data`
- log file: `aic-dagger-data-outputs/train/act/act_aic_dagger_data.log`
- training: `--policy.type=act`, `--batch_size=4`, `--steps=80000`, `--save_freq=5000`, `--log_freq=100`, `--policy.chunk_size=10`, `--policy.n_action_steps=10`
- disabled by default: Hub push and WandB

Override common training inputs:

```bash
pixi run TrainACTTrainedPolicy \
  --steps 1000 \
  --batch-size 8 \
  --output-dir outputs/train/act_aic_v3
```

Check the resolved training command without starting a run:

```bash
pixi run TrainACTTrainedPolicy --print-command
```

Any unknown argument is forwarded to `lerobot-train`, for example:

```bash
pixi run TrainACTTrainedPolicy --policy.optimizer_lr=1e-4
```

If the output directory already contains a previous run, choose a new
`--output-dir` or pass `--resume=true` through to `lerobot-train`.

## Train ACT on Multiple Local Datasets

Run the local LeRobot v0.5.1 training script with `LEROBOT_MULTI_ROOTS` set to
the configured AIC dataset folders. This task uses the same
`tools/train_custom_act.py` wrapper defaults as `TrainACTTrainedPolicy`, but
swaps the underlying training command to `tools/lerobot_train_v051.py`.
The local trainer uses FP16 mixed precision when `--policy.use_amp=true` on
CUDA, and disables mixed precision otherwise:

```bash
pixi run TrainACTTrainedPolicyMulti
```

For a workstation-safe background run, use `tools/start_act_training.sh`.
It starts ACT training inside a systemd scope with `MemoryMax=20G` and
`MemorySwapMax=8G`.

- `--batch-size 16`
- `--num-workers 4`
- `--save-freq 5000`

The task reads from these local dataset roots without physically merging them:

- `~/datasets/aic-dagger-data/dagger_sc_138_cheatcode`
- `~/datasets/aic-dagger-data/dagger_sc_372_cheatcode`
- `~/datasets/aic-dagger-data/dagger_sc_493_cheatcode`
- `~/datasets/aic-dagger-data/dagger_sc_85_cheatcode`
- `~/datasets/aic-dagger-data/dagger_sfp_753_cheatcode`
- `~/datasets/aic-dagger-data/dagger_sfp_300_finealign_v3`
- `~/datasets/aic-dagger-data/dagger_sfp_iter3_finealign`

## Train SmolVLATrainedPolicy

Run SmolVLA training on the same local multi-dataset roots used by
`TrainACTTrainedPolicyMulti`:

```bash
pixi run TrainSmolVLATrainedPolicy
```

This task uses `tools/train_smolvla.py`, `--policy.type=smolvla`, and the local
LeRobot v0.5.1 training script at `tools/lerobot_train_v051.py`.
The wrapper loads pretrained VLM weights by default. `--no-load-vlm-weights`
is reserved for explicit training-from-scratch experiments; in that case,
also configure `--policy.train_expert_only=false` and
`--policy.freeze_vision_encoder=false` so the randomly initialized VLM can train.
The background helper sets `HF_HUB_OFFLINE=1`, so the model weights,
configuration, and processor files must already be available in `~/hf_cache`.

Check the resolved training command without starting a run:

```bash
pixi run TrainSmolVLATrainedPolicy --print-command
```

Run a short smoke job with smaller settings:

```bash
pixi run TrainSmolVLATrainedPolicy \
  --steps 1000 \
  --batch-size 1
```

For a workstation-safe background run, use:

```bash
bash tools/start_smolvla_training.sh
```

The SmolVLA helper starts the task inside a systemd scope with `MemoryMax=20G`
and `MemorySwapMax=8G`, and passes conservative training settings:

- `--batch-size 1`
- `--num-workers 1`
- `--save-freq 10000`
- `--chunk-size 50`
- `--n-action-steps 50`
