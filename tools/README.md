# Tools

Small repo-owned command-line helpers for repeatable local workflows.

Run tools through Pixi from the repository root so they use the locked AIC
environment:

```bash
pixi run <task-name> [tool arguments]
```

## Download Hugging Face Dataset

Download the default AIC dataset to the external T7 paths:

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

## Test LeRobot Dataset Loading

Load the default dataset and print the dataset object, length, and sample keys:

```bash
pixi run test-lerobot-dataset-load
```

Override the Hugging Face repo, local root, or cache root:

```bash
pixi run test-lerobot-dataset-load \
  --repo-id bha-51/aic-dagger-data \
  --root /media/mbed/T7/datasets/aic/aic-dagger-data \
  --hf-home /media/mbed/T7/hf_cache
```

Useful flags:

- `--sample-index <n>`: print keys for a specific dataset item.
- `--episode <id>`: restrict loading to one episode; repeat for multiple episodes.
- `--revision <ref>`: load a specific branch, tag, or commit.
- `--no-download-videos`: skip video downloads during load.
- `--force-cache-sync`: refresh LeRobot cache metadata/files.

## Train Custom ACT

Run a short ACT training smoke job on the default local dataset:

```bash
pixi run TrainCustomACT
```

Defaults:

- dataset repo: `bha-51/aic-dagger-data`
- dataset root: `/media/mbed/T7/datasets/aic/aic-dagger-data`
- output dir: `/media/mbed/T7/datasets/aic/aic-dagger-data-outputs/train/act_aic_dagger_data`
- log file: `/media/mbed/T7/datasets/aic/aic-dagger-data-outputs/train/act_aic_dagger_data.log`
- training: `--policy.type=act`, `--batch_size=2`, `--steps=80000`, `--save_freq=5000`, `--log_freq=100`, `--policy.chunk_size=10`, `--policy.n_action_steps=10`
- disabled by default: Hub push and WandB

Override common training inputs:

```bash
pixi run TrainCustomACT \
  --steps 1000 \
  --batch-size 8 \
  --output-dir outputs/train/act_aic_v3
```

Check the resolved training command without starting a run:

```bash
pixi run TrainCustomACT --print-command
```

Any unknown argument is forwarded to `lerobot-train`, for example:

```bash
pixi run TrainCustomACT --policy.optimizer_lr=1e-4
```

If the output directory already contains a previous run, choose a new
`--output-dir` or pass `--resume=true` through to `lerobot-train`.

## Train ACT on Multiple Local Datasets

Run the local LeRobot v0.5.1 training script with `LEROBOT_MULTI_ROOTS` set to
the configured AIC dataset folders. This task uses the same
`tools/train_custom_act.py` wrapper defaults as `TrainCustomACT`, but swaps the
underlying training command to `tools/lerobot_train_v051.py`:

```bash
pixi run TrainCustomACTMulti
```

For a workstation-safe background run, use `tools/start_training_helper.sh`.
It starts `TrainCustomACTMulti` inside a systemd scope with `MemoryMax=20G` and
`MemorySwapMax=8G`, and passes defensive training settings:

- `--batch-size 2`
- `--num-workers 1`
- `--save-freq 10000`

The task reads from these local dataset roots without physically merging them:

- `/media/mbed/T7/datasets/aic/aic-dagger-data/dagger_sc_138_cheatcode`
- `/media/mbed/T7/datasets/aic/aic-dagger-data/dagger_sc_372_cheatcode`
- `/media/mbed/T7/datasets/aic/aic-dagger-data/dagger_sc_493_cheatcode`
- `/media/mbed/T7/datasets/aic/aic-dagger-data/dagger_sc_85_cheatcode`
- `/media/mbed/T7/datasets/aic/aic-dagger-data/dagger_sfp_483_cheatcode`
- `/media/mbed/T7/datasets/aic/aic-dagger-data/dagger_sfp_753_cheatcode`
- `/media/mbed/T7/datasets/aic/aic-dagger-data/dagger_sfp_300_finealign_v3`
- `/media/mbed/T7/datasets/aic/aic-dagger-data/dagger_sfp_iter3_finealign`
