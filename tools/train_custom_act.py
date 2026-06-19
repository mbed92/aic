#!/usr/bin/env python3
"""Train a local ACT policy on the default AIC LeRobot dataset."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_REPO_ID = "bha-51/aic-dagger-data"
DEFAULT_DATASET_ROOT = "/home/mbed/aic-dagger-data"
DEFAULT_HF_HOME = "/home/mbed/hf_cache"
DEFAULT_RUN_NAME = "act_aic_dagger_data"
DEFAULT_OUTPUT_DIR = f"aic-dagger-data-outputs/train/{DEFAULT_RUN_NAME}"


def _parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Train an ACT policy with lerobot-train and tee output to a log file. "
            "Unknown arguments are forwarded to lerobot-train."
        )
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repository id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help=f"Local LeRobot dataset root. Default: {DEFAULT_DATASET_ROOT}",
    )
    parser.add_argument(
        "--hf-home",
        default=DEFAULT_HF_HOME,
        help=f"Hugging Face cache root to use via HF_HOME. Default: {DEFAULT_HF_HOME}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Training output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--job-name",
        default=DEFAULT_RUN_NAME,
        help=f"LeRobot job name. Default: {DEFAULT_RUN_NAME}",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Training log path. Default: <output-dir>.log",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size. Default: 4",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=80000,
        help="Number of training steps. Default: 80000",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=5000,
        help="Checkpoint save frequency. Default: 5000",
    )
    parser.add_argument(
        "--log-freq",
        type=int,
        default=100,
        help="Training log frequency. Default: 100",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Policy device passed as --policy.device. Default: cuda",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Set --policy.push_to_hub=true. Default is false.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Set --wandb.enable=true. Default is false.",
    )
    parser.add_argument(
        "--enable-xet",
        action="store_true",
        help="Do not set HF_HUB_DISABLE_XET=1 before training.",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the resolved lerobot-train command and exit without training.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Set --policy.chunk_size. Default is 10.",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=10,
        help="Set --policy.n_action_steps. Default is 10.",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Set --policy.use_amp=true. Default is false.",
    )
    parser.add_argument(
        "--train-command",
        default="lerobot-train",
        help="Training command to invoke. Default: lerobot-train",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers. Default: 4",
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        help="Set --policy.temporal_ensemble_coeff. Default is None.",
    )

    return parser.parse_known_args(argv)


def _build_command(args: argparse.Namespace, extra_args: Sequence[str]) -> list[str]:
    command = [
        *shlex.split(args.train_command),
        f"--policy.type=act",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={Path(args.dataset_root).expanduser()}",
        f"--output_dir={Path(args.output_dir).expanduser()}",
        f"--job_name={args.job_name}",
        f"--num_workers={args.num_workers}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--policy.device={args.device}",
        f"--policy.push_to_hub={str(args.push_to_hub).lower()}",
        f"--wandb.enable={str(args.wandb).lower()}",
        f"--policy.use_amp={str(args.use_amp).lower()}",
    ]

    if args.chunk_size is not None:
        command.append(f"--policy.chunk_size={args.chunk_size}")
    if args.n_action_steps is not None:
        command.append(f"--policy.n_action_steps={args.n_action_steps}")
    if args.temporal_ensemble_coeff is not None:
        command.append(f"--policy.temporal_ensemble_coeff={args.temporal_ensemble_coeff}")

    command.extend(extra_args)
    return command


def _tee_process(command: Sequence[str], env: dict[str, str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        command_text = shlex.join(command)
        print("Running:", command_text)
        print("Log file:", log_file)
        log.write(f"Running: {command_text}\n")
        log.flush()

        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def _remove_empty_output_dir(output_dir: Path) -> None:
    if not output_dir.exists() or not output_dir.is_dir():
        return
    if any(output_dir.iterdir()):
        return
    output_dir.rmdir()
    print(f"Removed empty output directory from previous failed start: {output_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    args, extra_args = _parse_args(argv)

    output_dir = Path(args.output_dir).expanduser()
    log_file = (
        Path(args.log_file).expanduser()
        if args.log_file is not None
        else output_dir.with_suffix(".log")
    )

    env = os.environ.copy()
    env["HF_HOME"] = str(Path(args.hf_home).expanduser())
    if not args.enable_xet:
        env["HF_HUB_DISABLE_XET"] = "1"

    command = _build_command(args, extra_args)
    if args.print_command:
        print(" ".join(shlex.quote(part) for part in command))
        print("Log file:", log_file)
        print("HF_HOME:", env["HF_HOME"])
        print("HF_HUB_DISABLE_XET:", env.get("HF_HUB_DISABLE_XET", "<unset>"))
        return 0

    _remove_empty_output_dir(output_dir)
    return _tee_process(command, env, log_file)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
