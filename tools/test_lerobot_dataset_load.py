#!/usr/bin/env python3
"""Smoke-test loading a LeRobot dataset from a local checkout/cache."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence


DEFAULT_REPO_ID = "bha-51/aic-dagger-data"
DEFAULT_ROOT = "/media/mbed/T7/datasets/aic/aic-dagger-data"
DEFAULT_HF_HOME = "/media/mbed/T7/hf_cache"


def _parse_episode(value: str) -> int:
    try:
        episode = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid episode id: {value}") from exc
    if episode < 0:
        raise argparse.ArgumentTypeError("episode id must be non-negative")
    return episode


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a LeRobot dataset and print a small sanity summary."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repository id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help=f"Local dataset root. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--hf-home",
        default=DEFAULT_HF_HOME,
        help=f"Hugging Face cache root to use via HF_HOME. Default: {DEFAULT_HF_HOME}",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch, tag, or commit hash to load.",
    )
    parser.add_argument(
        "--episode",
        action="append",
        type=_parse_episode,
        default=None,
        help="Restrict loading to one episode id. Can be provided multiple times.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Dataset item index whose keys should be printed. Default: 0",
    )
    parser.add_argument(
        "--force-cache-sync",
        action="store_true",
        help="Ask LeRobot to sync cache metadata/files even if they exist locally.",
    )
    parser.add_argument(
        "--no-download-videos",
        action="store_true",
        help="Do not download video files while loading the dataset.",
    )
    parser.add_argument(
        "--video-backend",
        default=None,
        help="Optional video backend passed to LeRobotDataset.",
    )
    parser.add_argument(
        "--enable-xet",
        action="store_true",
        help="Do not set HF_HUB_DISABLE_XET=1 before loading.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.sample_index < 0:
        raise ValueError("--sample-index must be non-negative")

    hf_home = Path(args.hf_home).expanduser()
    root = Path(args.root).expanduser()
    os.environ["HF_HOME"] = str(hf_home)
    if not args.enable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print("LeRobot dataset load test")
    print(f"  repo_id: {args.repo_id}")
    print(f"  root: {root}")
    print(f"  HF_HOME: {hf_home}")
    print(f"  revision: {args.revision or '<default>'}")
    print(f"  episodes: {args.episode or '<all>'}")
    print(f"  sample_index: {args.sample_index}")
    print(f"  HF_HUB_DISABLE_XET: {os.environ.get('HF_HUB_DISABLE_XET', '<unset>')}")

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=root,
        episodes=args.episode,
        revision=args.revision,
        force_cache_sync=args.force_cache_sync,
        download_videos=not args.no_download_videos,
        video_backend=args.video_backend,
    )

    dataset_len = len(dataset)
    print(dataset)
    print("Length:", dataset_len)

    if dataset_len == 0:
        print("Dataset is empty; no sample keys to print.")
        return 0
    if args.sample_index >= dataset_len:
        raise IndexError(
            f"--sample-index {args.sample_index} is out of range for dataset length {dataset_len}"
        )

    sample = dataset[args.sample_index]
    print(f"Sample {args.sample_index} keys:", sorted(sample.keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
