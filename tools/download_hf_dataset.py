#!/usr/bin/env python3
"""Download a Hugging Face dataset snapshot for local AIC experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence


DEFAULT_REPO_ID = "bha-51/aic-dagger-data"
DEFAULT_LOCAL_DIR = "/media/mbed/T7/datasets/aic/aic-dagger-data"
DEFAULT_HF_HOME = "/media/mbed/T7/hf_cache"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset snapshot into a local directory."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face repository id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--local-dir",
        default=DEFAULT_LOCAL_DIR,
        help=f"Directory where the dataset files should be materialized. Default: {DEFAULT_LOCAL_DIR}",
    )
    parser.add_argument(
        "--hf-home",
        default=DEFAULT_HF_HOME,
        help=f"Hugging Face cache root to use via HF_HOME. Default: {DEFAULT_HF_HOME}",
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
        choices=("dataset", "model", "space", "kernel"),
        help="Hugging Face repository type. Default: dataset",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch, tag, or commit hash to download.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help="Only download files matching this pattern. Can be provided multiple times.",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=None,
        help="Skip files matching this pattern. Can be provided multiple times.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download files even if they already exist locally.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Avoid network access and only resolve files that already exist locally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ask huggingface_hub what would be downloaded without writing files.",
    )
    parser.add_argument(
        "--enable-xet",
        action="store_true",
        help="Do not set HF_HUB_DISABLE_XET=1 before downloading.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum concurrent file downloads. Default: 8",
    )
    return parser.parse_args(argv)


def _format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def _print_dry_run_summary(files: object) -> None:
    if not isinstance(files, list):
        print("Dry-run result:", files)
        return

    will_download = [file for file in files if getattr(file, "will_download", False)]
    cached = [file for file in files if getattr(file, "is_cached", False)]
    total_bytes = sum(getattr(file, "file_size", 0) or 0 for file in will_download)

    print(f"Dry run files: {len(files)}")
    print(f"Would download: {len(will_download)} files ({_format_size(total_bytes)})")
    print(f"Already cached: {len(cached)} files")

    for file in will_download[:20]:
        filename = getattr(file, "filename", "<unknown>")
        file_size = getattr(file, "file_size", 0) or 0
        print(f"  download {filename} ({_format_size(file_size)})")
    if len(will_download) > 20:
        print(f"  ... {len(will_download) - 20} more files")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    hf_home = Path(args.hf_home).expanduser()
    local_dir = Path(args.local_dir).expanduser()
    if not args.dry_run:
        hf_home.mkdir(parents=True, exist_ok=True)
        local_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    if not args.enable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    from huggingface_hub import snapshot_download

    repo_type = None if args.repo_type == "model" else args.repo_type

    print("Hugging Face snapshot download")
    print(f"  repo_id: {args.repo_id}")
    print(f"  repo_type: {args.repo_type}")
    print(f"  revision: {args.revision or '<default>'}")
    print(f"  local_dir: {local_dir}")
    print(f"  HF_HOME: {hf_home}")
    print(f"  HF_HUB_DISABLE_XET: {os.environ.get('HF_HUB_DISABLE_XET', '<unset>')}")

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type=repo_type,
        revision=args.revision,
        local_dir=local_dir,
        allow_patterns=args.allow_pattern,
        ignore_patterns=args.ignore_pattern,
        force_download=args.force_download,
        local_files_only=args.local_files_only,
        dry_run=args.dry_run,
        max_workers=args.max_workers,
    )

    if args.dry_run:
        _print_dry_run_summary(path)
    else:
        print("Downloaded to:", path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
