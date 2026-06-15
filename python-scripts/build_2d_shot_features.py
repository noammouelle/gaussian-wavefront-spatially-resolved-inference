#!/usr/bin/env python
"""Stream HDF5 runs into compact, reusable 2D shot-level artifacts."""

import argparse
import logging
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "helpers"))

from shot_feature_pipeline import ExtractionConfig, build_feature_artifacts, discover_run_paths  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_root", type=Path, help="Dataset directory containing run_### directories")
    parser.add_argument("--output", type=Path, required=True, help="Artifact output directory")
    parser.add_argument("--port", default="Z0", choices=("Z0", "Z100"))
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--xy-min", type=float, default=-5e-3)
    parser.add_argument("--xy-max", type=float, default=5e-3)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent run readers. Start with 1-2 because HDF5 decompression and disk I/O dominate.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    paths = discover_run_paths(args.data_root, port=args.port, max_runs=args.max_runs)
    config = ExtractionConfig(args.bins, args.xy_min, args.xy_max, args.workers)
    logging.info("Building artifacts from %d runs with %d worker(s)", len(paths), args.workers)
    manifest = build_feature_artifacts(paths, args.output, config, overwrite=args.overwrite)
    logging.info(
        "Done: %d runs, %d shots, %.1f min",
        manifest["n_runs"],
        manifest["n_shots"],
        manifest["elapsed_seconds"] / 60,
    )


if __name__ == "__main__":
    main()
