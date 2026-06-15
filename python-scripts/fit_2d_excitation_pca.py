#!/usr/bin/env python
"""Fit PCA from compact 2D shot-feature artifacts."""

import argparse
import json
import logging
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "helpers"))

from shot_feature_pipeline import fit_pca_artifact  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("feature_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--components", type=int, default=10)
    parser.add_argument("--min-atoms-per-pixel", type=int, default=20)
    parser.add_argument("--min-shot-fraction", type=float, default=0.95)
    parser.add_argument(
        "--representation",
        choices=("contrast", "fraction"),
        default="contrast",
        help="Contrast is affine-equivalent to fraction after centering and scaling.",
    )
    parser.add_argument(
        "--backend",
        choices=("cpu", "gpu"),
        default="cpu",
        help="GPU uses CuPy covariance eigendecomposition; CPU is usually already fast.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    started = time.perf_counter()
    result = fit_pca_artifact(
        args.feature_dir,
        args.output,
        n_components=args.components,
        min_atoms_per_pixel=args.min_atoms_per_pixel,
        min_shot_fraction=args.min_shot_fraction,
        backend=args.backend,
        representation=args.representation,
    )
    logging.info("PCA artifact written to %s in %.2fs", args.output, time.perf_counter() - started)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
