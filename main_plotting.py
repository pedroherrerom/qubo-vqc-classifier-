#!/usr/bin/env python3
"""main_plotting.py — Standalone post-processing and plot regeneration.

Regenerates all plots from existing CSV outputs without re-running training.

Usage
-----
    python main_plotting.py --dir results/<experiment_id>/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate VQC experiment plots from existing CSV outputs."
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Path to the experiment output directory.",
    )
    parser.add_argument(
        "--metric",
        default="roc_auc",
        choices=["roc_auc", "f1", "accuracy"],
        help="Metric to display in model comparison bar chart.",
    )
    args = parser.parse_args()

    exp_dir = Path(args.dir)
    if not exp_dir.is_dir():
        print(f"ERROR: {exp_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from vqc_modules.process_metrics import generate_all_plots

    generate_all_plots(exp_dir)
    print(f"Plots saved to {exp_dir / 'plots'}")


if __name__ == "__main__":
    main()
