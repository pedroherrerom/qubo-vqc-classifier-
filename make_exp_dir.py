#!/usr/bin/env python3
"""make_exp_dir.py — Login-node helper: pre-create a numbered experiment dir.

Creates the directory before submitting the SLURM job array so all tasks
write into the same root without race conditions.

Usage
-----
    EXP_DIR=$(python make_exp_dir.py --config configVQC.json)
    echo "Experiment directory: $EXP_DIR"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-create a numbered experiment directory."
    )
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    args = parser.parse_args()

    from vqc_modules.cli import load_config_file
    from vqc_modules.experiment import setup_experiment

    config = load_config_file(args.config)
    exp_dir, _ = setup_experiment(config)

    # Print the directory path for shell capture
    print(exp_dir)


if __name__ == "__main__":
    main()
