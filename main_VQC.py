#!/usr/bin/env python3
"""main_VQC.py — Entry point for the QUBO-VQC classifier pipeline.

Usage
-----
Single-node (full pipeline):
    python main_VQC.py --config configVQC.json

SLURM array task (called by vqc_array_job.sh):
    python main_VQC.py --config configVQC.json --mode task --task-id $SLURM_ARRAY_TASK_ID

Post-array aggregation (called by aggregate_job.sh):
    python main_VQC.py --config configVQC.json --mode aggregate --exp-dir <path>
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from vqc_modules.cli import build_config
    from vqc_modules.experiment import setup_experiment, get_logger

    config = build_config()
    mode = config.pop("mode", "run")

    # In aggregate/task modes the experiment directory already exists
    exp_dir_override = config.pop("exp_dir", None)

    if exp_dir_override is not None:
        exp_dir = Path(exp_dir_override)
        exp_dir.mkdir(parents=True, exist_ok=True)
        from vqc_modules.experiment import _configure_logger

        logger = _configure_logger(exp_dir)
    else:
        exp_dir, logger = setup_experiment(config)

    logger.info("Mode: %s", mode)

    if mode == "run":
        from vqc_modules.pipeline import run_pipeline

        run_pipeline(config, exp_dir)

    elif mode == "task":
        from vqc_modules.pipeline import run_task

        run_task(config, exp_dir)

    elif mode == "aggregate":
        from vqc_modules.pipeline import run_aggregate

        run_aggregate(config, exp_dir)

    else:
        logger.error("Unknown mode: %s", mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
