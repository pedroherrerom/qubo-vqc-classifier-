"""experiment.py — Experiment IDs, directory management, and logging setup."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path


def _next_experiment_number(outdir: Path) -> int:
    """Return the next sequential experiment number in *outdir*."""
    existing = [
        int(m.group(1))
        for d in outdir.iterdir()
        if d.is_dir() and (m := re.match(r"^(\d+)_", d.name))
    ] if outdir.exists() else []
    return max(existing, default=0) + 1


def make_experiment_id(config: dict) -> str:
    """Build a human-readable experiment identifier from the config."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fm = config.get("fm_type", "FM")
    ans = config.get("ansatz_type", "ANS")
    k = config.get("k", "?")
    stages = config.get("stages", [])
    stage_tag = "QFS" if "annealing" in stages else "noQFS"
    return f"{ts}_{fm}_{ans}_k{k}_{stage_tag}"


def setup_experiment(config: dict) -> tuple[Path, logging.Logger]:
    """
    Create the experiment output directory and configure logging.

    Returns
    -------
    exp_dir : Path
        Root directory for this experiment's outputs.
    logger : logging.Logger
        Logger writing to both console and ``experiment.log``.
    """
    outdir = Path(config.get("outdir", "results"))
    outdir.mkdir(parents=True, exist_ok=True)

    exp_id = make_experiment_id(config)
    n = _next_experiment_number(outdir)
    exp_dir = outdir / f"{n:04d}_{exp_id}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)

    # Persist full config snapshot
    config_path = exp_dir / "experiment_config.json"
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2, default=str)

    logger = _configure_logger(exp_dir)
    logger.info("Experiment directory: %s", exp_dir)
    logger.info("Config snapshot: %s", config_path)
    return exp_dir, logger


def setup_task_directory(exp_dir: Path, task_id: int) -> Path:
    """Create and return the per-task sub-directory inside *exp_dir*."""
    task_dir = exp_dir / "tasks" / f"task_{task_id:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "plots").mkdir(exist_ok=True)
    return task_dir


def _configure_logger(exp_dir: Path, name: str = "vqc") -> logging.Logger:
    """Return a logger that writes to stdout and ``experiment.log``."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(exp_dir / "experiment.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def get_logger(name: str = "vqc") -> logging.Logger:
    """Return (or create) the named logger without attaching extra handlers."""
    return logging.getLogger(name)
