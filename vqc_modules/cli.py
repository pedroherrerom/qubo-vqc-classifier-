"""cli.py — Argument parsing with JSON config overlay.

Priority (highest wins): CLI flags > JSON config > built-in defaults.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
#  Built-in defaults                                                           #
# --------------------------------------------------------------------------- #
_DEFAULTS: dict[str, Any] = {
    # Data
    "train_path": None,
    "test_path": None,
    "csv_path": None,
    "target": "Label",
    "outdir": "results",
    "id_cols": [],
    "test_size": 0.2,
    "stages": ["annealing", "quantum"],
    # Experiment
    "seed": 42,
    "num_runs": 5,
    "k": 6,
    # QUBO / SA
    "sa_num_reads": 500,
    "sa_bins": 10,
    # Optimizer
    "optimizer": "PSO",
    "opt_maxiter": 70,
    "opt_population_size": 128,
    "checkpoint_every": 10,
    # Circuit
    "fm_type": "ZZFeatureMap",
    "fm_reps": 1,
    "ansatz_type": "EfficientSU2",
    "ansatz_reps": 1,
    "ansatz_entanglement": "circular",
    "ansatz_rotation_blocks": ["ry", "rz"],
    # Backend
    "vqc_num_shots": 1024,
    "vqc_readout": "single_qubit",
    "vqc_train_infrastructure": "local",
    "vqc_test_infrastructure": "local",
    "vqc_n_workers": 1,
    "vqc_cores_per_worker": 1,
    # Array mode (set by orchestration scripts)
    "task_id": None,
    "runs_per_task": 1,
    "exp_dir": None,
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="QUBO-VQC binary molecular classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Meta
    p.add_argument("--config", default=None, help="Path to JSON config file")
    p.add_argument("--mode", choices=["run", "task", "aggregate"], default="run")

    # Data
    p.add_argument("--train-path", dest="train_path")
    p.add_argument("--test-path", dest="test_path")
    p.add_argument("--csv-path", dest="csv_path", help="Legacy single-CSV mode")
    p.add_argument("--target", dest="target")
    p.add_argument("--outdir", dest="outdir")
    p.add_argument(
        "--id-cols",
        dest="id_cols",
        nargs="*",
        help="Column names to drop (IDs, irrelevant)",
    )
    p.add_argument("--test-size", dest="test_size", type=float)
    p.add_argument(
        "--stages",
        nargs="*",
        choices=["annealing", "quantum"],
        help="Pipeline stages to run",
    )

    # Experiment
    p.add_argument("--seed", type=int)
    p.add_argument("--num-runs", dest="num_runs", type=int)
    p.add_argument("--k", type=int, help="Number of features to select / qubits")

    # QUBO / SA
    p.add_argument("--sa-num-reads", dest="sa_num_reads", type=int)
    p.add_argument("--sa-bins", dest="sa_bins", type=int)

    # Optimizer
    p.add_argument("--optimizer", choices=["PSO"])
    p.add_argument("--opt-maxiter", dest="opt_maxiter", type=int)
    p.add_argument("--opt-population-size", dest="opt_population_size", type=int)
    p.add_argument("--checkpoint-every", dest="checkpoint_every", type=int)

    # Circuit
    p.add_argument(
        "--fm-type",
        dest="fm_type",
        choices=["ZZFeatureMap", "ZFeatureMap", "PauliFeatureMap"],
    )
    p.add_argument("--fm-reps", dest="fm_reps", type=int)
    p.add_argument(
        "--ansatz-type",
        dest="ansatz_type",
        choices=["RealAmplitudes", "EfficientSU2", "TwoLocal"],
    )
    p.add_argument("--ansatz-reps", dest="ansatz_reps", type=int)
    p.add_argument(
        "--ansatz-entanglement",
        dest="ansatz_entanglement",
        choices=["full", "linear", "circular", "sca"],
    )
    p.add_argument(
        "--ansatz-rotation-blocks",
        dest="ansatz_rotation_blocks",
        nargs="*",
    )

    # Backend
    p.add_argument("--vqc-num-shots", dest="vqc_num_shots", type=int)
    p.add_argument(
        "--vqc-readout",
        dest="vqc_readout",
        choices=["single_qubit", "parity"],
    )
    p.add_argument(
        "--vqc-train-infrastructure",
        dest="vqc_train_infrastructure",
        choices=["local", "cunqa"],
    )
    p.add_argument(
        "--vqc-test-infrastructure",
        dest="vqc_test_infrastructure",
        choices=["local", "cunqa"],
    )
    p.add_argument("--vqc-n-workers", dest="vqc_n_workers", type=int)
    p.add_argument("--vqc-cores-per-worker", dest="vqc_cores_per_worker", type=int)

    # Array mode
    p.add_argument("--task-id", dest="task_id", type=int)
    p.add_argument("--runs-per-task", dest="runs_per_task", type=int)
    p.add_argument("--exp-dir", dest="exp_dir")

    return p


def build_config(argv: list[str] | None = None) -> dict[str, Any]:
    """
    Parse CLI arguments, overlay JSON config, and return merged config dict.

    Priority (highest first): CLI > JSON > built-in defaults.
    """
    parser = _build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    cli_dict = {k: v for k, v in vars(args).items() if v is not None}

    # Start from defaults
    cfg: dict[str, Any] = dict(_DEFAULTS)

    # Overlay JSON config
    config_path = cli_dict.pop("config", None)
    if config_path is not None:
        with open(config_path) as fh:
            json_cfg = json.load(fh)
        cfg.update(json_cfg)

    # Overlay CLI (non-None values only)
    cfg.update(cli_dict)

    # Normalise id_cols to list
    if isinstance(cfg.get("id_cols"), str):
        cfg["id_cols"] = [cfg["id_cols"]]

    return cfg


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load a JSON config file and fill missing keys from defaults."""
    cfg = dict(_DEFAULTS)
    with open(path) as fh:
        cfg.update(json.load(fh))
    return cfg
