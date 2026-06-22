"""pipeline.py — Pipeline orchestration.

Supports three execution modes:

``run``        — single-node, full pipeline (default).
``task``       — array task: runs a slice of independent runs and saves to
                 ``tasks/task_NNNN/``.
``aggregate``  — merges all task outputs into the experiment root and
                 generates final plots.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("vqc")


# -------------------------------------------------------------------------
# Single-node pipeline
# -------------------------------------------------------------------------

def run_pipeline(config: dict[str, Any], exp_dir: Path) -> None:
    """Execute the full pipeline on a single node."""
    from .data_processing import DataPreprocessor, load_datasets  # noqa: PLC0415
    from .feature_selection import select_features  # noqa: PLC0415
    from .quantum_circuits import build_feature_map, build_ansatz  # noqa: PLC0415
    from .training import train_vqc, predict_proba  # noqa: PLC0415
    from .metrics import (  # noqa: PLC0415
        youden_threshold,
        apply_threshold,
        compute_metrics,
    )
    from .model_comparison import run_all_baselines  # noqa: PLC0415
    from .process_metrics import (  # noqa: PLC0415
        aggregate_predictions,
        aggregate_raw_metrics,
        aggregate_model_comparison,
        generate_all_plots,
    )
    from .serialization import dump_json  # noqa: PLC0415

    # ---- Data ----
    train_df, test_df = load_datasets(config)
    preprocessor = DataPreprocessor(
        id_cols=config.get("id_cols", []),
        target=config["target"],
    )
    X_train, X_test, y_train, y_test, feature_names = preprocessor.fit_transform(
        train_df, test_df
    )

    # ---- Feature selection ----
    stages = config.get("stages", ["annealing", "quantum"])
    if "annealing" in stages:
        k = config.get("k", X_train.shape[1])
        selected_idx, X_train_sel, selected_names = select_features(
            X_train, y_train, k, feature_names, config
        )
        X_test_sel = X_test[:, selected_idx]
        logger.info("Selected features: %s", selected_names)
    else:
        X_train_sel = X_train
        X_test_sel = X_test
        selected_names = feature_names

    n_qubits = X_train_sel.shape[1]

    # ---- Circuit construction ----
    feature_map = build_feature_map(n_qubits, config)
    ansatz = build_ansatz(n_qubits, config)

    # ---- Multi-run VQC training ----
    num_runs = config.get("num_runs", 5)
    checkpoint_path = exp_dir / "quantum_train_historical.csv"
    all_predictions: list[dict[str, Any]] = []
    per_run_metrics: list[dict[str, Any]] = []
    all_baseline_results: list[list[dict[str, Any]]] = []

    for run_id in range(num_runs):
        logger.info("=== Run %d / %d ===", run_id + 1, num_runs)

        # Training
        params, history = train_vqc(
            feature_map,
            ansatz,
            X_train_sel,
            y_train,
            config,
            run_id=run_id,
            checkpoint_path=checkpoint_path,
        )

        # Train-set predictions for threshold calibration
        y_prob_train = predict_proba(feature_map, ansatz, X_train_sel, params, config)
        threshold, _ = youden_threshold(y_train, y_prob_train)

        # Test-set evaluation
        y_prob_test = predict_proba(feature_map, ansatz, X_test_sel, params, config)
        y_pred_test = apply_threshold(y_prob_test, threshold)
        metrics = compute_metrics(y_test, y_prob_test, threshold=threshold)
        per_run_metrics.append({"run": run_id, **metrics})

        logger.info(
            "Run %d results — acc=%.3f  f1=%.3f  auc=%.3f  thr=%.4f",
            run_id,
            metrics["accuracy"],
            metrics["f1"],
            metrics["roc_auc"],
            threshold,
        )

        all_predictions.append(
            {
                "run": run_id,
                "y_true": y_test,
                "y_pred": y_pred_test,
                "y_prob": y_prob_test,
            }
        )

        # Classical baselines (once per run with same split)
        baselines = run_all_baselines(
            X_train_sel, y_train, X_test_sel, y_test, config, run_id
        )
        all_baseline_results.append(baselines)

    # ---- Aggregation & export ----
    aggregate_predictions(all_predictions, exp_dir)
    aggregate_raw_metrics(per_run_metrics, exp_dir)
    comparison_df, runs_df = aggregate_model_comparison(
        all_baseline_results, per_run_metrics, exp_dir
    )

    history_df = pd.read_csv(checkpoint_path) if checkpoint_path.exists() else None
    generate_all_plots(
        exp_dir,
        history_df=history_df,
        all_predictions=all_predictions,
        comparison_df=comparison_df,
        runs_df=runs_df,
    )

    logger.info("Pipeline complete. Outputs in: %s", exp_dir)


# -------------------------------------------------------------------------
# Array-task mode
# -------------------------------------------------------------------------

def run_task(config: dict[str, Any], exp_dir: Path) -> None:
    """
    Execute a single SLURM array task: run a slice of independent VQC runs
    and write outputs to ``tasks/task_NNNN/``.
    """
    from .experiment import setup_task_directory  # noqa: PLC0415

    task_id = config.get("task_id", 0)
    runs_per_task = config.get("runs_per_task", 1)
    global_seed = config.get("seed", 42)

    task_dir = setup_task_directory(exp_dir, task_id)
    logger.info("Task %d → %s", task_id, task_dir)

    # Compute which run IDs this task owns
    run_start = task_id * runs_per_task
    run_ids = list(range(run_start, run_start + runs_per_task))

    # Use fixed QUBO seed (same features across all tasks)
    task_config = dict(config)
    task_config["seed"] = global_seed  # fixed for feature selection
    task_config["num_runs"] = len(run_ids)

    # Run pipeline into task_dir
    task_config["outdir"] = str(task_dir.parent.parent)  # not used directly
    run_pipeline_slice(task_config, task_dir, run_ids)


def run_pipeline_slice(
    config: dict[str, Any],
    out_dir: Path,
    run_ids: list[int],
) -> None:
    """Run the pipeline for the specified *run_ids* and save outputs to *out_dir*."""
    from .data_processing import DataPreprocessor, load_datasets  # noqa: PLC0415
    from .feature_selection import select_features  # noqa: PLC0415
    from .quantum_circuits import build_feature_map, build_ansatz  # noqa: PLC0415
    from .training import train_vqc, predict_proba  # noqa: PLC0415
    from .metrics import youden_threshold, apply_threshold, compute_metrics  # noqa: PLC0415
    from .model_comparison import run_all_baselines  # noqa: PLC0415
    from .process_metrics import aggregate_predictions, aggregate_raw_metrics  # noqa: PLC0415
    from .serialization import dump_json  # noqa: PLC0415

    train_df, test_df = load_datasets(config)
    preprocessor = DataPreprocessor(id_cols=config.get("id_cols", []), target=config["target"])
    X_train, X_test, y_train, y_test, feature_names = preprocessor.fit_transform(train_df, test_df)

    stages = config.get("stages", ["annealing", "quantum"])
    if "annealing" in stages:
        k = config.get("k", X_train.shape[1])
        selected_idx, X_train_sel, selected_names = select_features(
            X_train, y_train, k, feature_names, config
        )
        X_test_sel = X_test[:, selected_idx]
    else:
        X_train_sel, X_test_sel = X_train, X_test

    n_qubits = X_train_sel.shape[1]
    feature_map = build_feature_map(n_qubits, config)
    ansatz = build_ansatz(n_qubits, config)

    checkpoint_path = out_dir / "quantum_train_historical.csv"
    all_predictions: list[dict[str, Any]] = []
    per_run_metrics: list[dict[str, Any]] = []
    all_baselines: list[list[dict[str, Any]]] = []

    for run_id in run_ids:
        # Unique seed per run for the optimiser
        run_config = dict(config)
        run_config["seed"] = config.get("seed", 42) + run_id

        params, history = train_vqc(
            feature_map, ansatz, X_train_sel, y_train, run_config,
            run_id=run_id, checkpoint_path=checkpoint_path
        )
        y_prob_train = predict_proba(feature_map, ansatz, X_train_sel, params, run_config)
        threshold, _ = youden_threshold(y_train, y_prob_train)
        y_prob_test = predict_proba(feature_map, ansatz, X_test_sel, params, run_config)
        y_pred_test = apply_threshold(y_prob_test, threshold)
        metrics = compute_metrics(y_test, y_prob_test, threshold=threshold)
        per_run_metrics.append({"run": run_id, **metrics})
        all_predictions.append({"run": run_id, "y_true": y_test, "y_pred": y_pred_test, "y_prob": y_prob_test})
        all_baselines.append(run_all_baselines(X_train_sel, y_train, X_test_sel, y_test, run_config, run_id))

    aggregate_predictions(all_predictions, out_dir)
    aggregate_raw_metrics(per_run_metrics, out_dir)

    # Save baseline results for later aggregation
    dump_json(all_baselines, out_dir / "baseline_results.json")


# -------------------------------------------------------------------------
# Aggregation mode
# -------------------------------------------------------------------------

def run_aggregate(config: dict[str, Any], exp_dir: Path) -> None:
    """
    Merge outputs from all task sub-directories and generate final plots.
    """
    from .process_metrics import (  # noqa: PLC0415
        aggregate_predictions,
        aggregate_raw_metrics,
        aggregate_model_comparison,
        generate_all_plots,
    )
    from .serialization import load_json  # noqa: PLC0415

    tasks_dir = exp_dir / "tasks"
    if not tasks_dir.exists():
        logger.error("No tasks directory found in %s", exp_dir)
        return

    task_dirs = sorted(tasks_dir.glob("task_*"))
    logger.info("Aggregating %d task directories…", len(task_dirs))

    all_predictions: list[dict[str, Any]] = []
    per_run_metrics: list[dict[str, Any]] = []
    all_baseline_results: list[list[dict[str, Any]]] = []
    history_frames: list[pd.DataFrame] = []

    for td in task_dirs:
        # Predictions
        pred_csv = td / "quantum_aggregated_predictions.csv"
        if pred_csv.exists():
            df = pd.read_csv(pred_csv)
            for run_id, grp in df.groupby("run"):
                all_predictions.append(
                    {
                        "run": run_id,
                        "y_true": grp["y_true"].values,
                        "y_pred": grp["y_pred"].values,
                        "y_prob": grp["y_prob"].values,
                    }
                )

        # Metrics
        metrics_json = td / "quantum_raw_metrics.json"
        if metrics_json.exists():
            per_run_metrics.extend(load_json(metrics_json))

        # Baselines
        baselines_json = td / "baseline_results.json"
        if baselines_json.exists():
            all_baseline_results.extend(load_json(baselines_json))

        # History
        hist_csv = td / "quantum_train_historical.csv"
        if hist_csv.exists():
            history_frames.append(pd.read_csv(hist_csv))

    history_df = pd.concat(history_frames, ignore_index=True) if history_frames else None
    if history_df is not None:
        history_df.to_csv(exp_dir / "quantum_train_historical.csv", index=False)

    aggregate_predictions(all_predictions, exp_dir)
    aggregate_raw_metrics(per_run_metrics, exp_dir)
    comparison_df, runs_df = aggregate_model_comparison(
        all_baseline_results, per_run_metrics, exp_dir
    )
    generate_all_plots(
        exp_dir,
        history_df=history_df,
        all_predictions=all_predictions,
        comparison_df=comparison_df,
        runs_df=runs_df,
    )
    logger.info("Aggregation complete → %s", exp_dir)
