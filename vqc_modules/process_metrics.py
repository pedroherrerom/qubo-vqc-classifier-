"""process_metrics.py — Post-processing: aggregation and plot dispatch.

Collects per-run outputs, merges them, computes aggregate statistics, and
calls the visualisation functions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

logger = logging.getLogger("vqc")


# -------------------------------------------------------------------------
# Aggregation helpers
# -------------------------------------------------------------------------

def aggregate_predictions(
    all_predictions: list[dict[str, Any]],
    out_dir: Path,
) -> pd.DataFrame:
    """
    Concatenate per-run prediction dicts into a single DataFrame and save CSV.

    Each dict must have keys: ``run``, ``y_true``, ``y_pred``, ``y_prob``.
    """
    rows = []
    for pred in all_predictions:
        run_id = pred["run"]
        for yt, yp, ypr in zip(pred["y_true"], pred["y_pred"], pred["y_prob"]):
            rows.append({"run": run_id, "y_true": int(yt), "y_pred": int(yp), "y_prob": float(ypr)})

    df = pd.DataFrame(rows)
    csv_path = out_dir / "quantum_aggregated_predictions.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved aggregated predictions → %s", csv_path)
    return df


def aggregate_raw_metrics(
    per_run_metrics: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    """Save per-run metrics to JSON."""
    path = out_dir / "quantum_raw_metrics.json"
    with open(path, "w") as fh:
        json.dump(per_run_metrics, fh, indent=2, default=str)
    logger.info("Saved raw metrics → %s", path)
    return {"path": str(path), "n_runs": len(per_run_metrics)}


def aggregate_model_comparison(
    all_baseline_results: list[list[dict[str, Any]]],
    vqc_metrics: list[dict[str, Any]],
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate baseline + VQC results into mean/std and per-run DataFrames.

    Returns
    -------
    comparison_df : mean ± std per model
    runs_df       : per-run values
    """
    metric_cols = ["accuracy", "f1", "roc_auc"]
    rows_runs = []

    # VQC rows
    for run_id, m in enumerate(vqc_metrics):
        rows_runs.append(
            {
                "model": "VQC",
                "run": run_id,
                "accuracy": m.get("accuracy", float("nan")),
                "f1": m.get("f1", float("nan")),
                "roc_auc": m.get("roc_auc", float("nan")),
            }
        )

    # Baseline rows
    for run_id, baselines in enumerate(all_baseline_results):
        for b in baselines:
            rows_runs.append(
                {
                    "model": b["model"],
                    "run": run_id,
                    "accuracy": b["accuracy"],
                    "f1": b["f1"],
                    "roc_auc": b["roc_auc"],
                }
            )

    runs_df = pd.DataFrame(rows_runs)

    # Aggregation
    agg_rows = []
    for model, grp in runs_df.groupby("model", sort=False):
        row: dict[str, Any] = {"model": model}
        for col in metric_cols:
            row[f"{col}_mean"] = grp[col].mean()
            row[f"{col}_std"] = grp[col].std(ddof=1) if len(grp) > 1 else 0.0
        agg_rows.append(row)

    comparison_df = pd.DataFrame(agg_rows)

    # Save
    comparison_df.to_csv(out_dir / "model_comparison.csv", index=False)
    runs_df.to_csv(out_dir / "model_comparison_runs.csv", index=False)
    logger.info("Saved model comparison CSVs → %s", out_dir)

    return comparison_df, runs_df


# -------------------------------------------------------------------------
# Plot dispatch
# -------------------------------------------------------------------------

def generate_all_plots(
    out_dir: Path,
    history_df: pd.DataFrame | None = None,
    all_predictions: list[dict[str, Any]] | None = None,
    comparison_df: pd.DataFrame | None = None,
    runs_df: pd.DataFrame | None = None,
) -> None:
    """
    Generate all standard plots from in-memory data or saved CSVs.

    Can be called after training (in-memory) or from ``main_plotting.py``
    (loads from disk).
    """
    from .visualizations import (  # noqa: PLC0415
        plot_pso_trajectory,
        plot_final_loss_distribution,
        plot_confusion_matrix,
        plot_roc_curves,
        plot_model_comparison,
    )

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Load from disk if not provided
    if history_df is None:
        csv = out_dir / "quantum_train_historical.csv"
        if csv.exists():
            history_df = pd.read_csv(csv)

    if comparison_df is None:
        csv = out_dir / "model_comparison.csv"
        if csv.exists():
            comparison_df = pd.read_csv(csv)

    if runs_df is None:
        csv = out_dir / "model_comparison_runs.csv"
        if csv.exists():
            runs_df = pd.read_csv(csv)

    if all_predictions is None:
        csv = out_dir / "quantum_aggregated_predictions.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            # Reconstruct list of per-run dicts
            all_predictions = [
                {
                    "run": run_id,
                    "y_true": grp["y_true"].values,
                    "y_pred": grp["y_pred"].values,
                    "y_prob": grp["y_prob"].values,
                }
                for run_id, grp in df.groupby("run")
            ]

    # PSO trajectory
    if history_df is not None and not history_df.empty:
        plot_pso_trajectory(history_df, plots_dir / "quantum_pso_trajectory.png")

        # Final loss per run
        final_losses = (
            history_df.groupby("run")["best_fitness"].last().tolist()
        )
        if final_losses:
            plot_final_loss_distribution(
                final_losses, plots_dir / "quantum_final_loss_distribution.png"
            )

    # Confusion matrix & ROC
    if all_predictions:
        cms, roc_data = [], []
        for pred in all_predictions:
            yt, yp, ypr = pred["y_true"], pred["y_pred"], pred["y_prob"]
            cms.append(confusion_matrix(yt, yp, labels=[0, 1]))
            fpr, tpr, _ = roc_curve(yt, ypr)
            from sklearn.metrics import auc as sk_auc  # noqa: PLC0415

            roc_data.append({"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": sk_auc(fpr, tpr)})

        cm_arr = np.stack(cms)
        plot_confusion_matrix(
            cm_arr.mean(axis=0),
            cm_arr.std(axis=0),
            plots_dir / "quantum_confusion_matrix.png",
        )
        plot_roc_curves(roc_data, plots_dir / "quantum_roc_curves.png")

    # Model comparison
    if comparison_df is not None and runs_df is not None:
        plot_model_comparison(
            comparison_df,
            runs_df,
            metric="roc_auc",
            out_path=plots_dir / "model_comparison_bars.png",
        )
