"""visualizations.py — All plot-generation functions.

Produces:
- PSO training trajectory (global best + swarm mean)
- Final loss distribution (histogram / KDE)
- Confusion matrix (mean ± std across runs)
- ROC curves (per-run + interpolated mean ± std band)
- Model comparison bar chart (mean ± std + per-run scatter)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("vqc")

# Matplotlib backend must be non-interactive for HPC
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

_PALETTE = sns.color_palette("husl", 8)
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})


# -------------------------------------------------------------------------
# PSO trajectory
# -------------------------------------------------------------------------

def plot_pso_trajectory(
    history_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Plot global-best fitness and swarm-mean fitness per generation.

    Parameters
    ----------
    history_df : DataFrame with columns [run, generation, best_fitness, mean_fitness]
    out_path   : output PNG path
    """
    if history_df.empty:
        logger.warning("No PSO history to plot.")
        return

    grouped = history_df.groupby("generation")
    gen = sorted(history_df["generation"].unique())
    best_mean = grouped["best_fitness"].mean().loc[gen].values
    best_std = grouped["best_fitness"].std().loc[gen].fillna(0).values
    mean_mean = grouped["mean_fitness"].mean().loc[gen].values

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gen, best_mean, label="Global best (mean)", color=_PALETTE[0])
    ax.fill_between(
        gen,
        best_mean - best_std,
        best_mean + best_std,
        alpha=0.25,
        color=_PALETTE[0],
    )
    ax.plot(gen, mean_mean, label="Swarm mean", color=_PALETTE[2], linestyle="--")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Loss")
    ax.set_title("PSO Training Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved PSO trajectory → %s", out_path)


# -------------------------------------------------------------------------
# Loss distribution
# -------------------------------------------------------------------------

def plot_final_loss_distribution(
    final_losses: list[float],
    out_path: Path,
) -> None:
    """Histogram / KDE of per-run final training losses."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(final_losses, bins=min(len(final_losses), 15), color=_PALETTE[1], alpha=0.8)
    ax.set_xlabel("Final loss")
    ax.set_ylabel("Count")
    ax.set_title("Final Loss Distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved loss distribution → %s", out_path)


# -------------------------------------------------------------------------
# Confusion matrix
# -------------------------------------------------------------------------

def plot_confusion_matrix(
    cm_mean: np.ndarray,
    cm_std: np.ndarray,
    out_path: Path,
) -> None:
    """
    Heatmap of the confusion matrix (mean ± std across runs).

    Parameters
    ----------
    cm_mean : (2, 2) mean confusion matrix
    cm_std  : (2, 2) std confusion matrix
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = np.array(
        [
            [
                f"{cm_mean[i, j]:.1f}\n±{cm_std[i, j]:.1f}"
                for j in range(2)
            ]
            for i in range(2)
        ]
    )
    sns.heatmap(
        cm_mean,
        annot=labels,
        fmt="",
        cmap="Blues",
        ax=ax,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    ax.set_title("Confusion Matrix (mean ± std)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", out_path)


# -------------------------------------------------------------------------
# ROC curves
# -------------------------------------------------------------------------

def plot_roc_curves(
    roc_data: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """
    Per-run ROC curves with interpolated mean ± std band.

    Parameters
    ----------
    roc_data : list of dicts with keys ``fpr``, ``tpr``, ``auc``
    """
    from sklearn.metrics import auc as sklearn_auc  # noqa: PLC0415

    base_fpr = np.linspace(0, 1, 101)
    tprs_interp = []

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, rd in enumerate(roc_data):
        fpr, tpr = np.array(rd["fpr"]), np.array(rd["tpr"])
        ax.plot(fpr, tpr, alpha=0.4, lw=1, color=_PALETTE[i % len(_PALETTE)])
        tprs_interp.append(np.interp(base_fpr, fpr, tpr))

    mean_tpr = np.mean(tprs_interp, axis=0)
    std_tpr = np.std(tprs_interp, axis=0)
    mean_auc = sklearn_auc(base_fpr, mean_tpr)

    ax.plot(
        base_fpr,
        mean_tpr,
        color="black",
        lw=2,
        label=f"Mean ROC (AUC = {mean_auc:.3f})",
    )
    ax.fill_between(
        base_fpr,
        mean_tpr - std_tpr,
        mean_tpr + std_tpr,
        alpha=0.2,
        color="grey",
        label="±1 std",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (VQC)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved ROC curves → %s", out_path)


# -------------------------------------------------------------------------
# Model comparison bars
# -------------------------------------------------------------------------

def plot_model_comparison(
    comparison_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    metric: str = "roc_auc",
    out_path: Path | None = None,
) -> None:
    """
    Grouped bar chart (mean ± std) with per-run scatter overlay.

    Parameters
    ----------
    comparison_df : aggregated mean/std per model (columns: model, metric_mean, metric_std)
    runs_df       : per-run values (columns: model, run, metric)
    metric        : metric to plot
    out_path      : output PNG path
    """
    models = comparison_df["model"].tolist()
    means = comparison_df[f"{metric}_mean"].values
    stds = comparison_df[f"{metric}_std"].values

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    bars = ax.bar(
        x, means, yerr=stds, capsize=5, color=_PALETTE[: len(models)], alpha=0.8
    )

    # Per-run scatter overlay
    for i, model in enumerate(models):
        run_vals = runs_df.loc[runs_df["model"] == model, metric].values
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(run_vals))
        ax.scatter(
            i + jitter,
            run_vals,
            color="black",
            s=20,
            alpha=0.6,
            zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Model Comparison — {metric.upper()}")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path)
        logger.info("Saved model comparison → %s", out_path)
    plt.close(fig)
