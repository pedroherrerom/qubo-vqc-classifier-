"""metrics.py — Loss function, readout probabilities, and Youden calibration."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("vqc")


# -------------------------------------------------------------------------
# Readout
# -------------------------------------------------------------------------

def readout_probabilities(
    counts: dict[str, int],
    n_qubits: int,
    mode: str = "single_qubit",
) -> float:
    """
    Convert a Qiskit measurement ``counts`` dict to a classification
    probability.

    Parameters
    ----------
    counts   : ``{"0101": 312, "1011": 712, …}`` (big-endian bitstrings)
    n_qubits : total qubits in circuit
    mode     : ``"single_qubit"``  → P(qubit_0 = |1⟩)
               ``"parity"``       → P(parity of all qubits = 1)

    Returns
    -------
    float in [0, 1]
    """
    total = sum(counts.values())
    if total == 0:
        return 0.5  # undefined → neutral

    if mode == "single_qubit":
        # Qiskit bitstrings are little-endian: rightmost bit = qubit 0
        p1 = sum(v for k, v in counts.items() if k[-1] == "1")
        return p1 / total

    if mode == "parity":
        p_odd = sum(v for k, v in counts.items() if k.count("1") % 2 == 1)
        return p_odd / total

    raise ValueError(f"Unknown readout mode: {mode!r}")


def batch_probabilities(
    counts_list: list[dict[str, int]],
    n_qubits: int,
    mode: str = "single_qubit",
) -> np.ndarray:
    """Vectorised version of :func:`readout_probabilities`."""
    return np.array([readout_probabilities(c, n_qubits, mode) for c in counts_list])


# -------------------------------------------------------------------------
# Loss
# -------------------------------------------------------------------------

def cross_entropy_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Binary cross-entropy loss (clipped for numerical stability)."""
    eps = 1e-7
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification accuracy."""
    return float(np.mean(y_true == y_pred))


# -------------------------------------------------------------------------
# Youden's J threshold calibration
# -------------------------------------------------------------------------

def youden_threshold(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[float, float]:
    """
    Find the decision threshold that maximises Youden's J statistic
    (sensitivity + specificity - 1) on the provided set.

    Parameters
    ----------
    y_true : binary ground-truth labels
    y_prob : classifier output probabilities in [0, 1]

    Returns
    -------
    threshold : optimal threshold t*
    j_score   : Youden's J at t*

    Notes
    -----
    The threshold is calibrated on the *training set* only and then applied
    to the test set without re-fitting.
    """
    from sklearn.metrics import roc_curve  # noqa: PLC0415

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    threshold = float(thresholds[best_idx])
    j_score = float(j_scores[best_idx])
    logger.debug("Youden threshold: %.4f (J=%.4f)", threshold, j_score)
    return threshold, j_score


def apply_threshold(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    """Apply a fixed threshold to convert probabilities to binary labels."""
    return (y_prob >= threshold).astype(int)


# -------------------------------------------------------------------------
# Full metrics suite
# -------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute a comprehensive metrics dictionary for one run.

    If *threshold* is None, Youden's J is used to calibrate it from
    *y_true* / *y_prob* (training-set call).
    """
    from sklearn.metrics import (  # noqa: PLC0415
        roc_auc_score,
        f1_score,
        classification_report,
    )

    if threshold is None:
        threshold, _ = youden_threshold(y_true, y_prob)

    y_pred = apply_threshold(y_prob, threshold)
    acc = accuracy(y_true, y_pred)

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")

    try:
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
    except ValueError:
        f1 = float("nan")

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "threshold": threshold,
        "accuracy": acc,
        "roc_auc": auc,
        "f1": f1,
        "classification_report": report,
    }
