"""model_comparison.py — Classical baselines (LogReg, SVM-RBF) and QSVC.

Each baseline is trained and evaluated using the same train/test split as the
VQC, producing per-run metrics that are later aggregated.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("vqc")


def _run_sklearn_classifier(
    clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str,
) -> dict[str, Any]:
    """Fit *clf* and return a metrics dict."""
    from sklearn.metrics import (  # noqa: PLC0415
        accuracy_score,
        f1_score,
        roc_auc_score,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        df = clf.decision_function(X_test)
        y_prob = (df - df.min()) / (df.max() - df.min() + 1e-12)
    else:
        y_prob = y_pred.astype(float)

    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except ValueError:
        auc = float("nan")

    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": auc,
    }


def run_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
) -> dict[str, Any]:
    """Train and evaluate Logistic Regression."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    return _run_sklearn_classifier(clf, X_train, y_train, X_test, y_test, "LogReg")


def run_svm_rbf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
) -> dict[str, Any]:
    """Train and evaluate SVM with RBF kernel."""
    from sklearn.svm import SVC  # noqa: PLC0415
    from sklearn.calibration import CalibratedClassifierCV  # noqa: PLC0415

    base_svc = SVC(kernel="rbf", random_state=seed)
    clf = CalibratedClassifierCV(base_svc, ensemble=False)
    return _run_sklearn_classifier(clf, X_train, y_train, X_test, y_test, "SVM-RBF")


def run_qsvc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Train and evaluate QSVC using a quantum kernel.

    Falls back to classical SVM-RBF if qiskit-machine-learning is unavailable.
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # noqa: PLC0415

    try:
        from qiskit_machine_learning.kernels import FidelityQuantumKernel  # noqa: PLC0415
        from qiskit_machine_learning.algorithms import QSVC as _QSVC  # noqa: PLC0415
        from .quantum_circuits import build_feature_map  # noqa: PLC0415

        n_qubits = X_train.shape[1]
        fm = build_feature_map(n_qubits, config)
        kernel = FidelityQuantumKernel(feature_map=fm)
        clf = _QSVC(quantum_kernel=kernel)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        try:
            # QSVC doesn't expose predict_proba; use decision_function proxy
            df = clf.decision_function(X_test)
            y_prob = (df - df.min()) / (df.max() - df.min() + 1e-12)
            auc = float(roc_auc_score(y_test, y_prob))
        except Exception:
            auc = float("nan")

        return {
            "model": "QSVC",
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": auc,
        }

    except ImportError:
        logger.warning(
            "qiskit-machine-learning not installed; skipping QSVC. "
            "Install with: pip install qiskit-machine-learning"
        )
        result = run_svm_rbf(X_train, y_train, X_test, y_test)
        result["model"] = "QSVC (fallback=SVM-RBF)"
        return result


def run_all_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict[str, Any],
    run_id: int = 0,
) -> list[dict[str, Any]]:
    """Run all classical and QSVC baselines for a single data split."""
    seed = config.get("seed", 42) + run_id
    results = []

    logger.info("Run %d — running classical baselines…", run_id)
    results.append(run_logistic_regression(X_train, y_train, X_test, y_test, seed))
    results.append(run_svm_rbf(X_train, y_train, X_test, y_test, seed))
    results.append(run_qsvc(X_train, y_train, X_test, y_test, config))

    for r in results:
        logger.info(
            "  %-25s acc=%.3f  f1=%.3f  auc=%.3f",
            r["model"],
            r["accuracy"],
            r["f1"],
            r["roc_auc"],
        )

    return results
