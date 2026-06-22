"""feature_selection.py — QUBO-based feature selection via simulated annealing.

Implements Muecke et al. (2023):
  "Feature selection on quantum computers"
  Quantum Machine Intelligence 5(1), 11.
  https://doi.org/10.1007/s42484-023-00099-z
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("vqc")

# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------

def _mutual_information_matrix(
    X: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute feature-importance vector and pairwise-redundancy matrix using
    discretised mutual information.

    Parameters
    ----------
    X       : (n_samples, n_features)
    y       : (n_samples,) integer labels
    n_bins  : number of bins for feature discretisation

    Returns
    -------
    importance  : (n_features,) float  — MI(feature_i, y)
    redundancy  : (n_features, n_features) float  — MI(feature_i, feature_j)
    """
    from sklearn.feature_selection import mutual_info_classif  # noqa: PLC0415

    n_features = X.shape[1]

    importance = mutual_info_classif(X, y, discrete_features=False, random_state=0)

    # Pairwise redundancy: MI(X_i, X_j) via equal-width binning + histogram
    X_disc = np.apply_along_axis(
        lambda col: np.digitize(col, np.linspace(col.min(), col.max(), n_bins + 1)[1:-1]),
        0,
        X,
    )
    redundancy = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i, n_features):
            mi = _mi_discrete(X_disc[:, i], X_disc[:, j])
            redundancy[i, j] = mi
            redundancy[j, i] = mi

    return importance, redundancy


def _mi_discrete(a: np.ndarray, b: np.ndarray) -> float:
    """Mutual information between two discretised integer arrays."""
    n = len(a)
    joint = {}
    for ai, bi in zip(a, b):
        joint[(ai, bi)] = joint.get((ai, bi), 0) + 1

    p_joint = np.array(list(joint.values())) / n
    p_a = np.bincount(a, minlength=int(a.max()) + 1) / n
    p_b = np.bincount(b, minlength=int(b.max()) + 1) / n

    mi = 0.0
    for (ai, bi), cnt in joint.items():
        p = cnt / n
        if p > 0 and p_a[ai] > 0 and p_b[bi] > 0:
            mi += p * np.log(p / (p_a[ai] * p_b[bi]) + 1e-12)
    return max(mi, 0.0)


def _build_qubo(
    importance: np.ndarray,
    redundancy: np.ndarray,
    alpha: float,
) -> dict[tuple[int, int], float]:
    """Build the QUBO dictionary Q for the feature selection problem."""
    n = len(importance)
    Q: dict[tuple[int, int], float] = {}

    for i in range(n):
        # Linear term: -alpha * importance[i]
        Q[(i, i)] = Q.get((i, i), 0.0) - alpha * importance[i]

    for i in range(n):
        for j in range(i + 1, n):
            # Quadratic (redundancy) term
            Q[(i, j)] = Q.get((i, j), 0.0) + redundancy[i, j]

    return Q


def _solve_qubo(
    Q: dict[tuple[int, int], float],
    num_reads: int,
    seed: int,
) -> np.ndarray:
    """Solve the QUBO with simulated annealing; return best binary sample."""
    try:
        import neal  # noqa: PLC0415
        import dimod  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "QUBO feature selection requires 'neal' and 'dimod'. "
            "Install them via: pip install neal dimod"
        ) from exc

    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=num_reads, seed=seed)
    best = response.first.sample
    n = max(max(k) for k in Q) + 1
    return np.array([best[i] for i in range(n)], dtype=int)


def _count_selected(
    Q: dict[tuple[int, int], float],
    num_reads: int,
    seed: int,
) -> tuple[int, np.ndarray]:
    sample = _solve_qubo(Q, num_reads, seed)
    return int(sample.sum()), sample


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------

def select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    feature_names: list[str],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Select exactly *k* features using QUBO + simulated annealing.

    A binary search over the regularisation weight ``α`` ensures that exactly
    *k* features are selected.

    Parameters
    ----------
    X_train      : (n_samples, n_features)
    y_train      : (n_samples,)
    k            : target number of features
    feature_names: column names corresponding to X_train columns
    config       : pipeline config dict (reads ``sa_num_reads``, ``sa_bins``,
                   ``seed``)

    Returns
    -------
    selected_idx  : (k,) integer indices into the original feature array
    X_selected    : (n_samples, k) — X_train with only selected features
    selected_names: list of selected feature names
    """
    num_reads = config.get("sa_num_reads", 500)
    n_bins = config.get("sa_bins", 10)
    seed = config.get("seed", 42)

    n_features = X_train.shape[1]
    if k >= n_features:
        logger.warning(
            "k=%d >= n_features=%d — skipping QUBO, returning all features.", k, n_features
        )
        return (
            np.arange(n_features),
            X_train,
            feature_names,
        )

    logger.info("Computing mutual information matrices (n_features=%d)…", n_features)
    importance, redundancy = _mutual_information_matrix(X_train, y_train, n_bins)

    # Binary search over alpha
    alpha_lo, alpha_hi = 0.0, 1.0
    best_sample = np.zeros(n_features, dtype=int)
    n_selected = 0

    # Expand upper bound if necessary
    for _ in range(20):
        Q = _build_qubo(importance, redundancy, alpha_hi)
        n_selected, sample = _count_selected(Q, num_reads, seed)
        if n_selected >= k:
            break
        alpha_hi *= 2.0

    for _ in range(50):
        alpha_mid = (alpha_lo + alpha_hi) / 2.0
        Q = _build_qubo(importance, redundancy, alpha_mid)
        n_selected, sample = _count_selected(Q, num_reads, seed)
        logger.debug("α=%.6f → %d features selected", alpha_mid, n_selected)

        if n_selected == k:
            best_sample = sample
            break
        elif n_selected < k:
            alpha_lo = alpha_mid
        else:
            alpha_hi = alpha_mid
            best_sample = sample

    if n_selected != k:
        logger.warning(
            "Binary search converged to %d features (target=%d). "
            "Using best found sample.",
            int(best_sample.sum()),
            k,
        )
        # If we overshot, take top-k by importance among selected
        if int(best_sample.sum()) > k:
            selected_idx_all = np.where(best_sample == 1)[0]
            top_k = selected_idx_all[np.argsort(-importance[selected_idx_all])[:k]]
            best_sample = np.zeros(n_features, dtype=int)
            best_sample[top_k] = 1
        elif int(best_sample.sum()) < k:
            # Fill remaining slots with highest-importance unselected features
            unselected = np.where(best_sample == 0)[0]
            deficit = k - int(best_sample.sum())
            fill = unselected[np.argsort(-importance[unselected])[:deficit]]
            best_sample[fill] = 1

    selected_idx = np.where(best_sample == 1)[0]
    selected_names = [feature_names[i] for i in selected_idx]
    logger.info("Selected %d features: %s", len(selected_idx), selected_names)

    return selected_idx, X_train[:, selected_idx], selected_names
