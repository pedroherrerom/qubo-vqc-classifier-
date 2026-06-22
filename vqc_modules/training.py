"""training.py — polypus.qml.train adapter with PSO optimiser.

Wraps the Rust-backed ``polypus.qml.train`` function, captures per-generation
fitness values from Rust stdout via ``_PolypusStdoutInterceptor``, and
checkpoints progress to CSV.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from .serialization import _PolypusStdoutInterceptor

logger = logging.getLogger("vqc")

# Regex to parse polypus stdout lines of the form:
#   gen=<N> best=<F> mean=<F>
_GEN_RE = re.compile(
    r"gen\s*=\s*(\d+)\s+best\s*=\s*([\d.eE+\-]+)\s+mean\s*=\s*([\d.eE+\-]+)"
)


def _parse_polypus_output(
    lines: list[str],
) -> list[dict[str, float]]:
    """Parse generation-level fitness records from polypus stdout lines."""
    records = []
    for line in lines:
        m = _GEN_RE.search(line)
        if m:
            records.append(
                {
                    "generation": int(m.group(1)),
                    "best_fitness": float(m.group(2)),
                    "mean_fitness": float(m.group(3)),
                }
            )
    return records


def _checkpoint_fitness(
    records: list[dict[str, float]],
    run_id: int,
    checkpoint_path: Path,
) -> None:
    """Append fitness records (with run_id column) to a CSV checkpoint file."""
    if not records:
        return
    write_header = not checkpoint_path.exists()
    with open(checkpoint_path, "a", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["run", "generation", "best_fitness", "mean_fitness"]
        )
        if write_header:
            writer.writeheader()
        for rec in records:
            writer.writerow({"run": run_id, **rec})


# -------------------------------------------------------------------------
# Public training function
# -------------------------------------------------------------------------

def train_vqc(
    feature_map,
    ansatz,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, Any],
    run_id: int = 0,
    checkpoint_path: Path | None = None,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """
    Train the VQC using ``polypus.qml.train`` (PSO).

    Parameters
    ----------
    feature_map      : Qiskit QuantumCircuit (feature map)
    ansatz           : Qiskit QuantumCircuit (trainable ansatz)
    X_train          : (n_samples, n_features)
    y_train          : (n_samples,) integer labels {0, 1}
    config           : pipeline config dict
    run_id           : integer index of this independent run (for checkpoints)
    checkpoint_path  : Path to append fitness CSV (``quantum_train_historical.csv``)

    Returns
    -------
    optimal_params : (n_params,) float — best PSO parameters
    history        : list of per-generation fitness dicts
    """
    n_workers = config.get("vqc_n_workers", 1)
    cores_per_worker = config.get("vqc_cores_per_worker", 1)
    num_shots = config.get("vqc_num_shots", 1024)
    maxiter = config.get("opt_maxiter", 70)
    population_size = config.get("opt_population_size", 128)
    readout = config.get("vqc_readout", "single_qubit")
    checkpoint_every = config.get("checkpoint_every", 10)
    seed = config.get("seed", 42) + run_id  # unique seed per run

    logger.info(
        "Run %d — training VQC (PSO): maxiter=%d, population=%d, shots=%d",
        run_id,
        maxiter,
        population_size,
        num_shots,
    )

    try:
        import polypus.qml as pqml  # noqa: PLC0415

        with _PolypusStdoutInterceptor() as cap:
            result = pqml.train(
                feature_map=feature_map,
                ansatz=ansatz,
                X=X_train,
                y=y_train,
                optimizer="PSO",
                max_iter=maxiter,
                population_size=population_size,
                n_workers=n_workers,
                cores_per_worker=cores_per_worker,
                num_shots=num_shots,
                readout=readout,
                seed=seed,
            )

        stdout_lines = cap.lines
        history = _parse_polypus_output(stdout_lines)

    except ImportError:
        logger.warning(
            "polypus not available — falling back to scipy PSO for local testing."
        )
        result, history = _scipy_pso_fallback(
            feature_map,
            ansatz,
            X_train,
            y_train,
            config,
            run_id,
            seed,
        )

    if checkpoint_path is not None:
        _checkpoint_fitness(history, run_id, checkpoint_path)

    optimal_params = np.array(result) if not isinstance(result, np.ndarray) else result
    logger.info("Run %d — training complete. Final history length: %d", run_id, len(history))
    return optimal_params, history


# -------------------------------------------------------------------------
# Local fallback (no polypus)
# -------------------------------------------------------------------------

def _scipy_pso_fallback(
    feature_map,
    ansatz,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, Any],
    run_id: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """
    Minimal PSO-like fallback using scipy differential evolution when polypus
    is not installed. Intended for local development / CI only.
    """
    from scipy.optimize import differential_evolution  # noqa: PLC0415
    from qiskit_aer import AerSimulator  # noqa: PLC0415
    from qiskit.primitives import StatevectorSampler  # noqa: PLC0415
    from .metrics import readout_probabilities, cross_entropy_loss  # noqa: PLC0415

    circuit = feature_map.compose(ansatz)
    n_params_fm = feature_map.num_parameters
    n_params_ans = ansatz.num_parameters

    backend = AerSimulator()
    num_shots = config.get("vqc_num_shots", 1024)
    readout_mode = config.get("vqc_readout", "single_qubit")
    n_qubits = feature_map.num_qubits
    maxiter = config.get("opt_maxiter", 70)
    history: list[dict[str, float]] = []
    gen_counter = [0]

    def _objective(params: np.ndarray) -> float:
        from qiskit import transpile  # noqa: PLC0415

        probs = []
        for x in X_train:
            param_vals = list(x[:n_params_fm]) + list(params)
            bound = circuit.assign_parameters(
                dict(zip(circuit.parameters, param_vals))
            )
            bound.measure_all()
            t = transpile(bound, backend)
            job = backend.run(t, shots=num_shots)
            counts = job.result().get_counts()
            probs.append(readout_probabilities(counts, n_qubits, readout_mode))

        y_prob = np.array(probs)
        loss = cross_entropy_loss(y_train, y_prob)
        gen_counter[0] += 1
        history.append(
            {
                "generation": gen_counter[0],
                "best_fitness": loss,
                "mean_fitness": loss,
            }
        )
        return loss

    bounds = [(-np.pi, np.pi)] * n_params_ans
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-np.pi, np.pi, n_params_ans)

    result = differential_evolution(
        _objective,
        bounds,
        maxiter=maxiter,
        seed=seed,
        x0=x0,
        tol=1e-4,
        workers=1,
    )
    return result.x, history


# -------------------------------------------------------------------------
# Inference
# -------------------------------------------------------------------------

def predict_proba(
    feature_map,
    ansatz,
    X: np.ndarray,
    params: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """
    Run the trained VQC on *X* and return per-sample probabilities.

    Parameters
    ----------
    feature_map : Qiskit QuantumCircuit
    ansatz      : Qiskit QuantumCircuit
    X           : (n_samples, n_features)
    params      : (n_params,) trained ansatz parameters
    config      : pipeline config

    Returns
    -------
    y_prob : (n_samples,) float
    """
    from qiskit_aer import AerSimulator  # noqa: PLC0415
    from qiskit import transpile  # noqa: PLC0415
    from .metrics import readout_probabilities  # noqa: PLC0415

    num_shots = config.get("vqc_num_shots", 1024)
    readout_mode = config.get("vqc_readout", "single_qubit")
    n_qubits = feature_map.num_qubits
    n_params_fm = feature_map.num_parameters

    infrastructure = config.get("vqc_test_infrastructure", "local")
    from .backends import get_backend  # noqa: PLC0415

    backend = get_backend(infrastructure, config)

    circuit = feature_map.compose(ansatz)
    circuit.measure_all()
    circuit_t = transpile(circuit, backend)

    probs = []
    for x in X:
        param_vals = list(x[:n_params_fm]) + list(params)
        bound = circuit_t.assign_parameters(
            dict(zip(circuit_t.parameters, param_vals))
        )
        job = backend.run(bound, shots=num_shots)
        counts = job.result().get_counts()
        probs.append(readout_probabilities(counts, n_qubits, readout_mode))

    return np.array(probs)
