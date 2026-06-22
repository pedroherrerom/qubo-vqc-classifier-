"""backends.py — CUNQA and local Aer execution backends.

Provides a unified ``get_sampler`` factory that returns either a
``qiskit_aer.AerSimulator``-backed sampler (local) or a CUNQA QPU sampler
(HPC), with automatic fallback to local Aer.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("vqc")


def get_aer_backend(num_shots: int = 1024):
    """Return a configured ``AerSimulator`` instance."""
    try:
        from qiskit_aer import AerSimulator  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "qiskit-aer is required for local simulation. "
            "Install it via: pip install qiskit-aer"
        ) from exc
    backend = AerSimulator()
    logger.debug("Aer backend initialised (shots=%d).", num_shots)
    return backend


def get_cunqa_backend(config: dict[str, Any]):
    """
    Return a CUNQA QPU backend.

    Falls back silently to local Aer if CUNQA is not available or no QPUs
    are provisioned.
    """
    try:
        import cunqa  # noqa: PLC0415, F401

        # CUNQA backend initialisation (cluster-specific)
        from cunqa import get_qpu  # noqa: PLC0415

        backend = get_qpu()
        logger.info("CUNQA QPU backend acquired.")
        return backend
    except (ImportError, Exception) as exc:
        logger.warning(
            "CUNQA backend unavailable (%s). Falling back to local Aer.", exc
        )
        return get_aer_backend(config.get("vqc_num_shots", 1024))


def get_backend(infrastructure: str, config: dict[str, Any]):
    """
    Factory: return the appropriate backend for *infrastructure*.

    Parameters
    ----------
    infrastructure : ``"local"`` or ``"cunqa"``
    config         : pipeline config dict
    """
    if infrastructure == "cunqa":
        return get_cunqa_backend(config)
    return get_aer_backend(config.get("vqc_num_shots", 1024))
