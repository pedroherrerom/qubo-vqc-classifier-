"""serialization.py — NumPy-safe JSON encoder and Polypus stdout interceptor."""

from __future__ import annotations

import json
import os
import sys
import threading
from io import StringIO
from typing import Any


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy scalars and arrays."""

    def default(self, obj: Any) -> Any:  # noqa: ANN401
        try:
            import numpy as np  # noqa: PLC0415

            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
        except ImportError:
            pass
        return super().default(obj)


def dump_json(obj: Any, path: str | os.PathLike, **kwargs: Any) -> None:
    """Serialize *obj* to *path* using :class:`NumpyEncoder`."""
    with open(path, "w") as fh:
        json.dump(obj, fh, cls=NumpyEncoder, indent=2, **kwargs)


def load_json(path: str | os.PathLike) -> Any:
    """Load JSON from *path*."""
    with open(path) as fh:
        return json.load(fh)


class _PolypusStdoutInterceptor:
    """
    OS-level file-descriptor pipe that captures stdout written by Rust extensions
    (e.g. polypus) without touching Python's ``sys.stdout``.

    Usage::

        with _PolypusStdoutInterceptor() as cap:
            polypus.qml.train(...)
        lines = cap.lines
    """

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._buf = StringIO()
        self._orig_fd: int | None = None
        self._pipe_r: int | None = None
        self._pipe_w: int | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    def __enter__(self) -> "_PolypusStdoutInterceptor":
        self._pipe_r, self._pipe_w = os.pipe()
        self._orig_fd = os.dup(sys.stdout.fileno())
        os.dup2(self._pipe_w, sys.stdout.fileno())
        os.close(self._pipe_w)
        self._pipe_w = None

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        sys.stdout.flush()
        os.dup2(self._orig_fd, sys.stdout.fileno())  # type: ignore[arg-type]
        os.close(self._orig_fd)  # type: ignore[arg-type]
        self._orig_fd = None
        os.close(self._pipe_r)  # type: ignore[arg-type]
        self._pipe_r = None
        if self._thread is not None:
            self._thread.join()

    def _reader(self) -> None:
        with os.fdopen(self._pipe_r, "r", errors="replace") as fh:  # type: ignore[arg-type]
            for line in fh:
                self._lines.append(line.rstrip("\n"))

    @property
    def lines(self) -> list[str]:
        return list(self._lines)
