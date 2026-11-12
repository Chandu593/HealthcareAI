"""Structured logging setup shared by all pipeline phases.

Per project Section 33, every phase should report start/completion,
execution time, rows/patients/hospitals processed, and warnings/errors.
This module provides a consistent logger + a small PhaseTimer context
manager so individual phase modules don't reimplement this boilerplate.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional


def setup_logging(log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("eicu_fl")
    logger.setLevel(level)
    for handler in logger.handlers[:]:
        handler.close()
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / "pipeline.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def close_logging(logger: Optional[logging.Logger] = None) -> None:
    """Closes and detaches all handlers from the pipeline logger.

    FIXED 2026-08-24: an open logging.FileHandler holds an OS-level
    lock on its target file on Windows. setup_logging() only closes
    the PREVIOUS handler when called again at the START of the NEXT
    phase -- nothing ever closes the LAST phase's handler at the end
    of a run. tempfile.TemporaryDirectory.cleanup() (every integration
    test's tearDown()) then fails with `PermissionError: [WinError 32]
    ... pipeline.log` trying to delete a log file still held open by
    that handler. This never surfaced on Linux, where deleting an open
    file is permitted -- only on a real Windows run. Call this in
    tearDown(), before cleaning up any tempdir that a log_dir pointed
    into.
    """
    logger = logger or logging.getLogger("eicu_fl")
    for handler in logger.handlers[:]:
        handler.close()
    logger.handlers.clear()


class PhaseTimer:
    """Context manager that logs phase start/end + elapsed time.

    Usage:
        with PhaseTimer(logger, "PHASE1_INGESTION"):
            ... do work ...
    """

    def __init__(self, logger: logging.Logger, phase_name: str):
        self.logger = logger
        self.phase_name = phase_name
        self._start: float = 0.0

    def __enter__(self) -> "PhaseTimer":
        self._start = time.time()
        self.logger.info(f"[{self.phase_name}] START")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = time.time() - self._start
        if exc_type is None:
            self.logger.info(f"[{self.phase_name}] COMPLETE ({elapsed:.2f}s)")
        else:
            self.logger.error(f"[{self.phase_name}] FAILED after {elapsed:.2f}s: {exc_val}")
