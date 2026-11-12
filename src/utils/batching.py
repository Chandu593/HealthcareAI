"""
Generic adaptive-batch-size processing with automatic backoff on
retryable errors (out-of-memory being the motivating case).

WHY THIS EXISTS
----------------
A real full-mode run (199,108 patients) crashed inside the LSTM
encoder's forward pass.

Root cause: Phase 4 ran ALL 199,108 patients through the LSTM in a
SINGLE forward pass -- there was no batching at all. An LSTM's forward
pass needs internal buffers roughly proportional to
(total_timesteps_in_batch x hidden_dim x num_layers x 4_gates), and
with every patient in one batch, total_timesteps was ~2.8M -- an
~11.6GB-order-of-magnitude working set, consistent with the failure.

-------
This module is intentionally torch-agnostic: it knows nothing about
LSTMs, tensors, or even what "a batch" computes. It only owns batch
SIZING, RETRY, and PROGRESS LOGGING. The caller supplies a
`process_batch(start, end)` callback that does the actual work (and is
responsible for writing its own output, e.g. into a pre-allocated
array) plus an `is_retryable_error(exc)` predicate. This separation
means the batching/backoff logic itself is fully unit-testable without
needing torch installed (this project's development sandbox has no
torch and no internet to install it -- see docs/timeseries_modality.md)
-- the tests inject a fake process_batch that raises a controlled
exception, with zero dependency on the real LSTM code path.

On a retryable error, the batch size is halved and the SAME [start,
end) range is retried -- no progress is lost, and no range is skipped
or double-processed. If batch_size would drop below `min_batch_size`,
the error is re-raised rather than retried forever.
"""

from __future__ import annotations

import gc
import logging
from typing import Callable


def is_memory_error(exc: Exception) -> bool:
    """True for Python's own MemoryError, and for the RuntimeError
    messages PyTorch's C++ allocator raises on both CPU and CUDA
    out-of-memory (these surface as RuntimeError in Python, not
    MemoryError, so string-matching the message is necessary)."""
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return any(kw in msg for kw in ("out of memory", "not enough memory", "allocate", "alloc_cpu", "cuda oom"))
    return False


def run_batched_with_backoff(
    n_items: int,
    process_batch: Callable[[int, int], None],
    initial_batch_size: int,
    logger: logging.Logger,
    is_retryable_error: Callable[[Exception], bool] = is_memory_error,
    min_batch_size: int = 1,
    log_every_n_batches: int = 20,
    gc_every_n_batches: int = 50,
    label: str = "BATCH",
) -> int:
    """Processes [0, n_items) in batches, shrinking batch_size on a
    retryable error and retrying the same range. Returns the final
    batch_size actually used (useful for logging/diagnostics -- if it
    ended up much smaller than initial_batch_size, that's a signal the
    configured default is too aggressive for this machine).

    Raises whatever `process_batch` raises if it's not retryable, or if
    it's retryable but batch_size has already hit `min_batch_size`.
    """
    if n_items == 0:
        return initial_batch_size
    if initial_batch_size < min_batch_size:
        raise ValueError(f"initial_batch_size ({initial_batch_size}) < min_batch_size ({min_batch_size})")

    start = 0
    batch_size = initial_batch_size
    n_batches_completed = 0

    while start < n_items:
        end = min(start + batch_size, n_items)
        try:
            process_batch(start, end)
        except Exception as exc:  # noqa: BLE001 -- must inspect ANY exception type via is_retryable_error
            if not is_retryable_error(exc) or batch_size <= min_batch_size:
                raise
            new_batch_size = max(min_batch_size, batch_size // 2)
            logger.warning(
                f"[{label}] retryable error at batch_size={batch_size} "
                f"(items {start}-{end} of {n_items}): {exc} -- "
                f"retrying with batch_size={new_batch_size}"
            )
            batch_size = new_batch_size
            gc.collect()
            continue  # retry the SAME `start` -- do not advance, nothing was lost

        start = end
        n_batches_completed += 1
        if n_batches_completed % log_every_n_batches == 0 or start >= n_items:
            logger.info(f"[{label}] progress: {start:,}/{n_items:,} (batch_size={batch_size})")
        if n_batches_completed % gc_every_n_batches == 0:
            gc.collect()

    return batch_size
