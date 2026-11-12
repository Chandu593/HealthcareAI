import logging
import unittest

from src.utils.batching import is_memory_error, run_batched_with_backoff


def _silent_logger() -> logging.Logger:
    logger = logging.getLogger("test_batching")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)  # suppress output during tests
    return logger


class TestIsMemoryError(unittest.TestCase):
    def test_python_memory_error_is_retryable(self):
        self.assertTrue(is_memory_error(MemoryError("out of memory")))

    def test_torch_style_runtime_error_is_retryable(self):
        exc = RuntimeError(
            "[enforce fail at alloc_cpu.cpp:117] data. DefaultCPUAllocator: "
            "not enough memory: you tried to allocate 5780424704 bytes."
        )
        self.assertTrue(is_memory_error(exc))

    def test_cuda_oom_style_message_is_retryable(self):
        self.assertTrue(is_memory_error(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")))

    def test_unrelated_runtime_error_is_not_retryable(self):
        self.assertFalse(is_memory_error(RuntimeError("shape mismatch between tensors")))

    def test_unrelated_exception_type_is_not_retryable(self):
        self.assertFalse(is_memory_error(ValueError("bad input")))


class TestRunBatchedWithBackoff(unittest.TestCase):
    def test_processes_all_items_no_errors(self):
        processed_ranges = []

        def process_batch(start, end):
            processed_ranges.append((start, end))

        run_batched_with_backoff(
            n_items=100, process_batch=process_batch, initial_batch_size=30, logger=_silent_logger(),
        )
        # Ranges must cover [0,100) exactly, no gaps, no overlaps, no duplicates
        covered = set()
        for start, end in processed_ranges:
            for i in range(start, end):
                self.assertNotIn(i, covered, "item processed more than once")
                covered.add(i)
        self.assertEqual(covered, set(range(100)))

    def test_zero_items_is_a_noop(self):
        calls = []
        run_batched_with_backoff(
            n_items=0, process_batch=lambda s, e: calls.append((s, e)),
            initial_batch_size=10, logger=_silent_logger(),
        )
        self.assertEqual(calls, [])

    def test_backoff_on_retryable_error_halves_batch_size_and_retries_same_range(self):
        """Simulates the real scenario: batch_size=100 raises OOM,
        50 raises OOM, 25 succeeds. Must not skip or duplicate any item,
        and must not lose progress already made on earlier batches."""
        call_log = []
        processed = set()

        def process_batch(start, end):
            batch_size = end - start
            call_log.append((start, end, batch_size))
            if start == 0 and batch_size > 25:
                raise RuntimeError("not enough memory: you tried to allocate 999 bytes")
            processed.update(range(start, end))

        final_batch_size = run_batched_with_backoff(
            n_items=60, process_batch=process_batch, initial_batch_size=100,
            logger=_silent_logger(), min_batch_size=1,
        )
        self.assertEqual(processed, set(range(60)))
        self.assertLessEqual(final_batch_size, 25)
        first_batch_attempts = [c for c in call_log if c[0] == 0]
        self.assertGreater(len(first_batch_attempts), 1)

    def test_non_retryable_error_propagates_immediately_without_retry(self):
        call_count = 0

        def process_batch(start, end):
            nonlocal call_count
            call_count += 1
            raise ValueError("this is not a memory error")

        with self.assertRaises(ValueError):
            run_batched_with_backoff(
                n_items=10, process_batch=process_batch, initial_batch_size=5, logger=_silent_logger(),
            )
        self.assertEqual(call_count, 1, "non-retryable error must not be retried")

    def test_raises_when_even_min_batch_size_fails(self):
        def always_oom(start, end):
            raise RuntimeError("not enough memory: allocate 1 byte")

        with self.assertRaises(RuntimeError):
            run_batched_with_backoff(
                n_items=10, process_batch=always_oom, initial_batch_size=8,
                logger=_silent_logger(), min_batch_size=1,
            )

    def test_does_not_loop_forever_at_min_batch_size(self):
        """Regression guard: once batch_size == min_batch_size and the
        error persists, it must raise on the very next failure, not
        attempt an infinite halving-that-never-goes-below-1 loop."""
        attempts = {"count": 0}

        def process_batch(start, end):
            attempts["count"] += 1
            if attempts["count"] > 20:
                self.fail("looped far more than needed to reach min_batch_size -- possible infinite loop")
            raise RuntimeError("not enough memory")

        with self.assertRaises(RuntimeError):
            run_batched_with_backoff(
                n_items=1000, process_batch=process_batch, initial_batch_size=1000,
                logger=_silent_logger(), min_batch_size=1,
            )

    def test_invalid_initial_batch_size_raises_valueerror(self):
        with self.assertRaises(ValueError):
            run_batched_with_backoff(
                n_items=10, process_batch=lambda s, e: None,
                initial_batch_size=0, logger=_silent_logger(), min_batch_size=1,
            )

    def test_batch_size_only_shrinks_never_grows_back(self):
        """Documents current behavior: batch_size only ever shrinks
        within a single run_batched_with_backoff call, never grows back.
        Intentional (simpler; a transient OOM is unlikely to un-happen
        mid-run) -- pinned here so a future change is deliberate, not an
        accidental regression."""
        sizes_seen = []

        def process_batch(start, end):
            sizes_seen.append(end - start)
            if start == 0 and (end - start) > 5:
                raise RuntimeError("not enough memory")

        run_batched_with_backoff(
            n_items=40, process_batch=process_batch, initial_batch_size=40, logger=_silent_logger(),
        )
        self.assertTrue(all(s <= sizes_seen[1] for s in sizes_seen[1:]))

    def test_final_batch_may_be_smaller_than_batch_size(self):
        """n_items not evenly divisible by batch_size -- last batch
        must still be processed correctly (partial batch)."""
        processed_ranges = []
        run_batched_with_backoff(
            n_items=25, process_batch=lambda s, e: processed_ranges.append((s, e)),
            initial_batch_size=10, logger=_silent_logger(),
        )
        self.assertEqual(processed_ranges, [(0, 10), (10, 20), (20, 25)])


if __name__ == "__main__":
    unittest.main()
