"""Tests for src/utils/logging_setup.py.

Includes a regression test for a real Windows-only bug found via the
user's real run (2026-08-24): every integration test's tearDown() calls
tempfile.TemporaryDirectory.cleanup(), which failed with
`PermissionError: [WinError 32] ... pipeline.log` -- an open
logging.FileHandler was still holding that file open, because
setup_logging() only closes the PREVIOUS handler when called again at
the start of the NEXT phase; nothing ever closed the LAST phase's
handler at the end of a run. This never surfaced on Linux, where
deleting an open file is permitted, only on the user's real machine.

The failure mode itself (PermissionError on delete) is Windows-only and
can't be reproduced in this sandbox -- but the underlying, portable
CONTRACT the fix relies on ("after close_logging(), zero handlers
remain attached, and the file handler's stream is genuinely closed, not
just detached") is fully testable and enforced here, platform-
independent.
"""

import logging
import os
import shutil
import tempfile
import unittest

from src.utils.logging_setup import close_logging, setup_logging


class TestSetupLogging(unittest.TestCase):
    def tearDown(self):
        close_logging()

    def test_attaches_stream_and_file_handler_when_log_dir_given(self):
        tmpdir = tempfile.mkdtemp()
        try:
            logger = setup_logging(log_dir=os.path.join(tmpdir, "logs"))
            handler_types = {type(h).__name__ for h in logger.handlers}
            self.assertEqual(handler_types, {"StreamHandler", "FileHandler"})
        finally:
            close_logging()
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_attaches_only_stream_handler_without_log_dir(self):
        logger = setup_logging(log_dir=None)
        handler_types = {type(h).__name__ for h in logger.handlers}
        self.assertEqual(handler_types, {"StreamHandler"})

    def test_calling_again_closes_previous_handlers_not_just_accumulates(self):
        tmpdir = tempfile.mkdtemp()
        try:
            setup_logging(log_dir=os.path.join(tmpdir, "logs_a"))
            logger = setup_logging(log_dir=os.path.join(tmpdir, "logs_b"))
            # Still exactly 2 handlers (stream + file), not 4 -- confirms
            # setup_logging() doesn't just keep appending on repeat calls
            # (this part of the behavior already existed before the
            # 2026-08-24 fix and is locked in here too).
            self.assertEqual(len(logger.handlers), 2)
        finally:
            close_logging()
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestCloseLogging(unittest.TestCase):
    def tearDown(self):
        close_logging()

    def test_detaches_all_handlers(self):
        tmpdir = tempfile.mkdtemp()
        try:
            logger = setup_logging(log_dir=os.path.join(tmpdir, "logs"))
            self.assertGreater(len(logger.handlers), 0)
            close_logging()
            self.assertEqual(len(logger.handlers), 0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_file_handler_stream_is_genuinely_closed_not_just_detached(self):
        # This is the actual regression target: logging.FileHandler
        # sets its own `.stream` attribute to None inside close() once
        # the underlying OS file descriptor is released -- checking
        # this directly (rather than just "is it detached from the
        # logger") is what proves the OS-level lock a Windows delete
        # would trip on is actually gone, not just that Python stopped
        # routing messages to it.
        tmpdir = tempfile.mkdtemp()
        try:
            logger = setup_logging(log_dir=os.path.join(tmpdir, "logs"))
            file_handler = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
            self.assertIsNotNone(file_handler.stream, "sanity check: handler should start open")

            close_logging()

            self.assertIsNone(
                file_handler.stream,
                "FileHandler.stream still set after close_logging() -- the OS-level file "
                "handle isn't actually released, which is exactly what causes "
                "PermissionError: [WinError 32] on tempdir cleanup on Windows.",
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_log_file_can_be_deleted_immediately_after_close_logging(self):
        # The actual end-to-end contract every integration test's
        # tearDown() now relies on.
        tmpdir = tempfile.mkdtemp()
        log_dir = os.path.join(tmpdir, "logs")
        logger = setup_logging(log_dir=log_dir)
        logger.info("a message, so the file actually exists and has been written to")

        close_logging()

        shutil.rmtree(tmpdir)  # would raise/PermissionError-equivalent on Windows pre-fix
        self.assertFalse(os.path.exists(tmpdir))

    def test_safe_to_call_when_no_handlers_attached(self):
        close_logging()  # already closed by a previous test/call
        close_logging()  # must not raise on an already-clean logger
        logger = logging.getLogger("eicu_fl")
        self.assertEqual(len(logger.handlers), 0)

    def test_setup_logging_works_normally_after_close_logging(self):
        """Guards against a fix that closes handlers correctly but
        leaves the logger in a state where a subsequent setup_logging()
        call (the very next phase/test) misbehaves."""
        tmpdir = tempfile.mkdtemp()
        try:
            setup_logging(log_dir=os.path.join(tmpdir, "logs_1"))
            close_logging()

            logger = setup_logging(log_dir=os.path.join(tmpdir, "logs_2"))
            self.assertEqual(len(logger.handlers), 2)
            logger.info("works fine after a close/reopen cycle")
        finally:
            close_logging()
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
