"""Context manager that suppresses logging to stdout while executing the wrapped code."""

__all__ = ["no_stdout"]

import contextlib
import os
import sys


@contextlib.contextmanager
def no_stdout():
    """Context manager that suppresses logging to stdout while executing the wrapped code."""
    original_stdout_fd = sys.stdout.fileno()
    saved_stdout_fd = os.dup(original_stdout_fd)
    null_fd = os.open(os.devnull, os.O_RDWR)
    os.dup2(null_fd, original_stdout_fd)
    os.close(null_fd)

    yield

    os.dup2(saved_stdout_fd, original_stdout_fd)
    os.close(saved_stdout_fd)
