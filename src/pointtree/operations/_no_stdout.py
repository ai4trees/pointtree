"""Context manager that suppresses logging to stdout while executing the wrapped code."""

__all__ = ["no_stdout"]

import contextlib
import io
import os
import sys


@contextlib.contextmanager
def no_stdout():
    """Context manager that suppresses logging to stdout while executing the wrapped code."""
    try:
        original_stdout_fd = sys.stdout.fileno()
    except (AttributeError, io.UnsupportedOperation):
        # in some cases, sys.stdout has no real file descriptor (e.g. in Jupyter)
        # in these cases, we fall back to a Python-level redirect, which still suppresses stdout from the calling code,
        # though not from native/C extensions writing directly to the OS-level stdout.
        with contextlib.redirect_stdout(io.StringIO()):
            yield
        return

    saved_stdout_fd = os.dup(original_stdout_fd)
    null_fd = os.open(os.devnull, os.O_RDWR)
    os.dup2(null_fd, original_stdout_fd)
    os.close(null_fd)

    try:
        yield
    finally:
        os.dup2(saved_stdout_fd, original_stdout_fd)
        os.close(saved_stdout_fd)
