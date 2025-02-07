"""A context manager that tracks the execution time of the contained code."""

__all__ = ["Timer"]

from time import perf_counter

from ._time_tracker import TimeTracker


class Timer:
    """
    A context manager that tracks the execution time of the contained code.

    Args:
        desc: Description of the tracked code.
        time_tracker: Time tracker in which the measured execution time is to be stored.
    """

    def __init__(self, desc: str, time_tracker: TimeTracker):
        self._desc = desc
        self._time_tracker = time_tracker
        self._start_time = None

    def __enter__(self):
        self._start_time = perf_counter()

    def __exit__(self, *_):
        execution_time = perf_counter() - self._start_time
        self._time_tracker.save(self._desc, execution_time)
