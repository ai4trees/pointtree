""" A context manager that tracks the execution time and memory usage of the contained code. """

__all__ = ["Profiler"]

import psutil
import os
import time

import numpy as np

from ._performance_tracker import PerformanceTracker


class Profiler:
    """
    A context manager that tracks the execution time and memory usage of the contained code.

    Args:
        desc: Description of the tracked code.
        performance_tracker: Performance tracker in which the measured performance metrics are to be stored.
    """

    def __init__(self, desc: str, performance_tracker: PerformanceTracker):
        self._desc = desc
        self._performance_tracker = performance_tracker
        self._start_time_wall_clock = None
        self._start_time_cpu = None
        self._start_memory = None

    def __enter__(self):
        process = psutil.Process(os.getpid())
        self._start_memory = process.memory_info().rss
        self._start_time_wall_clock = time.perf_counter()
        self._start_time_cpu = time.process_time()

    def __exit__(self, *_):
        execution_time_wall_clock = time.perf_counter() - self._start_time_wall_clock
        execution_time_cpu = time.process_time() - self._start_time_cpu
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss
        memory_increment = memory_usage - self._start_memory

        self._performance_tracker.save(
            self._desc,
            execution_time_wall_clock,
            execution_time_cpu,
            np.round(memory_usage / 1e9, 4),
            np.round(memory_increment / 1e9, 4),
        )
