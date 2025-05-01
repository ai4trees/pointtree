"""Tests for pointtree.evaluation.Profiler."""

import time

import numpy as np
import pytest


from pointtree.evaluation import Profiler, PerformanceTracker


def test_profiler():
    """Test for pointtree.evaluation.Profiler."""

    performance_tracker = PerformanceTracker()
    profiler = Profiler("test", performance_tracker)
    wait_time = 2
    memory_usage = 1e6 * 8 / 1e9

    data = np.empty(0, dtype=np.float64)  # pylint: disable=unused-variable
    with profiler:
        data = np.random.randn(int(1e6)).astype(np.float64)
        for _ in range(int(1e6)):  # busy waiting to obtain CPU time > 0
            pass
        time.sleep(2)

    performance_metrics = performance_tracker.to_pandas()
    tracked_wallclock_time = performance_metrics.loc[
        performance_metrics["Description"] == "test", "Wallclock Time [s]"
    ].to_numpy()
    tracked_cpu_time = performance_metrics.loc[performance_metrics["Description"] == "test", "CPU Time [s]"].to_numpy()
    tracked_memory_usage = performance_metrics.loc[
        performance_metrics["Description"] == "test", "Memory Usage [GB]"
    ].to_numpy()
    tracked_memory_increment = performance_metrics.loc[
        performance_metrics["Description"] == "test", "Memory Increment [GB]"
    ].to_numpy()

    assert len(performance_metrics) == 1
    assert tracked_wallclock_time[0] >= wait_time
    assert tracked_cpu_time[0] > 0
    assert tracked_memory_usage[0] >= memory_usage
    assert tracked_memory_increment >= memory_usage
