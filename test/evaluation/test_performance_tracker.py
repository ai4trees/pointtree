""" Tests for pointtree.evaluation.PerformanceTracker. """

import pandas as pd

from pointtree.evaluation import PerformanceTracker


def test_time_tracker():
    """Test for pointtree.evaluation.PerformanceTracker."""

    performance_tracker = PerformanceTracker()
    performance_tracker.save("test 1", 2, 2, 3, 2)
    performance_tracker.reset()
    performance_tracker.save("test 1", 1, 1, 2, 1)
    performance_tracker.save("test 1", 2, 2, 3, 2)
    performance_tracker.save("test 2", 3, 3, 4, 3)

    performance_metrics = performance_tracker.to_pandas()

    expected_performance_metrics = pd.DataFrame(
        [["test 1", 3, 3, 5, 3], ["test 2", 3, 3, 4, 3]],
        columns=["Description", "Wallclock Time [s]", "CPU Time [s]", "Memory Usage [GB]", "Memory Increment [GB]"],
    )

    assert list(expected_performance_metrics["Description"]) == list(performance_metrics["Description"])
    assert list(expected_performance_metrics["Wallclock Time [s]"]) == list(performance_metrics["Wallclock Time [s]"])
    assert list(expected_performance_metrics["CPU Time [s]"]) == list(performance_metrics["CPU Time [s]"])
    assert list(expected_performance_metrics["Memory Usage [GB]"]) == list(performance_metrics["Memory Usage [GB]"])
    assert list(expected_performance_metrics["Memory Increment [GB]"]) == list(
        performance_metrics["Memory Increment [GB]"]
    )
