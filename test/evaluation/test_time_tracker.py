""" Tests for pointtree.evaluation.TimeTracker. """

import pandas as pd

from pointtree.evaluation import TimeTracker


def test_time_tracker():
    """Test for pointtree.evaluation.TimeTracker."""

    time_tracker = TimeTracker()
    time_tracker.save("test 1", 2)
    time_tracker.reset()
    time_tracker.save("test 1", 1)
    time_tracker.save("test 1", 2)
    time_tracker.save("test 2", 3)

    tracked_times = time_tracker.to_pandas()

    expected_tracked_times = pd.DataFrame([["test 1", 3], ["test 2", 3]], columns=["Description", "Runtime"])

    assert list(expected_tracked_times["Description"]) == list(tracked_times["Description"])
    assert list(expected_tracked_times["Runtime"]) == list(tracked_times["Runtime"])
