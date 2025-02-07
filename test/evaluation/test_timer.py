"""Tests for pointtree.evaluation.Timer."""

import time


from pointtree.evaluation import Timer, TimeTracker


def test_timer():
    """Test for pointtree.evaluation.Timer."""

    time_tracker = TimeTracker()
    timer = Timer("test", time_tracker)
    wait_time = 2

    with timer:
        time.sleep(2)

    tracked_times = time_tracker.to_pandas()
    tracked_time = tracked_times.loc[tracked_times["Description"] == "test", "Runtime"].to_numpy()

    assert len(tracked_time) == 1
    assert tracked_time[0] >= wait_time
