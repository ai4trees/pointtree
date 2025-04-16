"""Tests for pointtree.instance_segmentation.PriorityQueue"""

import pytest

from pointtree.instance_segmentation import PriorityQueue


class TestPriorityQueue:
    """Tests for pointtree.instance_segmentation.PriorityQueue"""

    def test_valid_queue(self):
        pq = PriorityQueue()
        for key, entry, priority in zip(["a", "b", "c"], [("a", 1), ("b", 0), ("c", 2)], [1, 0, 2]):
            pq.add(key, entry, priority)

        priority_get_a, entry_get_a = pq.get("a")

        assert 1 == priority_get_a
        assert ("a", 1) == entry_get_a

        pq.update("c", ("d", 3))
        pq.add("b", ("b", -1), -1)

        popped_priorities = []
        popped_keys = []
        popped_entries = []
        while len(pq) > 0:
            priority, key, entry = pq.pop()
            popped_priorities.append(priority)
            popped_keys.append(key)
            popped_entries.append(entry)

        assert [-1, 1, 2] == popped_priorities
        assert ["b", "a", "c"] == popped_keys
        assert [("b", -1), ("a", 1), ("d", 3)] == popped_entries

    def test_get_invalid_key(self):
        pq = PriorityQueue()
        assert pq.get("a") is None

    def test_update_invalid_key(self):
        pq = PriorityQueue()
        with pytest.raises(KeyError):
            pq.update("a", ("a", 0))

    def test_pop_empty_queue(self):
        pq = PriorityQueue()
        with pytest.raises(KeyError):
            pq.pop()
