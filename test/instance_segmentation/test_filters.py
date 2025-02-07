"""Tests for :code:`pointtree.instance_segmentation.filters`."""

import numpy as np

from pointtree.instance_segmentation.filters import filter_instances_min_points, filter_instances_vertical_extent


class TestFilterInstances:
    """Tests for :code:`pointtree.instance_segmentation.filters`."""

    def test_filter_instances_min_points(self):
        instance_ids = np.array([0] * 3 + [-1] * 7 + [1] * 10, dtype=np.int64)
        unique_instance_ids = np.array([0, 1], dtype=np.int64)
        expected_filtered_instance_ids = np.array([-1] * 10 + [0] * 10, dtype=np.int64)

        filtered_instance_ids, filtered_unique_instance_ids = filter_instances_min_points(
            instance_ids, unique_instance_ids, min_points=5
        )

        np.testing.assert_array_equal(expected_filtered_instance_ids, filtered_instance_ids)
        np.testing.assert_array_equal(np.array([0], dtype=np.int64), filtered_unique_instance_ids)

    def test_filter_instances_min_points_none(self):
        instance_ids = np.arange(10, dtype=np.int64)
        unique_instance_ids = instance_ids

        filtered_instance_ids, filtered_unique_instance_ids = filter_instances_min_points(
            instance_ids, unique_instance_ids, min_points=None
        )

        np.testing.assert_array_equal(instance_ids, filtered_instance_ids)
        np.testing.assert_array_equal(unique_instance_ids, filtered_unique_instance_ids)

    def test_filter_instances_vertical_extent(self):
        coords = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0], [1, 0, 1], [1, 1, 10]], dtype=np.float64)
        instance_ids = np.array([0] * 2 + [-1] + [1] * 2, dtype=np.int64)
        unique_instance_ids = np.array([0, 1], dtype=np.int64)
        expected_filtered_instance_ids = np.array([-1] * 3 + [0] * 2, dtype=np.int64)

        filtered_instance_ids, filtered_unique_instance_ids = filter_instances_vertical_extent(
            coords, instance_ids, unique_instance_ids, min_vertical_extent=3
        )

        np.testing.assert_array_equal(expected_filtered_instance_ids, filtered_instance_ids)
        np.testing.assert_array_equal(np.array([0], dtype=np.int64), filtered_unique_instance_ids)

    def test_filter_instances_vertical_extent_none(self):
        coords = np.zeros((10, 3), dtype=np.float64)
        instance_ids = np.array([0] * 5 + [1] * 5, dtype=np.int64)
        unique_instance_ids = instance_ids

        filtered_instance_ids, filtered_unique_instance_ids = filter_instances_vertical_extent(
            coords, instance_ids, unique_instance_ids, min_vertical_extent=None
        )

        np.testing.assert_array_equal(instance_ids, filtered_instance_ids)
        np.testing.assert_array_equal(unique_instance_ids, filtered_unique_instance_ids)
