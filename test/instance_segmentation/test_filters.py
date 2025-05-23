"""Tests for :code:`pointtree.instance_segmentation.filters`."""

import numpy as np
import pytest

from pointtree.instance_segmentation.filters import (
    filter_instances_intensity,
    filter_instances_min_points,
    filter_instances_pca,
    filter_instances_vertical_extent,
)


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

    @pytest.mark.parametrize("inplace", [True, False])
    def test_filter_instances_vertical_extent(self, inplace: bool):
        xyz = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0], [1, 0, 1], [1, 1, 10]], dtype=np.float64)
        instance_ids = np.array([0] * 2 + [-1] + [1] * 2, dtype=np.int64)
        original_instance_ids = instance_ids.copy()
        unique_instance_ids = np.array([0, 1], dtype=np.int64)
        expected_filtered_instance_ids = np.array([-1] * 3 + [0] * 2, dtype=np.int64)

        filtered_instance_ids, filtered_unique_instance_ids = filter_instances_vertical_extent(
            xyz, instance_ids, unique_instance_ids, min_vertical_extent=3, inplace=inplace
        )

        np.testing.assert_array_equal(expected_filtered_instance_ids, filtered_instance_ids)
        np.testing.assert_array_equal(np.array([0], dtype=np.int64), filtered_unique_instance_ids)

        if inplace:
            np.testing.assert_array_equal(expected_filtered_instance_ids, instance_ids)
        else:
            np.testing.assert_array_equal(original_instance_ids, instance_ids)

    def test_filter_instances_vertical_extent_none(self):
        xyz = np.zeros((10, 3), dtype=np.float64)
        instance_ids = np.array([0] * 5 + [1] * 5, dtype=np.int64)
        unique_instance_ids = instance_ids

        filtered_instance_ids, filtered_unique_instance_ids = filter_instances_vertical_extent(
            xyz, instance_ids, unique_instance_ids, min_vertical_extent=None
        )

        np.testing.assert_array_equal(instance_ids, filtered_instance_ids)
        np.testing.assert_array_equal(unique_instance_ids, filtered_unique_instance_ids)

    @pytest.mark.parametrize("min_explained_variance", [None, 0.9])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_filter_instances_pca_variance(self, min_explained_variance: float, inplace: bool):
        instance_ids = np.array([0] * 3 + [-1] + [1] * 5, dtype=np.int64)
        original_instance_ids = instance_ids.copy()
        unique_instance_ids = np.array([0, 1], dtype=np.int64)

        if min_explained_variance is not None:
            expected_filtered_instance_ids = np.array([0] * 3 + [-1] * 6, dtype=np.int64)
            expected_unique_instance_ids = np.array([0], dtype=np.int64)
        else:
            expected_filtered_instance_ids = original_instance_ids
            expected_unique_instance_ids = np.array([0, 1], dtype=np.int64)

        xyz = np.array(
            [
                [0, 0, 0],
                [0, 0, 0.5],
                [0, 0, 1],
                [1, 1, 0],
                [2.6, 2.1, 0],
                [2, 2, 1],
                [1.5, 0.8, 2],
                [2, 2, 3],
                [2, 2, 4],
            ],
            dtype=np.float64,
        )

        filtered_instance_ids, filtered_unique_instance_ids = filter_instances_pca(
            xyz,
            instance_ids,
            unique_instance_ids,
            min_explained_variance=min_explained_variance,
            max_inclination=45,
            inplace=inplace,
        )

        np.testing.assert_array_equal(expected_filtered_instance_ids, filtered_instance_ids)
        np.testing.assert_array_equal(expected_unique_instance_ids, filtered_unique_instance_ids)

        if inplace:
            np.testing.assert_array_equal(expected_filtered_instance_ids, instance_ids)
        else:
            np.testing.assert_array_equal(original_instance_ids, instance_ids)

    @pytest.mark.parametrize("max_inclination", [None, 45])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_filter_instances_pca_angle(self, max_inclination: float, inplace: bool):
        instance_ids = np.array([0] * 3 + [-1] + [1] * 3, dtype=np.int64)
        original_instance_ids = instance_ids.copy()
        unique_instance_ids = np.array([0, 1], dtype=np.int64)

        if max_inclination is not None:
            expected_filtered_instance_ids = np.array([0] * 3 + [-1] * 4, dtype=np.int64)
            expected_unique_instance_ids = np.array([0], dtype=np.int64)
        else:
            expected_filtered_instance_ids = original_instance_ids
            expected_unique_instance_ids = np.array([0, 1], dtype=np.int64)

        xyz = np.array(
            [[0, 0, 0], [0, 0, 0.5], [0, 0, 1], [1, 1, 0], [2, 2, 0], [2.5, 2, 0], [3, 2, 0]], dtype=np.float64
        )

        filtered_instance_ids, filtered_unique_instance_ids = filter_instances_pca(
            xyz,
            instance_ids,
            unique_instance_ids,
            min_explained_variance=0.5,
            max_inclination=max_inclination,
            inplace=inplace,
        )

        np.testing.assert_array_equal(expected_filtered_instance_ids, filtered_instance_ids)
        np.testing.assert_array_equal(expected_unique_instance_ids, filtered_unique_instance_ids)

        if inplace:
            np.testing.assert_array_equal(expected_filtered_instance_ids, instance_ids)
        else:
            np.testing.assert_array_equal(original_instance_ids, instance_ids)

    def test_filter_instances_pca_none(self):
        xyz = np.zeros((10, 3), dtype=np.float64)
        instance_ids = np.array([0] * 5 + [1] * 5, dtype=np.int64)
        unique_instance_ids = instance_ids

        filtered_instance_ids, filtered_unique_instance_ids = filter_instances_pca(
            xyz,
            instance_ids,
            unique_instance_ids,
            min_explained_variance=None,
            max_inclination=None,
        )

        np.testing.assert_array_equal(instance_ids, filtered_instance_ids)
        np.testing.assert_array_equal(unique_instance_ids, filtered_unique_instance_ids)

    def test_filter_instances_intensity(self):
        intensities = np.array([6000, 4000, 7000, 4000, 3000, 6000], dtype=np.float64)
        instance_ids = np.array([0] * 3 + [-1] + [1] * 2, dtype=np.int64)
        unique_instance_ids = np.array([0, 1], dtype=np.int64)
        expected_filtered_instance_ids = np.array([0] * 3 + [-1] * 3, dtype=np.int64)

        filtered_instance_ids, filtered_unique_instance_ids = filter_instances_intensity(
            intensities, instance_ids, unique_instance_ids, min_intensity=5000, threshold_percentile=0.6
        )

        np.testing.assert_array_equal(expected_filtered_instance_ids, filtered_instance_ids)
        np.testing.assert_array_equal(np.array([0], dtype=np.int64), filtered_unique_instance_ids)

    def test_test_filter_instances_intensity_none(self):
        intensities = np.zeros((10), dtype=np.float64)
        instance_ids = np.array([0] * 5 + [1] * 5, dtype=np.int64)
        unique_instance_ids = instance_ids

        filtered_instance_ids, filtered_unique_instance_ids = filter_instances_intensity(
            intensities, instance_ids, unique_instance_ids, min_intensity=None, threshold_percentile=0.8
        )

        np.testing.assert_array_equal(instance_ids, filtered_instance_ids)
        np.testing.assert_array_equal(unique_instance_ids, filtered_unique_instance_ids)
