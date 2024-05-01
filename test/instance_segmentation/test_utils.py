""" Tests for utility methods in forest3d.instance_segmentation. """

import numpy as np

from forest3d.instance_segmentation import remap_instance_ids


class TestUtils:
    """Tests for utility methods in forest3d.instance_segmentation."""

    def test_remap_instance_ids_empty(self):
        remapped_instance_ids, unique_instance_ids = remap_instance_ids(np.array([], dtype=np.int64))

        assert len(remapped_instance_ids) == 0
        assert len(unique_instance_ids) == 0

    def test_remap_instance_ids_no_remapping_necessary(self):
        num_instances = 10
        instance_ids = np.arange(num_instances, dtype=np.int64)

        remapped_instance_ids, unique_instance_ids = remap_instance_ids(instance_ids)

        np.testing.assert_array_equal(instance_ids, remapped_instance_ids)
        np.testing.assert_array_equal(instance_ids, unique_instance_ids)

    def test_remap_instance_ids_remapping_necessary(self):
        offset = 5
        num_instances = 10
        instance_ids = np.concatenate(
            [
                np.arange(num_instances / 2, dtype=np.int64),
                np.arange(num_instances / 2, dtype=np.int64),
                np.arange(offset + num_instances / 2, offset + num_instances, dtype=np.int64),
            ]
        )

        expected_remapped_instance_ids = np.concatenate(
            [
                np.arange(num_instances / 2, dtype=np.int64),
                np.arange(num_instances / 2, dtype=np.int64),
                np.arange(num_instances / 2, num_instances, dtype=np.int64),
            ]
        )
        expected_unique_instance_ids = np.arange(num_instances, dtype=np.int64)

        remapped_instance_ids, unique_instance_ids = remap_instance_ids(instance_ids)

        np.testing.assert_array_equal(expected_remapped_instance_ids, remapped_instance_ids)
        np.testing.assert_array_equal(expected_unique_instance_ids, unique_instance_ids)
