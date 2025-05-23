"""Tests for pointtorch.instance_segmentation.tree_x_presets."""

from typing import Any

import numpy as np
import pytest

from pointtree.instance_segmentation import TreeXAlgorithm
from pointtree.instance_segmentation.tree_x_presets import TreeXPresetDefault, TreeXPresetTLS, TreeXPresetULS

from test.utils import (  # pylint: disable=wrong-import-order
    generate_tree_point_cloud,
)


class TestTreeXPresets:
    """Tests for pointtorch.instance_segmentation.tree_x_presets."""

    @pytest.mark.parametrize("preset", [TreeXPresetDefault(), TreeXPresetTLS(), TreeXPresetULS()])
    def test_full_algorithm_no_trees_detected(self, preset: Any):
        assert len(preset) > 0

        xyz, _, _, _, _ = generate_tree_point_cloud(np.float64, "C", generate_intensities=False)

        algorithm = TreeXAlgorithm(**preset)

        instance_ids, _, _ = algorithm(xyz)

        assert len(xyz) == len(instance_ids)
