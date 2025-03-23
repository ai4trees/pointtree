"""Tests for pointtree.evaluation.semantic_segmentation_metrics."""

import numpy as np
import pytest


from pointtree.evaluation import semantic_segmentation_metrics


class TestSemanticSegmentationMetrics:
    """Tests for pointtree.evaluation.semantic_segmentation_metrics."""

    def test_semantic_segmentation_metrics(self):  # pylint: disable=too-many-locals
        target = np.array([1, 1, 1, 0, 0, 2, 1, 1, 3, 2], dtype=np.int64)
        prediction = np.array([1, 1, 0, 2, 0, 2, 2, 1, 3, 3], dtype=np.int64)
        class_map = {"Ground": 0, "Tree": 1, "LowVegetation": 2, "Building": 3}
        aggregate_classes = {"Vegetation": [1, 2], "Other": [0, 3]}

        metrics = semantic_segmentation_metrics(target, prediction, class_map, aggregate_classes=aggregate_classes)

        ground_iou = 1 / 3
        tree_iou = 3 / 5
        low_vegetation_iou = 1 / 4
        vegetation_iou = 5 / 8
        building_iou = 1 / 2
        other_iou = 2 / 5

        assert metrics["GroundIoU"] == ground_iou
        assert metrics["TreeIoU"] == tree_iou
        assert metrics["LowVegetationIoU"] == low_vegetation_iou
        assert metrics["VegetationIoU"] == vegetation_iou
        assert metrics["BuildingIoU"] == building_iou
        assert metrics["MeanIoU"] == (ground_iou + tree_iou + low_vegetation_iou + building_iou) / 4
        assert metrics["MeanIoUAggregated"] == (vegetation_iou + other_iou) / 2

        ground_precision = 1 / 2
        tree_precision = 1
        low_vegetation_precision = 1 / 3
        building_precision = 1 / 2
        vegetation_precision = 5 / 6
        other_precision = 2 / 4

        assert metrics["GroundPrecision"] == ground_precision
        assert metrics["TreePrecision"] == tree_precision
        assert metrics["LowVegetationPrecision"] == low_vegetation_precision
        assert metrics["VegetationPrecision"] == vegetation_precision
        assert metrics["BuildingPrecision"] == building_precision
        assert (
            metrics["MeanPrecision"]
            == (ground_precision + tree_precision + low_vegetation_precision + building_precision) / 4
        )
        assert metrics["MeanPrecisionAggregated"] == (vegetation_precision + other_precision) / 2

        ground_recall = 1 / 2
        tree_recall = 3 / 5
        low_vegetation_recall = 1 / 2
        vegetation_recall = 5 / 7
        building_recall = 1
        other_recall = 2 / 3

        assert metrics["GroundRecall"] == ground_recall
        assert metrics["TreeRecall"] == tree_recall
        assert metrics["LowVegetationRecall"] == low_vegetation_recall
        assert metrics["VegetationRecall"] == vegetation_recall
        assert metrics["BuildingRecall"] == building_recall
        assert metrics["MeanRecall"] == (ground_recall + tree_recall + low_vegetation_recall + building_recall) / 4
        assert metrics["MeanRecallAggregated"] == (vegetation_recall + other_recall) / 2

    def test_semantic_segmentation_metrics_all_correct(self):
        target = np.random.randint(0, 3, (20,), dtype=np.int64)
        class_map = {"Ground": 0, "Tree": 1, "LowVegetation": 2}
        aggregate_classes = {"Vegetation": [1, 2]}

        metrics = semantic_segmentation_metrics(target, target, class_map, aggregate_classes=aggregate_classes)

        assert metrics["GroundIoU"] == 1
        assert metrics["TreeIoU"] == 1
        assert metrics["LowVegetationIoU"] == 1
        assert metrics["VegetationIoU"] == 1

        assert metrics["GroundPrecision"] == 1
        assert metrics["TreePrecision"] == 1
        assert metrics["LowVegetationPrecision"] == 1
        assert metrics["VegetationPrecision"] == 1

        assert metrics["GroundRecall"] == 1
        assert metrics["TreeRecall"] == 1
        assert metrics["LowVegetationRecall"] == 1
        assert metrics["VegetationRecall"] == 1

    def test_semantic_segmentation_metrics_invalid(self):
        target = np.random.randint(0, 3, (20,), dtype=np.int64)
        prediction = np.random.randint(0, 3, (21,), dtype=np.int64)
        class_map = {}

        with pytest.raises(ValueError):
            semantic_segmentation_metrics(target, prediction, class_map)
