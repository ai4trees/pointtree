"""Tests for the metric calculation methods in pointtree.evaluation."""

import numpy as np
import pandas as pd
import pytest


from pointtree.evaluation import match_instances, instance_segmentation_metrics, semantic_segmentation_metrics


class TestMetrics:
    """Tests for the metric calculation methods in pointtree.evaluation."""

    def test_match_instances(self):
        target = np.array([1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 1, 3, 3, 3, 3], dtype=np.int64)
        prediction = np.array([2, 2, 2, 2, -1, 3, 1, 0, 0, 1, 2, -1, -1, -1, -1], dtype=np.int64)

        expected_matched_target_ids = np.array([0, -1, 1, -1], dtype=np.int64)
        expected_matched_predicted_ids = np.array([0, 2, -1, -1], dtype=np.int64)
        expected_metrics = pd.DataFrame(
            [[1, 2, 4 / 5, 4 / 5, 1], [0, 0, 2 / 3, 1, 2 / 3]],
            columns=["Target", "Prediction", "IoU", "Precision", "Recall"],
        )
        expected_metrics.sort_values(by="IoU", inplace=True)

        matched_target_ids, matched_predicted_ids, metrics = match_instances(target, prediction)

        metrics.sort_values(by="IoU", inplace=True)

        np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids)
        np.testing.assert_array_equal(matched_predicted_ids, expected_matched_predicted_ids)
        np.testing.assert_array_equal(expected_metrics["Target"], metrics["Target"])
        np.testing.assert_array_equal(expected_metrics["Prediction"], metrics["Prediction"])
        np.testing.assert_array_equal(expected_metrics["IoU"], metrics["IoU"])
        np.testing.assert_array_equal(expected_metrics["Precision"], metrics["Precision"])
        np.testing.assert_array_equal(expected_metrics["Recall"], metrics["Recall"])

    def test_match_instances_all_correct(self):
        target = np.arange(10, dtype=np.int64)
        prediction = np.arange(10, dtype=np.int64)

        matched_target_ids, matched_predicted_ids, ious = match_instances(target, prediction)

        np.testing.assert_array_equal(target, matched_target_ids)
        np.testing.assert_array_equal(matched_predicted_ids, prediction)
        np.testing.assert_array_equal(target, ious["Target"])
        np.testing.assert_array_equal(prediction, ious["Prediction"])
        np.testing.assert_array_equal(np.ones(len(target), dtype=np.float64), ious["IoU"])
        np.testing.assert_array_equal(np.ones(len(target), dtype=np.float64), ious["Precision"])
        np.testing.assert_array_equal(np.ones(len(target), dtype=np.float64), ious["Recall"])

    def test_match_instances_non_continuous_target_ids(self):
        target = np.array([0, 2], dtype=np.int64)
        prediction = np.zeros_like(target)

        with pytest.raises(ValueError):
            match_instances(target, prediction)

    def test_match_instances_non_continuous_prediction_ids(self):
        prediction = np.array([0, 2], dtype=np.int64)
        target = np.zeros_like(prediction)

        with pytest.raises(ValueError):
            match_instances(target, prediction)

    def test_instance_segmentation_metrics(self):
        matched_target_ids = np.array([1, 0, -1, 2], dtype=np.int64)
        matched_predicted_ids = np.array([0, 1, 3, -1, -1], dtype=np.int64)
        segmentation_metrics = pd.DataFrame(
            [[1, 0, 0.6, 0.5, 0.7], [0, 1, 0.7, 0.6, 0.8], [2, 3, 0.8, 0.7, 0.9]],
            columns=["Target", "Prediction", "IoU", "Precision", "Recall"],
        )

        metrics = instance_segmentation_metrics(matched_target_ids, matched_predicted_ids, segmentation_metrics)

        expected_f1_score = 2 * 3 / (2 * 3 + 1 + 2)
        expected_m_iou = 0.7
        expected_m_precision = 0.6
        expected_m_recall = 0.8

        assert metrics["detection_recall"] == 3 / 5
        assert metrics["detection_precision"] == 3 / 4
        assert metrics["detection_f1_score"] == pytest.approx(expected_f1_score)
        assert metrics["segmentation_m_iou"] == pytest.approx(expected_m_iou)
        assert metrics["segmentation_m_precision"] == pytest.approx(expected_m_precision)
        assert metrics["segmentation_m_recall"] == pytest.approx(expected_m_recall)
        assert metrics["panoptic_quality"] == pytest.approx(expected_f1_score * expected_m_iou)

    def test_instance_segmentation_metrics_all_correct(self):
        matched_target_ids = np.arange(10, dtype=np.int64)
        matched_predicted_ids = np.arange(10, dtype=np.int64)
        segmentation_metrics = pd.DataFrame(
            np.column_stack([matched_target_ids, matched_predicted_ids]), columns=["Target", "Prediction"]
        )
        segmentation_metrics["IoU"] = 1
        segmentation_metrics["Precision"] = 1
        segmentation_metrics["Recall"] = 1

        metrics = instance_segmentation_metrics(matched_target_ids, matched_predicted_ids, segmentation_metrics)

        assert metrics["detection_recall"] == 1
        assert metrics["detection_precision"] == 1
        assert metrics["detection_f1_score"] == 1
        assert metrics["segmentation_m_iou"] == 1
        assert metrics["segmentation_m_precision"] == 1
        assert metrics["segmentation_m_recall"] == 1
        assert metrics["panoptic_quality"] == 1

    def test_instance_segmentation_metrics_invalid(self):
        matched_target_ids = np.array([1, 0, -1], dtype=np.int64)
        matched_predicted_ids = np.array([0, 1, 3, -1, -1], dtype=np.int64)
        segmentation_metrics = pd.DataFrame(
            [[1, 0, 0.6, 0.5, 0.7], [0, 1, 0.7, 0.6, 0.8], [2, 3, 0.8, 0.7, 0.9]],
            columns=["Target", "Prediction", "IoU", "Precision", "Recall"],
        )

        with pytest.raises(ValueError):
            instance_segmentation_metrics(matched_target_ids, matched_predicted_ids, segmentation_metrics)

    def test_semantic_segmentation_metrics(self):  # pylint: disable=too-many-locals
        target = np.array([1, 1, 1, 0, 0, 2, 1, 1, 3, 2], dtype=np.int64)
        prediction = np.array([1, 1, 0, 2, 0, 2, 2, 1, 3, 3], dtype=np.int64)
        class_map = {"ground": 0, "tree": 1, "low_vegetation": 2, "building": 3}
        aggregate_classes = {"vegetation": [1, 2], "other": [0, 3]}

        metrics = semantic_segmentation_metrics(target, prediction, class_map, aggregate_classes=aggregate_classes)

        ground_iou = 1 / 3
        tree_iou = 3 / 5
        low_vegetation_iou = 1 / 4
        vegetation_iou = 5 / 8
        building_iou = 1 / 2
        other_iou = 2 / 5

        assert metrics["ground_iou"] == ground_iou
        assert metrics["tree_iou"] == tree_iou
        assert metrics["low_vegetation_iou"] == low_vegetation_iou
        assert metrics["vegetation_iou"] == vegetation_iou
        assert metrics["building_iou"] == building_iou
        assert metrics["m_iou"] == (ground_iou + tree_iou + low_vegetation_iou + building_iou) / 4
        assert metrics["m_iou_aggregated"] == (vegetation_iou + other_iou) / 2

        ground_precision = 1 / 2
        tree_precision = 1
        low_vegetation_precision = 1 / 3
        building_precision = 1 / 2
        vegetation_precision = 5 / 6
        other_precision = 2 / 4

        assert metrics["ground_precision"] == ground_precision
        assert metrics["tree_precision"] == tree_precision
        assert metrics["low_vegetation_precision"] == low_vegetation_precision
        assert metrics["vegetation_precision"] == vegetation_precision
        assert metrics["building_precision"] == building_precision
        assert (
            metrics["m_precision"]
            == (ground_precision + tree_precision + low_vegetation_precision + building_precision) / 4
        )
        assert metrics["m_precision_aggregated"] == (vegetation_precision + other_precision) / 2

        ground_recall = 1 / 2
        tree_recall = 3 / 5
        low_vegetation_recall = 1 / 2
        vegetation_recall = 5 / 7
        building_recall = 1
        other_recall = 2 / 3

        assert metrics["ground_recall"] == ground_recall
        assert metrics["tree_recall"] == tree_recall
        assert metrics["low_vegetation_recall"] == low_vegetation_recall
        assert metrics["vegetation_recall"] == vegetation_recall
        assert metrics["building_recall"] == building_recall
        assert metrics["m_recall"] == (ground_recall + tree_recall + low_vegetation_recall + building_recall) / 4
        assert metrics["m_recall_aggregated"] == (vegetation_recall + other_recall) / 2

    def test_semantic_segmentation_metrics_all_correct(self):
        target = np.random.randint(0, 3, (20,), dtype=np.int64)
        class_map = {"ground": 0, "tree": 1, "low_vegetation": 2}
        aggregate_classes = {"vegetation": [1, 2]}

        metrics = semantic_segmentation_metrics(target, target, class_map, aggregate_classes=aggregate_classes)

        assert metrics["ground_iou"] == 1
        assert metrics["tree_iou"] == 1
        assert metrics["low_vegetation_iou"] == 1
        assert metrics["vegetation_iou"] == 1

        assert metrics["ground_precision"] == 1
        assert metrics["tree_precision"] == 1
        assert metrics["low_vegetation_precision"] == 1
        assert metrics["vegetation_precision"] == 1

        assert metrics["ground_recall"] == 1
        assert metrics["tree_recall"] == 1
        assert metrics["low_vegetation_recall"] == 1
        assert metrics["vegetation_recall"] == 1

    def test_semantic_segmentation_metrics_invalid(self):
        target = np.random.randint(0, 3, (20,), dtype=np.int64)
        prediction = np.random.randint(0, 3, (21,), dtype=np.int64)
        class_map = {}

        with pytest.raises(ValueError):
            semantic_segmentation_metrics(target, prediction, class_map)
