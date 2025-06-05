"""Tests for pointtree.evaluation.instance_segmentation_metrics."""

import numpy as np
import pytest

from pointtree.evaluation import (
    instance_detection_metrics,
    instance_segmentation_metrics,
    instance_segmentation_metrics_per_partition,
    evaluate_instance_segmentation,
)


class TestInstanceSegmentationMetrics:  # pylint: disable=too-many-public-methods
    """Tests for pointtree.evaluation.instance_segmentation_metrics."""

    @pytest.mark.parametrize("min_precision_fp", [0.5, 0.6])
    def test_instance_detection_metrics(self, min_precision_fp: float):
        target = np.array([1, 1, 1, 0, 0, 2, -1], dtype=np.int64)
        prediction = np.array([0, -1, 2, 2, 2, 1, 1], dtype=np.int64)

        matched_predicted_ids = np.array([2, -1, -1], dtype=np.int64)
        matched_target_ids = np.array([-1, -1, 0], dtype=np.int64)

        metrics = instance_detection_metrics(
            target,
            prediction,
            matched_predicted_ids,
            matched_target_ids,
            invalid_instance_id=-1,
            min_precision_fp=min_precision_fp,
        )

        assert metrics["TP"] == 1
        assert metrics["FP"] == 2 if min_precision_fp <= 0.5 else 1
        assert metrics["FN"] == 2
        assert metrics["Precision"] == metrics["TP"] / (metrics["TP"] + metrics["FP"])
        assert metrics["CommissionError"] == metrics["FP"] / (metrics["TP"] + metrics["FP"])
        assert metrics["Recall"] == metrics["TP"] / (metrics["TP"] + metrics["FN"])
        assert metrics["OmissionError"] == metrics["FN"] / (metrics["TP"] + metrics["FN"])
        assert metrics["F1Score"] == 2 * metrics["TP"] / (2 * metrics["TP"] + metrics["FP"] + metrics["FN"])

    def test_instance_detection_metrics_labeled_mask(self):
        target = np.array([1, 1, 1, 0, 0, 2, -1, -1, -1, -1, -1], dtype=np.int64)
        prediction = np.array([0, -1, 2, 2, 2, 1, 1, -1, 3, 3, 3], dtype=np.int64)
        labeled_mask = np.array([True] * 7 + [False] * 4, dtype=bool)

        matched_predicted_ids = np.array([2, -1, -1], dtype=np.int64)
        matched_target_ids = np.array([-1, -1, 0, -1], dtype=np.int64)

        metrics = instance_detection_metrics(
            target,
            prediction,
            matched_predicted_ids,
            matched_target_ids,
            invalid_instance_id=-1,
            min_precision_fp=0.5,
            labeled_mask=labeled_mask,
        )

        assert metrics["TP"] == 1
        assert metrics["FP"] == 2
        assert metrics["FN"] == 2
        assert metrics["Precision"] == metrics["TP"] / (metrics["TP"] + metrics["FP"])
        assert metrics["CommissionError"] == metrics["FP"] / (metrics["TP"] + metrics["FP"])
        assert metrics["Recall"] == metrics["TP"] / (metrics["TP"] + metrics["FN"])
        assert metrics["OmissionError"] == metrics["FN"] / (metrics["TP"] + metrics["FN"])
        assert metrics["F1Score"] == 2 * metrics["TP"] / (2 * metrics["TP"] + metrics["FP"] + metrics["FN"])

    @pytest.mark.parametrize("min_precision_fp", [0, 0.5])
    def test_instance_detection_metrics_all_correct(self, min_precision_fp: float):
        target = np.array([1, 1, 1, 0, 0, 2, -1, 2], dtype=np.int64)
        prediction = np.array([0, 0, 0, 2, 2, 1, -1, 1], dtype=np.int64)

        matched_predicted_ids = np.array([2, 0, 1], dtype=np.int64)
        matched_target_ids = np.array([1, 2, 0], dtype=np.int64)

        metrics = instance_detection_metrics(
            target,
            prediction,
            matched_predicted_ids,
            matched_target_ids,
            invalid_instance_id=-1,
            min_precision_fp=min_precision_fp,
        )

        assert metrics["TP"] == 3
        assert metrics["FP"] == 0
        assert metrics["FN"] == 0
        assert metrics["Precision"] == 1
        assert metrics["CommissionError"] == 0
        assert metrics["Recall"] == 1
        assert metrics["OmissionError"] == 0
        assert metrics["F1Score"] == 1

    def test_instance_detection_metrics_invalid_prediction(self):
        target = np.zeros(5, dtype=np.int64)
        prediction = np.zeros(4, dtype=np.int64)

        matched_predicted_ids = np.array([0], dtype=np.int64)
        matched_target_ids = np.array([0], dtype=np.int64)

        with pytest.raises(ValueError):
            instance_detection_metrics(
                target,
                prediction,
                matched_predicted_ids,
                matched_target_ids,
                invalid_instance_id=-1,
            )

    def test_instance_detection_metrics_invalid_matched_predicted_ids(self):
        target = np.zeros(5, dtype=np.int64)
        prediction = np.zeros(5, dtype=np.int64)

        matched_predicted_ids = np.array([0, 1], dtype=np.int64)
        matched_target_ids = np.array([0], dtype=np.int64)

        with pytest.raises(ValueError):
            instance_detection_metrics(
                target,
                prediction,
                matched_predicted_ids,
                matched_target_ids,
                invalid_instance_id=-1,
            )

    def test_instance_detection_metrics_invalid_matched_target_ids(self):
        target = np.zeros(5, dtype=np.int64)
        prediction = np.zeros(5, dtype=np.int64)

        matched_predicted_ids = np.array([0], dtype=np.int64)
        matched_target_ids = np.array([0, 1], dtype=np.int64)

        with pytest.raises(ValueError):
            instance_detection_metrics(
                target,
                prediction,
                matched_predicted_ids,
                matched_target_ids,
                invalid_instance_id=-1,
            )

    def test_instance_detection_metrics_all_false_negatives(self):
        target = np.array([1, 1, 1, 0, 0], dtype=np.int64)
        prediction = np.array([-1, -1, -1, -1, -1], dtype=np.int64)

        matched_predicted_ids = np.array([-1, -1], dtype=np.int64)
        matched_target_ids = np.array([], dtype=np.int64)

        metrics = instance_detection_metrics(
            target,
            prediction,
            matched_predicted_ids,
            matched_target_ids,
            invalid_instance_id=-1,
        )

        assert metrics["TP"] == 0
        assert metrics["FP"] == 0
        assert metrics["FN"] == 2
        assert np.isnan(metrics["Precision"])
        assert np.isnan(metrics["CommissionError"])
        assert metrics["Recall"] == 0
        assert metrics["OmissionError"] == 1
        assert metrics["F1Score"] == 0

    def test_instance_detection_metrics_all_false_positives(self):
        target = np.array([-1, -1, -1, -1, -1], dtype=np.int64)
        prediction = np.array([1, 1, 1, 0, 0], dtype=np.int64)

        matched_predicted_ids = np.array([], dtype=np.int64)
        matched_target_ids = np.array([-1, -1], dtype=np.int64)

        metrics = instance_detection_metrics(
            target, prediction, matched_predicted_ids, matched_target_ids, invalid_instance_id=-1, min_precision_fp=0
        )

        assert metrics["TP"] == 0
        assert metrics["FP"] == 2
        assert metrics["FN"] == 0
        assert metrics["Precision"] == 0
        assert metrics["CommissionError"] == 1
        assert np.isnan(metrics["Recall"])
        assert np.isnan(metrics["OmissionError"])
        assert metrics["F1Score"] == 0

    def test_instance_segmentation_metrics(self):
        target = np.array([1, 1, 1, 1, 0, 0, 0, 0, -1, 2], dtype=np.int64)
        prediction = np.array([0, 1, 0, 0, 0, 2, 2, 2, -1, -1], dtype=np.int64)

        matched_predicted_ids = np.array([2, 0, -1], dtype=np.int64)

        metrics, per_instance_metrics = instance_segmentation_metrics(
            target, prediction, matched_predicted_ids, invalid_instance_id=-1
        )

        assert metrics["MeanIoU"] == (3 / 5 + 3 / 4) / 2
        assert metrics["MeanRecall"] == (3 / 4 + 3 / 4) / 2
        assert metrics["MeanPrecision"] == (3 / 4 + 3 / 3) / 2

        assert len(per_instance_metrics) == 2
        assert per_instance_metrics.loc[per_instance_metrics["TargetID"] == 0, "PredictionID"].iloc[0] == 2
        assert per_instance_metrics.loc[per_instance_metrics["TargetID"] == 0, "IoU"].iloc[0] == 3 / 4
        assert per_instance_metrics.loc[per_instance_metrics["TargetID"] == 0, "Recall"].iloc[0] == 3 / 4
        assert per_instance_metrics.loc[per_instance_metrics["TargetID"] == 0, "Precision"].iloc[0] == 3 / 3

        assert per_instance_metrics.loc[per_instance_metrics["TargetID"] == 1, "PredictionID"].iloc[0] == 0
        assert per_instance_metrics.loc[per_instance_metrics["TargetID"] == 1, "Recall"].iloc[0] == 3 / 4
        assert per_instance_metrics.loc[per_instance_metrics["TargetID"] == 1, "Precision"].iloc[0] == 3 / 4

    def test_instance_segmentation_metrics_all_correct(self):
        target = np.array([1, 1, 1, 0, 0, 2, -1, 2], dtype=np.int64)
        prediction = np.array([0, 0, 0, 2, 2, 1, -1, 1], dtype=np.int64)

        matched_predicted_ids = np.array([2, 0, 1], dtype=np.int64)

        metrics, per_instance_metrics = instance_segmentation_metrics(
            target, prediction, matched_predicted_ids, invalid_instance_id=-1
        )

        assert metrics["MeanIoU"] == 1
        assert metrics["MeanRecall"] == 1
        assert metrics["MeanPrecision"] == 1

        assert len(per_instance_metrics) == 3
        assert per_instance_metrics.loc[per_instance_metrics["TargetID"] == 0, "PredictionID"].iloc[0] == 2
        assert per_instance_metrics.loc[per_instance_metrics["TargetID"] == 1, "PredictionID"].iloc[0] == 0
        assert per_instance_metrics.loc[per_instance_metrics["TargetID"] == 2, "PredictionID"].iloc[0] == 1

        assert (per_instance_metrics["IoU"] == 1).all()
        assert (per_instance_metrics["Precision"] == 1).all()
        assert (per_instance_metrics["Recall"] == 1).all()

    def test_instance_segmentation_metrics_no_matches(self):
        target = np.array([1, 1, -1, 0, 0], dtype=np.int64)
        prediction = np.array([-1, -1, -1, -1, -1], dtype=np.int64)

        matched_predicted_ids = np.array([-1, -1], dtype=np.int64)

        metrics, per_instance_metrics = instance_segmentation_metrics(
            target, prediction, matched_predicted_ids, invalid_instance_id=-1
        )

        assert np.isnan(metrics["MeanIoU"])
        assert np.isnan(metrics["MeanRecall"])
        assert np.isnan(metrics["MeanPrecision"])
        assert len(per_instance_metrics) == 0
        assert "TargetID" in per_instance_metrics
        assert "PredictionID" in per_instance_metrics

    def test_instance_segmentation_metrics_invalid_prediction(self):
        target = np.zeros(5, dtype=np.int64)
        prediction = np.zeros(4, dtype=np.int64)

        matched_predicted_ids = np.array([0], dtype=np.int64)

        with pytest.raises(ValueError):
            instance_segmentation_metrics(target, prediction, matched_predicted_ids, invalid_instance_id=-1)

    def test_instance_segmentation_metrics_invalid_matched_predicted_ids(self):
        target = np.zeros(5, dtype=np.int64)
        prediction = np.zeros(5, dtype=np.int64)

        matched_predicted_ids = np.array([0, 1], dtype=np.int64)

        with pytest.raises(ValueError):
            instance_segmentation_metrics(target, prediction, matched_predicted_ids, invalid_instance_id=-1)

    def test_instance_segmentation_metrics_per_xy_partition(self):

        xyz = np.array(
            [
                # tree 0
                [0, 0, 0],
                [0, 0, 0.1],
                [0, 0, 1],
                [1, 0, 1],
                [1, 0, 1],
                [1, 0, 1],
                [2, 0, 1],
                [3, 0, 1],
                [4, 0, 1],
                [5, 0, 1],
                [8, 0, 1],
                [9, 0, 1],
                [10, 0, 1],
                # tree 1
                [3, 0, 0],
                [3, 0, 0.1],
                [3, 0, 1],
                [4, 0, 1],
                [5, 0, 1],
                [8, 0, 1],
                [9, 0, 1],
                [10, 0, 1],
                [11, 0, 1],
                [12, 0, 1],
                [13, 0, 1],
            ],
            dtype=np.float64,
        )

        target = np.array([0] * 13 + [1] * 11, dtype=np.int64)
        prediction = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] + [0] * 11, dtype=np.int64)

        matched_predicted_ids = np.array([1, 0], dtype=np.int64)

        metrics, inst_metrics = instance_segmentation_metrics_per_partition(
            xyz, target, prediction, matched_predicted_ids, partition="xy", invalid_instance_id=-1
        )

        assert len(metrics) == 10

        correct_part = [0, 3, 4, 5, 6, 7, 8, 9]

        assert (metrics.loc[metrics["Partition"].isin(correct_part), "MeanIoU"] == 1).all()
        assert (metrics.loc[metrics["Partition"].isin(correct_part), "MeanPrecision"] == 1).all()
        assert (metrics.loc[metrics["Partition"].isin(correct_part), "MeanRecall"] == 1).all()

        assert metrics.loc[metrics["Partition"] == 1, "MeanIoU"].iloc[0] == (1 / 3 + 1) / 2
        assert metrics.loc[metrics["Partition"] == 1, "MeanPrecision"].iloc[0] == 1
        assert metrics.loc[metrics["Partition"] == 1, "MeanRecall"].iloc[0] == (1 / 3 + 1) / 2

        assert metrics.loc[metrics["Partition"] == 2, "MeanIoU"].iloc[0] == (1 / 3 + 1) / 2
        assert metrics.loc[metrics["Partition"] == 2, "MeanPrecision"].iloc[0] == (1 / 3 + 1) / 2
        assert metrics.loc[metrics["Partition"] == 2, "MeanRecall"].iloc[0] == 1

        assert len(inst_metrics) == 20

        target_0_mask = inst_metrics["TargetID"] == 0
        correct_part = [0, 2, 3, 4, 5, 8, 9]

        assert (inst_metrics.loc[target_0_mask & inst_metrics["Partition"].isin(correct_part), "IoU"] == 1).all()
        assert (inst_metrics.loc[target_0_mask & inst_metrics["Partition"].isin(correct_part), "Precision"] == 1).all()
        assert (inst_metrics.loc[target_0_mask & inst_metrics["Partition"].isin(correct_part), "Recall"] == 1).all()

        assert inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 1), "IoU"].iloc[0] == 1 / 3
        assert inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 1), "Precision"].iloc[0] == 1
        assert inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 1), "Recall"].iloc[0] == 1 / 3

        assert np.isnan(inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 6), "IoU"].iloc[0])
        assert np.isnan(inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 6), "Precision"].iloc[0])
        assert np.isnan(inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 6), "Recall"].iloc[0])

        assert np.isnan(inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 7), "IoU"].iloc[0])
        assert np.isnan(inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 7), "Precision"].iloc[0])
        assert np.isnan(inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 7), "Recall"].iloc[0])

        target_1_mask = inst_metrics["TargetID"] == 1
        correct_part = [0, 1, 5, 6, 7, 8, 9]

        assert (inst_metrics.loc[target_1_mask & inst_metrics["Partition"].isin(correct_part), "IoU"] == 1).all()
        assert (inst_metrics.loc[target_1_mask & inst_metrics["Partition"].isin(correct_part), "Precision"] == 1).all()
        assert (inst_metrics.loc[target_1_mask & inst_metrics["Partition"].isin(correct_part), "Recall"] == 1).all()

        assert inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 2), "IoU"].iloc[0] == 1 / 3
        assert inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 2), "Precision"].iloc[0] == 1 / 3
        assert inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 2), "Recall"].iloc[0] == 1

    def test_instance_segmentation_metrics_per_xy_partition_all_correct(self):

        xyz = np.array(
            [
                # tree 0
                [0, 0, 0],
                [0, 0, 0.1],
                [0, 0, 1],
                [1, 0, 1],
                [2, 0, 1],
                [3, 0, 1],
                [4, 0, 1],
                [5, 0, 1],
                [6, 0, 1],
                [7, 0, 1],
                [8, 0, 1],
                [9, 0, 1],
                [10, 0, 1],
                # tree 1
                [3, 3, 0],
                [3, 3, 0.1],
                [3, 3, 1],
                [4, 3, 1],
                [5, 3, 1],
                [6, 3, 1],
                [7, 3, 1],
                [8, 3, 1],
                [9, 3, 1],
                [10, 3, 1],
                [11, 3, 1],
                [12, 3, 1],
                [13, 3, 1],
            ],
            dtype=np.float64,
        )

        target = np.array([0] * 13 + [1] * 13, dtype=np.int64)
        prediction = np.array([1] * 13 + [0] * 13, dtype=np.int64)

        matched_predicted_ids = np.array([1, 0], dtype=np.int64)

        metrics, metrics_per_instance = instance_segmentation_metrics_per_partition(
            xyz, target, prediction, matched_predicted_ids, partition="xy", invalid_instance_id=-1
        )

        assert len(metrics) == 10
        assert (metrics["MeanIoU"] == 1).all()
        assert (metrics["MeanPrecision"] == 1).all()
        assert (metrics["MeanRecall"] == 1).all()

        assert len(metrics_per_instance) == 20
        assert (metrics_per_instance["IoU"] == 1).all()
        assert (metrics_per_instance["Precision"] == 1).all()
        assert (metrics_per_instance["Recall"] == 1).all()

    def test_instance_segmentation_metrics_per_z_partition(self):

        xyz = np.array(
            [
                # tree 0
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 2],
                [0, 0, 3],
                [0, 0, 4],
                [0, 0, 5],
                [0, 0, 7],
                [0, 0, 9],
                [0, 0, 10],
                # tree 1
                [3, 3, 0],
                [3, 3, 1],
                [3, 3, 2],
                [3, 3, 3],
                [3, 3, 4],
                [3, 3, 5],
                [3, 3, 7],
                [3, 3, 8],
                [3, 3, 10],
                # tree 2
                [5, 5, 0],
                [5, 5, 1],
            ],
            dtype=np.float64,
        )

        target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1] * 9 + [2] * 2, dtype=np.int64)
        prediction = np.array([1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] + [0] * 9 + [-1] * 2, dtype=np.int64)

        matched_predicted_ids = np.array([1, 0, -1], dtype=np.int64)

        metrics, inst_metrics = instance_segmentation_metrics_per_partition(
            xyz, target, prediction, matched_predicted_ids, partition="z", invalid_instance_id=-1
        )

        assert len(metrics) == 10

        correct_part = [0, 2, 3, 4, 5, 7, 8, 9]

        assert (metrics.loc[metrics["Partition"].isin(correct_part), "MeanIoU"] == 1).all()
        assert (metrics.loc[metrics["Partition"].isin(correct_part), "MeanPrecision"] == 1).all()
        assert (metrics.loc[metrics["Partition"].isin(correct_part), "MeanRecall"] == 1).all()

        assert metrics.loc[metrics["Partition"] == 1, "MeanIoU"].iloc[0] == 1 / 3
        assert metrics.loc[metrics["Partition"] == 1, "MeanPrecision"].iloc[0] == (1 / 3 + 1) / 2
        assert metrics.loc[metrics["Partition"] == 1, "MeanRecall"].iloc[0] == (1 / 3 + 1) / 2

        assert np.isnan(metrics.loc[metrics["Partition"] == 6, "MeanIoU"].iloc[0])
        assert np.isnan(metrics.loc[metrics["Partition"] == 6, "MeanPrecision"].iloc[0])
        assert np.isnan(metrics.loc[metrics["Partition"] == 6, "MeanRecall"].iloc[0])

        assert len(inst_metrics) == 20

        target_0_mask = inst_metrics["TargetID"] == 0

        assert inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 1), "IoU"].iloc[0] == 1 / 3
        assert inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 1), "Precision"].iloc[0] == 1
        assert inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 1), "Recall"].iloc[0] == 1 / 3

        assert np.isnan(inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 6), "IoU"].iloc[0])
        assert np.isnan(inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 6), "Precision"].iloc[0])
        assert np.isnan(inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 6), "Recall"].iloc[0])

        assert np.isnan(inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 8), "IoU"].iloc[0])
        assert np.isnan(inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 8), "Precision"].iloc[0])
        assert np.isnan(inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 8), "Recall"].iloc[0])

        target_1_mask = inst_metrics["TargetID"] == 1

        assert inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 1), "IoU"].iloc[0] == 1 / 3
        assert inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 1), "Precision"].iloc[0] == 1 / 3
        assert inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 1), "Recall"].iloc[0] == 1

        assert np.isnan(inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 6), "IoU"].iloc[0])
        assert np.isnan(inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 6), "Precision"].iloc[0])
        assert np.isnan(inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 6), "Recall"].iloc[0])

        assert np.isnan(inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 9), "IoU"].iloc[0])
        assert np.isnan(inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 9), "Precision"].iloc[0])
        assert np.isnan(inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 9), "Recall"].iloc[0])

    def test_instance_segmentation_metrics_per_z_partition_all_correct(self):

        xyz = np.array(
            [
                # tree 0
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 2],
                [0, 0, 3],
                [0, 0, 4],
                [0, 0, 5],
                [0, 0, 6],
                [0, 0, 7],
                [0, 0, 8],
                [0, 0, 9],
                [0, 0, 10],
                # tree 1
                [3, 3, 0],
                [3, 3, 1],
                [3, 3, 2],
                [3, 3, 3],
                [3, 3, 4],
                [3, 3, 5],
                [3, 3, 6],
                [3, 3, 7],
                [3, 3, 8],
                [3, 3, 9],
                [3, 3, 10],
            ],
            dtype=np.float64,
        )

        target = np.array([0] * 11 + [1] * 11, dtype=np.int64)
        prediction = np.array([1] * 11 + [0] * 11, dtype=np.int64)

        matched_predicted_ids = np.array([1, 0], dtype=np.int64)

        metrics, metrics_per_instance = instance_segmentation_metrics_per_partition(
            xyz, target, prediction, matched_predicted_ids, partition="z", invalid_instance_id=-1
        )

        assert len(metrics) == 10
        assert (metrics["MeanIoU"] == 1).all()
        assert (metrics["MeanPrecision"] == 1).all()
        assert (metrics["MeanRecall"] == 1).all()

        assert len(metrics_per_instance) == 20
        assert (metrics_per_instance["IoU"] == 1).all()
        assert (metrics_per_instance["Precision"] == 1).all()
        assert (metrics_per_instance["Recall"] == 1).all()

    def test_instance_segmentation_metrics_per_partition_invalid_partition(self):
        xyz = np.random.randn(5, 3)
        target = np.zeros(5, dtype=np.int64)
        prediction = np.zeros(5, dtype=np.int64)

        matched_predicted_ids = np.array([0], dtype=np.int64)

        with pytest.raises(ValueError):
            instance_segmentation_metrics_per_partition(
                xyz, target, prediction, matched_predicted_ids, partition="test"
            )

    @pytest.mark.parametrize("partition", ["xy", "z"])
    def test_instance_segmentation_metrics_per_partition_invalid_xyz(self, partition: str):
        xyz = np.random.randn(4, 3)
        target = np.zeros(5, dtype=np.int64)
        prediction = np.zeros(5, dtype=np.int64)

        matched_predicted_ids = np.array([0], dtype=np.int64)

        with pytest.raises(ValueError):
            instance_segmentation_metrics_per_partition(
                xyz, target, prediction, matched_predicted_ids, partition=partition
            )

    @pytest.mark.parametrize("partition", ["xy", "z"])
    def test_instance_segmentation_metrics_per_partition_invalid_prediction(self, partition: str):
        xyz = np.random.randn(5, 3)
        target = np.zeros(5, dtype=np.int64)
        prediction = np.zeros(4, dtype=np.int64)

        matched_predicted_ids = np.array([0], dtype=np.int64)

        with pytest.raises(ValueError):
            instance_segmentation_metrics_per_partition(
                xyz, target, prediction, matched_predicted_ids, partition=partition
            )

    @pytest.mark.parametrize("partition", ["xy", "z"])
    def test_instance_segmentation_metrics_per_partition_invalid_matched_predicted_ids(self, partition: str):
        xyz = np.random.randn(5, 3)
        target = np.zeros(5, dtype=np.int64)
        prediction = np.zeros(5, dtype=np.int64)

        matched_predicted_ids = np.array([0, 1], dtype=np.int64)

        with pytest.raises(ValueError):
            instance_segmentation_metrics_per_partition(
                xyz, target, prediction, matched_predicted_ids, partition=partition
            )

    @pytest.mark.parametrize("num_partitions", [5, 10])
    @pytest.mark.parametrize(
        "detection_metrics_matching_method",
        [
            "panoptic_segmentation",
            "point2tree",
            "for_instance",
            "for_ai_net",
            "for_ai_net_coverage",
            "segment_any_tree",
            "tree_learn",
        ],
    )
    @pytest.mark.parametrize(
        "segmentation_metrics_matching_method",
        [
            "panoptic_segmentation",
            "point2tree",
            "for_instance",
            "for_ai_net",
            "for_ai_net_coverage",
            "segment_any_tree",
            "tree_learn",
        ],
    )
    def test_evaluate_instance_segmentation(
        self, num_partitions: int, detection_metrics_matching_method: str, segmentation_metrics_matching_method: str
    ):
        xyz = np.array(
            [
                # tree 0
                [0, 0, 0],
                [1, 0, 1],
                [2, 0, 2],
                [3, 0, 3],
                [4, 0, 4],
                [5, 0, 5],
                [6, 0, 6],
                [7, 0, 7],
                [8, 0, 8],
                [9, 0, 9],
                [10, 0, 10],
                # tree 1
                [3, 3, 0],
                [4, 3, 1],
                [5, 3, 2],
                [6, 3, 3],
                [7, 3, 4],
                [8, 3, 5],
                [9, 3, 6],
                [10, 3, 7],
                [11, 3, 8],
                [12, 3, 9],
                [13, 3, 10],
            ],
            dtype=np.float64,
        )

        target = np.array([0] * 11 + [1] * 11, dtype=np.int64)
        prediction = np.array([1] * 11 + [0] * 11, dtype=np.int64)

        (
            metrics,
            metrics_per_instance,
            metrics_per_xy_partition,
            metrics_per_xy_partition_per_instance,
            metrics_per_z_partition,
            metrics_per_z_partition_per_instance,
        ) = evaluate_instance_segmentation(
            xyz,
            target,
            prediction,
            detection_metrics_matching_method=detection_metrics_matching_method,
            segmentation_metrics_matching_method=segmentation_metrics_matching_method,
            invalid_instance_id=-1,
            num_partitions=num_partitions,
        )

        assert metrics["DetectionTP"].iloc[0] == 2
        assert metrics["DetectionFP"].iloc[0] == 0
        assert metrics["DetectionFN"].iloc[0] == 0
        assert metrics["DetectionPrecision"].iloc[0] == 1.0
        assert metrics["DetectionRecall"].iloc[0] == 1.0
        assert metrics["DetectionF1Score"].iloc[0] == 1.0
        assert metrics["SegmentationMeanIoU"].iloc[0] == 1.0
        assert metrics["SegmentationMeanPrecision"].iloc[0] == 1.0
        assert metrics["SegmentationMeanRecall"].iloc[0] == 1.0

        assert len(metrics_per_instance) == 2
        assert (metrics_per_instance["IoU"] == 1.0).all()
        assert (metrics_per_instance["Precision"] == 1.0).all()
        assert (metrics_per_instance["Recall"] == 1.0).all()

        assert (metrics_per_instance["IoU"] == 1.0).all()
        assert (metrics_per_instance["Precision"] == 1.0).all()
        assert (metrics_per_instance["Recall"] == 1.0).all()

        assert len(metrics_per_xy_partition) == num_partitions
        assert (metrics_per_xy_partition["MeanIoU"] == 1.0).all()
        assert (metrics_per_xy_partition["MeanPrecision"] == 1.0).all()
        assert (metrics_per_xy_partition["MeanRecall"] == 1.0).all()

        assert len(metrics_per_xy_partition_per_instance) == num_partitions * 2
        assert (metrics_per_xy_partition_per_instance["IoU"] == 1.0).all()
        assert (metrics_per_xy_partition_per_instance["Precision"] == 1.0).all()
        assert (metrics_per_xy_partition_per_instance["Recall"] == 1.0).all()

        assert len(metrics_per_z_partition) == num_partitions
        assert (metrics_per_z_partition["MeanIoU"] == 1.0).all()
        assert (metrics_per_z_partition["MeanPrecision"] == 1.0).all()
        assert (metrics_per_z_partition["MeanRecall"] == 1.0).all()

        assert len(metrics_per_z_partition_per_instance) == num_partitions * 2
        assert (metrics_per_z_partition_per_instance["IoU"] == 1.0).all()
        assert (metrics_per_z_partition_per_instance["Precision"] == 1.0).all()
        assert (metrics_per_z_partition_per_instance["Recall"] == 1.0).all()

    def test_evaluate_instance_segmentation_without_partitions(self):
        xyz = np.array(
            [
                # tree 0
                [0, 0, 0],
                # tree 1
                [1, 1, 0],
                [1, 2, 0],
            ],
            dtype=np.float64,
        )

        target = np.array([0] * 1 + [1] * 2, dtype=np.int64)
        prediction = np.array([1] * 2 + [0] * 1, dtype=np.int64)

        (
            _,
            _,
            metrics_per_xy_partition,
            metrics_per_xy_partition_per_instance,
            metrics_per_z_partition,
            metrics_per_z_partition_per_instance,
        ) = evaluate_instance_segmentation(xyz, target, prediction, compute_partition_metrics=False)

        assert metrics_per_xy_partition is None
        assert metrics_per_xy_partition_per_instance is None
        assert metrics_per_z_partition is None
        assert metrics_per_z_partition_per_instance is None
