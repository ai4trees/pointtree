"""Tests for pointtree.evaluation.instance_segmentation_metrics."""  # pylint: disable = too-many-lines

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
    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_instance_detection_metrics(self, min_precision_fp: float, invalid_instance_id: int):
        start_instance_id = invalid_instance_id + 1
        target = np.array([1, 1, 1, 0, 0, 2, -1], dtype=np.int64) + start_instance_id
        prediction = np.array([0, -1, 2, 2, 2, 1, 1], dtype=np.int64) + start_instance_id
        xyz = np.random.randn(len(target), 3)

        matched_predicted_ids = np.array([2, -1, -1], dtype=np.int64) + start_instance_id
        matched_target_ids = np.array([-1, -1, 0], dtype=np.int64) + start_instance_id

        metrics = instance_detection_metrics(
            xyz,
            target,
            prediction,
            matched_predicted_ids,
            matched_target_ids,
            invalid_instance_id=invalid_instance_id,
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

    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_instance_detection_metrics_labeled_mask(self, invalid_instance_id: int):
        start_instance_id = invalid_instance_id + 1
        target = np.array([1, 1, 1, 0, 0, 2, -1, -1, -1, -1, -1], dtype=np.int64) + start_instance_id
        prediction = np.array([0, -1, 2, 2, 2, 1, 1, -1, 3, 3, 3], dtype=np.int64) + start_instance_id
        labeled_mask = np.array([True] * 7 + [False] * 4, dtype=bool)
        xyz = np.random.randn(len(target), 3)

        matched_predicted_ids = np.array([2, -1, -1], dtype=np.int64) + start_instance_id
        matched_target_ids = np.array([-1, -1, 0, -1], dtype=np.int64) + start_instance_id

        metrics = instance_detection_metrics(
            xyz,
            target,
            prediction,
            matched_predicted_ids,
            matched_target_ids,
            invalid_instance_id=invalid_instance_id,
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
    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_instance_detection_metrics_all_correct(self, min_precision_fp: float, invalid_instance_id: int):
        start_instance_id = invalid_instance_id + 1
        target = np.array([1, 1, 1, 0, 0, 2, -1, 2], dtype=np.int64) + start_instance_id
        prediction = np.array([0, 0, 0, 2, 2, 1, -1, 1], dtype=np.int64) + start_instance_id
        xyz = np.random.randn(len(target), 3)

        matched_predicted_ids = np.array([2, 0, 1], dtype=np.int64) + start_instance_id
        matched_target_ids = np.array([1, 2, 0], dtype=np.int64) + start_instance_id

        metrics = instance_detection_metrics(
            xyz,
            target,
            prediction,
            matched_predicted_ids,
            matched_target_ids,
            invalid_instance_id=invalid_instance_id,
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

    @pytest.mark.parametrize("min_tree_height_fp", [0.0, 2.0])
    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_instance_detection_metrics_min_tree_height_fp(self, min_tree_height_fp: float, invalid_instance_id: int):
        start_instance_id = invalid_instance_id + 1
        target = np.array([0, 0, 0, -1, -1, -1], dtype=np.int64) + start_instance_id
        prediction = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64) + start_instance_id
        xyz = np.array(
            [
                # tree 1
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 4.5],
                # tree 2
                [1, 1, 0],
                [1, 1, 1],
                [1, 1, 1.5],
            ]
        )

        matched_predicted_ids = np.array([0], dtype=np.int64) + start_instance_id
        matched_target_ids = np.array([0, -1], dtype=np.int64) + start_instance_id

        metrics = instance_detection_metrics(
            xyz,
            target,
            prediction,
            matched_predicted_ids,
            matched_target_ids,
            invalid_instance_id=invalid_instance_id,
            min_tree_height_fp=min_tree_height_fp,
        )

        assert metrics["TP"] == 1
        if min_tree_height_fp > 1.5:
            assert metrics["FP"] == 0
        else:
            assert metrics["FP"] == 1

    def test_instance_detection_metrics_invalid_prediction(self):
        target = np.zeros(5, dtype=np.int64)
        prediction = np.zeros(4, dtype=np.int64)
        xyz = np.random.randn(len(target), 3)

        matched_predicted_ids = np.array([0], dtype=np.int64)
        matched_target_ids = np.array([0], dtype=np.int64)

        with pytest.raises(ValueError):
            instance_detection_metrics(
                xyz,
                target,
                prediction,
                matched_predicted_ids,
                matched_target_ids,
                invalid_instance_id=-1,
            )

    def test_instance_detection_metrics_invalid_matched_predicted_ids(self):
        target = np.zeros(5, dtype=np.int64)
        prediction = np.zeros(5, dtype=np.int64)
        xyz = np.random.randn(len(target), 3)

        matched_predicted_ids = np.array([0, 1], dtype=np.int64)
        matched_target_ids = np.array([0], dtype=np.int64)

        with pytest.raises(ValueError):
            instance_detection_metrics(
                xyz,
                target,
                prediction,
                matched_predicted_ids,
                matched_target_ids,
                invalid_instance_id=-1,
            )

    def test_instance_detection_metrics_invalid_matched_target_ids(self):
        target = np.zeros(5, dtype=np.int64)
        prediction = np.zeros(5, dtype=np.int64)
        xyz = np.random.randn(len(target), 3)

        matched_predicted_ids = np.array([0], dtype=np.int64)
        matched_target_ids = np.array([0, 1], dtype=np.int64)

        with pytest.raises(ValueError):
            instance_detection_metrics(
                xyz,
                target,
                prediction,
                matched_predicted_ids,
                matched_target_ids,
                invalid_instance_id=-1,
            )

    def test_instance_detection_metrics_different_start_ids(self):
        target = np.zeros(5, dtype=np.int64)
        prediction = np.ones(5, dtype=np.int64)
        xyz = np.random.randn(len(target), 3)

        matched_predicted_ids = np.array([1], dtype=np.int64)
        matched_target_ids = np.array([0], dtype=np.int64)

        with pytest.raises(ValueError):
            instance_detection_metrics(
                xyz,
                target,
                prediction,
                matched_predicted_ids,
                matched_target_ids,
                invalid_instance_id=-1,
            )

    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_instance_detection_metrics_all_false_negatives(self, invalid_instance_id: int):
        start_instance_id = invalid_instance_id + 1
        target = np.array([1, 1, 1, 0, 0], dtype=np.int64) + start_instance_id
        prediction = np.array([-1, -1, -1, -1, -1], dtype=np.int64) + start_instance_id
        xyz = np.random.randn(len(target), 3)

        matched_predicted_ids = np.array([-1, -1], dtype=np.int64) + start_instance_id
        matched_target_ids = np.array([], dtype=np.int64) + start_instance_id

        metrics = instance_detection_metrics(
            xyz,
            target,
            prediction,
            matched_predicted_ids,
            matched_target_ids,
            invalid_instance_id=invalid_instance_id,
        )

        assert metrics["TP"] == 0
        assert metrics["FP"] == 0
        assert metrics["FN"] == 2
        assert np.isnan(metrics["Precision"])
        assert np.isnan(metrics["CommissionError"])
        assert metrics["Recall"] == 0
        assert metrics["OmissionError"] == 1
        assert metrics["F1Score"] == 0

    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_instance_detection_metrics_all_false_positives(self, invalid_instance_id: int):
        start_instance_id = invalid_instance_id + 1
        target = np.array([-1, -1, -1, -1, -1], dtype=np.int64) + start_instance_id
        prediction = np.array([1, 1, 1, 0, 0], dtype=np.int64) + start_instance_id
        xyz = np.random.randn(len(target), 3)

        matched_predicted_ids = np.array([], dtype=np.int64)
        matched_target_ids = np.full((2,), fill_value=invalid_instance_id, dtype=np.int64)

        metrics = instance_detection_metrics(
            xyz,
            target,
            prediction,
            matched_predicted_ids,
            matched_target_ids,
            invalid_instance_id=invalid_instance_id,
            min_precision_fp=0,
        )

        assert metrics["TP"] == 0
        assert metrics["FP"] == 2
        assert metrics["FN"] == 0
        assert metrics["Precision"] == 0
        assert metrics["CommissionError"] == 1
        assert np.isnan(metrics["Recall"])
        assert np.isnan(metrics["OmissionError"])
        assert metrics["F1Score"] == 0

    @pytest.mark.parametrize("include_unmatched_instances", [True, False])
    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_instance_segmentation_metrics(self, include_unmatched_instances: bool, invalid_instance_id: int):
        start_instance_id = invalid_instance_id + 1
        target = np.array([1, 1, 1, 1, 0, 0, 0, 0, -1, 2], dtype=np.int64) + start_instance_id
        prediction = np.array([0, 1, 0, 0, 0, 2, 2, 2, -1, -1], dtype=np.int64) + start_instance_id

        matched_predicted_ids = np.array([2, 0, -1], dtype=np.int64) + start_instance_id

        metrics, per_instance_metrics = instance_segmentation_metrics(
            target,
            prediction,
            matched_predicted_ids,
            include_unmatched_instances=include_unmatched_instances,
            invalid_instance_id=invalid_instance_id,
        )

        if include_unmatched_instances:
            assert metrics["MeanIoU"] == (3 / 5 + 3 / 4) / 3
            assert metrics["MeanRecall"] == (3 / 4 + 3 / 4) / 3
        else:
            assert metrics["MeanIoU"] == (3 / 5 + 3 / 4) / 2
            assert metrics["MeanRecall"] == (3 / 4 + 3 / 4) / 2
        assert metrics["MeanPrecision"] == (3 / 4 + 3 / 3) / 2

        if include_unmatched_instances:
            assert len(per_instance_metrics) == 3

            target_2 = 2 + start_instance_id
            pred_2 = invalid_instance_id
            assert (
                per_instance_metrics.loc[per_instance_metrics["TargetID"] == target_2, "PredictionID"].iloc[0] == pred_2
            )
            assert per_instance_metrics.loc[per_instance_metrics["TargetID"] == target_2, "IoU"].iloc[0] == 0
            assert per_instance_metrics.loc[per_instance_metrics["TargetID"] == target_2, "Recall"].iloc[0] == 0
            assert np.isnan(per_instance_metrics.loc[per_instance_metrics["TargetID"] == target_2, "Precision"].iloc[0])
        else:
            assert len(per_instance_metrics) == 2

        target_0 = start_instance_id
        pred_0 = 2 + start_instance_id
        assert per_instance_metrics.loc[per_instance_metrics["TargetID"] == target_0, "PredictionID"].iloc[0] == pred_0
        assert per_instance_metrics.loc[per_instance_metrics["TargetID"] == target_0, "IoU"].iloc[0] == 3 / 4
        assert per_instance_metrics.loc[per_instance_metrics["TargetID"] == target_0, "Recall"].iloc[0] == 3 / 4
        assert per_instance_metrics.loc[per_instance_metrics["TargetID"] == target_0, "Precision"].iloc[0] == 3 / 3

        target_1 = 1 + start_instance_id
        pred_1 = start_instance_id
        assert per_instance_metrics.loc[per_instance_metrics["TargetID"] == target_1, "PredictionID"].iloc[0] == pred_1
        assert per_instance_metrics.loc[per_instance_metrics["TargetID"] == target_1, "Recall"].iloc[0] == 3 / 4
        assert per_instance_metrics.loc[per_instance_metrics["TargetID"] == target_1, "Precision"].iloc[0] == 3 / 4

    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_instance_segmentation_metrics_all_correct(self, invalid_instance_id: int):
        start_instance_id = invalid_instance_id + 1
        target = np.array([1, 1, 1, 0, 0, 2, -1, 2], dtype=np.int64) + start_instance_id
        prediction = np.array([0, 0, 0, 2, 2, 1, -1, 1], dtype=np.int64) + start_instance_id

        matched_predicted_ids = np.array([2, 0, 1], dtype=np.int64) + start_instance_id

        metrics, per_instance_metrics = instance_segmentation_metrics(
            target, prediction, matched_predicted_ids, invalid_instance_id=invalid_instance_id
        )

        assert metrics["MeanIoU"] == 1
        assert metrics["MeanRecall"] == 1
        assert metrics["MeanPrecision"] == 1

        assert len(per_instance_metrics) == 3

        target_0 = start_instance_id
        pred_0 = 2 + start_instance_id
        assert per_instance_metrics.loc[per_instance_metrics["TargetID"] == target_0, "PredictionID"].iloc[0] == pred_0

        target_1 = 1 + start_instance_id
        pred_1 = start_instance_id
        assert per_instance_metrics.loc[per_instance_metrics["TargetID"] == target_1, "PredictionID"].iloc[0] == pred_1

        target_2 = 2 + start_instance_id
        pred_2 = 1 + start_instance_id
        assert per_instance_metrics.loc[per_instance_metrics["TargetID"] == target_2, "PredictionID"].iloc[0] == pred_2

        assert (per_instance_metrics["IoU"] == 1).all()
        assert (per_instance_metrics["Precision"] == 1).all()
        assert (per_instance_metrics["Recall"] == 1).all()

    @pytest.mark.parametrize("include_unmatched_instances", [True, False])
    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_instance_segmentation_metrics_no_matches(
        self, include_unmatched_instances: bool, invalid_instance_id: int
    ):
        start_instance_id = invalid_instance_id + 1
        target = np.array([1, 1, -1, 0, 0], dtype=np.int64) + start_instance_id
        prediction = np.full((5,), fill_value=invalid_instance_id, dtype=np.int64)

        matched_predicted_ids = np.full((2,), fill_value=invalid_instance_id, dtype=np.int64)

        metrics, per_instance_metrics = instance_segmentation_metrics(
            target,
            prediction,
            matched_predicted_ids,
            include_unmatched_instances=include_unmatched_instances,
            invalid_instance_id=invalid_instance_id,
        )

        if include_unmatched_instances:
            assert metrics["MeanIoU"] == 0
            assert metrics["MeanRecall"] == 0
            assert np.isnan(metrics["MeanPrecision"])
            np.testing.assert_array_equal(np.zeros(2, dtype=np.float64), per_instance_metrics["IoU"].to_numpy())
            np.testing.assert_array_equal(np.zeros(2, dtype=np.float64), per_instance_metrics["Recall"].to_numpy())
            assert per_instance_metrics["Precision"].isna().all()
        else:
            assert np.isnan(metrics["MeanIoU"])
            assert np.isnan(metrics["MeanRecall"])
            assert np.isnan(metrics["MeanPrecision"])
            assert len(per_instance_metrics) == 0
            assert "TargetID" in per_instance_metrics
            assert "PredictionID" in per_instance_metrics

    @pytest.mark.parametrize("include_unmatched_instances", [True, False])
    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_instance_segmentation_metrics_include_unmatched_instances(
        self, include_unmatched_instances: bool, invalid_instance_id: int
    ):
        start_instance_id = invalid_instance_id + 1
        target = np.array([1, 1, 1, 0, 0, 2, -1, 2], dtype=np.int64) + start_instance_id
        prediction = np.array([0, 0, 0, 2, 2, -1, -1, -1], dtype=np.int64) + start_instance_id

        matched_predicted_ids = np.array([2, 0, -1], dtype=np.int64) + start_instance_id

        metrics, per_instance_metrics = instance_segmentation_metrics(
            target,
            prediction,
            matched_predicted_ids,
            include_unmatched_instances=include_unmatched_instances,
            invalid_instance_id=invalid_instance_id,
        )

        target_2 = 2 + start_instance_id
        target_2_metrics = per_instance_metrics.loc[per_instance_metrics["TargetID"] == target_2]

        if include_unmatched_instances:
            assert metrics["MeanIoU"] == 2 / 3
            assert metrics["MeanRecall"] == 2 / 3
            assert metrics["MeanPrecision"] == 1

            assert len(per_instance_metrics) == 3

            assert len(target_2_metrics) == 1
            assert target_2_metrics["PredictionID"].iloc[0] == invalid_instance_id

            assert target_2_metrics["IoU"].iloc[0] == 0
            assert np.isnan(target_2_metrics["Precision"].iloc[0])
            assert target_2_metrics["Recall"].iloc[0] == 0
        else:
            assert metrics["MeanIoU"] == 1
            assert metrics["MeanRecall"] == 1
            assert metrics["MeanPrecision"] == 1

            assert len(per_instance_metrics) == 2
            assert len(target_2_metrics) == 0

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

    def test_instance_segmentation_metrics_different_start_ids(self):
        target = np.zeros(5, dtype=np.int64)
        prediction = np.ones(5, dtype=np.int64)

        matched_predicted_ids = np.array([1], dtype=np.int64)

        with pytest.raises(ValueError):
            instance_segmentation_metrics(target, prediction, matched_predicted_ids, invalid_instance_id=-1)

    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_instance_segmentation_metrics_per_xy_partition(self, invalid_instance_id: int):
        start_instance_id = invalid_instance_id + 1

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

        target = np.array([0] * 13 + [1] * 11, dtype=np.int64) + start_instance_id
        prediction = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] + [0] * 11, dtype=np.int64) + start_instance_id

        matched_predicted_ids = np.array([1, 0], dtype=np.int64) + start_instance_id

        metrics, inst_metrics = instance_segmentation_metrics_per_partition(
            xyz, target, prediction, matched_predicted_ids, partition="xy", invalid_instance_id=invalid_instance_id
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

        if invalid_instance_id == -1:
            target_0_mask = inst_metrics["TargetID"] == 0
        else:
            target_0_mask = inst_metrics["TargetID"] == 1

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

        target_1_mask = inst_metrics["TargetID"] == 1 + start_instance_id
        correct_part = [0, 1, 5, 6, 7, 8, 9]

        assert (inst_metrics.loc[target_1_mask & inst_metrics["Partition"].isin(correct_part), "IoU"] == 1).all()
        assert (inst_metrics.loc[target_1_mask & inst_metrics["Partition"].isin(correct_part), "Precision"] == 1).all()
        assert (inst_metrics.loc[target_1_mask & inst_metrics["Partition"].isin(correct_part), "Recall"] == 1).all()

        assert inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 2), "IoU"].iloc[0] == 1 / 3
        assert inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 2), "Precision"].iloc[0] == 1 / 3
        assert inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 2), "Recall"].iloc[0] == 1

    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_instance_segmentation_metrics_per_xy_partition_all_correct(self, invalid_instance_id: int):
        start_instance_id = invalid_instance_id + 1

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

        target = np.array([0] * 13 + [1] * 13, dtype=np.int64) + start_instance_id
        prediction = np.array([1] * 13 + [0] * 13, dtype=np.int64) + start_instance_id

        matched_predicted_ids = np.array([1, 0], dtype=np.int64) + start_instance_id

        metrics, metrics_per_instance = instance_segmentation_metrics_per_partition(
            xyz, target, prediction, matched_predicted_ids, partition="xy", invalid_instance_id=invalid_instance_id
        )

        assert len(metrics) == 10
        assert (metrics["MeanIoU"] == 1).all()
        assert (metrics["MeanPrecision"] == 1).all()
        assert (metrics["MeanRecall"] == 1).all()

        assert len(metrics_per_instance) == 20
        assert (metrics_per_instance["IoU"] == 1).all()
        assert (metrics_per_instance["Precision"] == 1).all()
        assert (metrics_per_instance["Recall"] == 1).all()

    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_instance_segmentation_metrics_per_z_partition(self, invalid_instance_id: int):
        start_instance_id = invalid_instance_id + 1

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

        target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1] * 9 + [2] * 2, dtype=np.int64) + start_instance_id
        prediction = (
            np.array([1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] + [0] * 9 + [-1] * 2, dtype=np.int64) + start_instance_id
        )

        matched_predicted_ids = np.array([1, 0, -1], dtype=np.int64) + start_instance_id

        metrics, inst_metrics = instance_segmentation_metrics_per_partition(
            xyz,
            target,
            prediction,
            matched_predicted_ids,
            partition="z",
            include_unmatched_instances=False,
            invalid_instance_id=invalid_instance_id,
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

        target_0_mask = inst_metrics["TargetID"] == start_instance_id

        assert inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 1), "IoU"].iloc[0] == 1 / 3
        assert inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 1), "Precision"].iloc[0] == 1
        assert inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 1), "Recall"].iloc[0] == 1 / 3

        assert np.isnan(inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 6), "IoU"].iloc[0])
        assert np.isnan(inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 6), "Precision"].iloc[0])
        assert np.isnan(inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 6), "Recall"].iloc[0])

        assert np.isnan(inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 8), "IoU"].iloc[0])
        assert np.isnan(inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 8), "Precision"].iloc[0])
        assert np.isnan(inst_metrics.loc[target_0_mask & (inst_metrics["Partition"] == 8), "Recall"].iloc[0])

        target_1_mask = inst_metrics["TargetID"] == 1 + start_instance_id

        assert inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 1), "IoU"].iloc[0] == 1 / 3
        assert inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 1), "Precision"].iloc[0] == 1 / 3
        assert inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 1), "Recall"].iloc[0] == 1

        assert np.isnan(inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 6), "IoU"].iloc[0])
        assert np.isnan(inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 6), "Precision"].iloc[0])
        assert np.isnan(inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 6), "Recall"].iloc[0])

        assert np.isnan(inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 9), "IoU"].iloc[0])
        assert np.isnan(inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 9), "Precision"].iloc[0])
        assert np.isnan(inst_metrics.loc[target_1_mask & (inst_metrics["Partition"] == 9), "Recall"].iloc[0])

    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_instance_segmentation_metrics_per_z_partition_all_correct(self, invalid_instance_id: int):
        start_instance_id = invalid_instance_id + 1

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

        target = np.array([0] * 11 + [1] * 11, dtype=np.int64) + start_instance_id
        prediction = np.array([1] * 11 + [0] * 11, dtype=np.int64) + start_instance_id

        matched_predicted_ids = np.array([1, 0], dtype=np.int64) + start_instance_id

        metrics, metrics_per_instance = instance_segmentation_metrics_per_partition(
            xyz, target, prediction, matched_predicted_ids, partition="z", invalid_instance_id=invalid_instance_id
        )

        assert len(metrics) == 10
        assert (metrics["MeanIoU"] == 1).all()
        assert (metrics["MeanPrecision"] == 1).all()
        assert (metrics["MeanRecall"] == 1).all()

        assert len(metrics_per_instance) == 20
        assert (metrics_per_instance["IoU"] == 1).all()
        assert (metrics_per_instance["Precision"] == 1).all()
        assert (metrics_per_instance["Recall"] == 1).all()

    @pytest.mark.parametrize("partition", ["xy", "z"])
    @pytest.mark.parametrize("include_unmatched_instances", [True, False])
    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_instance_segmentation_metric_per_partition_no_matches(
        self, partition: str, include_unmatched_instances: bool, invalid_instance_id: int
    ):
        start_instance_id = invalid_instance_id + 1
        target = np.array([1, 1, -1, 0, 0], dtype=np.int64) + start_instance_id
        prediction = np.full((5,), fill_value=invalid_instance_id, dtype=np.int64)
        xyz = np.random.randn(len(target), 3)

        matched_predicted_ids = np.full((2,), fill_value=invalid_instance_id, dtype=np.int64)

        metrics, per_instance_metrics = instance_segmentation_metrics_per_partition(
            xyz,
            target,
            prediction,
            matched_predicted_ids,
            partition=partition,
            include_unmatched_instances=include_unmatched_instances,
            invalid_instance_id=invalid_instance_id,
        )

        if include_unmatched_instances:
            assert ~np.isnan(metrics["MeanIoU"]).all()
            assert ~np.isnan(metrics["MeanRecall"]).all()
            assert np.isnan(metrics["MeanPrecision"]).all()
            assert len(per_instance_metrics) == 20
        else:
            assert np.isnan(metrics["MeanIoU"]).all()
            assert np.isnan(metrics["MeanRecall"]).all()
            assert np.isnan(metrics["MeanPrecision"]).all()
            assert len(per_instance_metrics) == 0
            assert "TargetID" in per_instance_metrics
            assert "PredictionID" in per_instance_metrics
            assert "Partition" in per_instance_metrics

    @pytest.mark.parametrize("include_unmatched_instances", [True, False])
    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_instance_segmentation_metrics_per_z_partition_include_unmatched_instances(
        self, include_unmatched_instances: bool, invalid_instance_id: int
    ):
        start_instance_id = invalid_instance_id + 1

        xyz = np.array(
            [
                # tree 0
                [0, 0, 2],
                [0, 0, 3],
                [0, 0, 4],
                # tree 1
                [3, 3, 2],
                [3, 3, 3],
                [3, 3, 4],
            ],
            dtype=np.float64,
        )

        target = np.array([0] * 3 + [1] * 3, dtype=np.int64) + start_instance_id
        prediction = np.array([0] * 3 + [-1] * 3, dtype=np.int64) + start_instance_id

        matched_predicted_ids = np.array([0, -1], dtype=np.int64) + start_instance_id

        metrics, metrics_per_instance = instance_segmentation_metrics_per_partition(
            xyz,
            target,
            prediction,
            matched_predicted_ids,
            partition="z",
            include_unmatched_instances=include_unmatched_instances,
            invalid_instance_id=invalid_instance_id,
            num_partitions=4,
        )

        if include_unmatched_instances:
            np.testing.assert_array_equal(
                metrics["MeanIoU"].to_numpy(), np.array([0.5, np.nan, 0.5, np.nan], dtype=np.float64)
            )
            np.testing.assert_array_equal(
                metrics["MeanPrecision"].to_numpy(), np.array([1, np.nan, 1, np.nan], dtype=np.float64)
            )
            np.testing.assert_array_equal(
                metrics["MeanRecall"].to_numpy(), np.array([0.5, np.nan, 0.5, np.nan], dtype=np.float64)
            )

            np.testing.assert_array_equal(
                metrics_per_instance["IoU"].to_numpy(),
                np.array([1, np.nan, 1, np.nan, 0, np.nan, 0, np.nan], dtype=np.float64),
            )
            np.testing.assert_array_equal(
                metrics_per_instance["Precision"].to_numpy(),
                np.array([1, np.nan, 1, np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64),
            )
            np.testing.assert_array_equal(
                metrics_per_instance["Recall"].to_numpy(),
                np.array([1, np.nan, 1, np.nan, 0, np.nan, 0, np.nan], dtype=np.float64),
            )
        else:
            assert len(metrics) == 4
            np.testing.assert_array_equal(
                metrics["MeanIoU"].to_numpy(), np.array([1, np.nan, 1, np.nan], dtype=np.float64)
            )
            np.testing.assert_array_equal(
                metrics["MeanPrecision"].to_numpy(), np.array([1, np.nan, 1, np.nan], dtype=np.float64)
            )
            np.testing.assert_array_equal(
                metrics["MeanRecall"].to_numpy(), np.array([1, np.nan, 1, np.nan], dtype=np.float64)
            )

            np.testing.assert_array_equal(
                metrics_per_instance["IoU"].to_numpy(), np.array([1, np.nan, 1, np.nan], dtype=np.float64)
            )
            np.testing.assert_array_equal(
                metrics_per_instance["Precision"].to_numpy(), np.array([1, np.nan, 1, np.nan], dtype=np.float64)
            )
            np.testing.assert_array_equal(
                metrics_per_instance["Recall"].to_numpy(), np.array([1, np.nan, 1, np.nan], dtype=np.float64)
            )

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

    @pytest.mark.parametrize("partition", ["xy", "z"])
    def test_instance_segmentation_metrics_per_partition_different_start_ids(self, partition: str):
        xyz = np.random.randn(5, 3)
        target = np.zeros(5, dtype=np.int64)
        prediction = np.ones(5, dtype=np.int64)

        matched_predicted_ids = np.array([1], dtype=np.int64)

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
    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_evaluate_instance_segmentation(
        self,
        num_partitions: int,
        detection_metrics_matching_method: str,
        segmentation_metrics_matching_method: str,
        invalid_instance_id: int,
    ):
        start_instance_id = invalid_instance_id + 1

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

        target = np.array([0] * 11 + [1] * 11, dtype=np.int64) + start_instance_id
        prediction = np.array([1] * 11 + [0] * 11, dtype=np.int64) + start_instance_id

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
            invalid_instance_id=invalid_instance_id,
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
