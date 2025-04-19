"""Tests for pointtree.evaluation.match_instances."""

import numpy as np
import pytest


from pointtree.evaluation import (
    match_instances,
    match_instances_tree_learn,
    match_instances_iou,
    match_instances_for_ai_net_coverage,
)


class TestMetrics:
    """Tests for pointtree.evaluation.match_instances."""

    @pytest.mark.parametrize(
        "method",
        ["panoptic_segmentation", "point2tree", "for_instance", "for_ai_net", "segment_any_tree", "tree_learn"],
    )
    def test_match_instances(self, method: str):
        target = np.array([1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 1, 3, 3, 3, 3], dtype=np.int64)
        prediction = np.array([2, 2, 2, 2, -1, 3, 1, 0, 0, 1, 2, -1, -1, -1, -1], dtype=np.int64)
        xyz = np.zeros((len(target), 3), dtype=np.float64)
        xyz[:, 2] = [0, 1, 10, 0, 1, 2, 5, 1, 2, 3, 4, 0, 0, 0, 15]

        if method == "point2tree":
            expected_matched_target_ids = np.array([0, 2, 1, -1], dtype=np.int64)
            expected_matched_predicted_ids = np.array([0, 2, 1, -1], dtype=np.int64)
        else:
            expected_matched_target_ids = np.array([0, -1, 1, -1], dtype=np.int64)
            expected_matched_predicted_ids = np.array([0, 2, -1, -1], dtype=np.int64)

        matched_target_ids, matched_predicted_ids = match_instances(xyz, target, prediction, method=method)

        np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids)
        np.testing.assert_array_equal(matched_predicted_ids, expected_matched_predicted_ids)

    @pytest.mark.parametrize(
        "method",
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
    def test_match_instances_all_correct(self, method: str):
        xyz = np.array(
            [
                [0, 0, 0],
                [0, 0, 20],
                [1, 1, 0],
                [1, 1, 30],
                [2, 2, 0],
                [2, 2, 10],
            ],
            dtype=np.float64,
        )
        target = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        prediction = np.array([1, 1, 0, 0, 2, 2], dtype=np.int64)

        matched_target_ids, matched_predicted_ids = match_instances(xyz, target, prediction, method=method)

        np.testing.assert_array_equal(np.array([1, 0, 2], dtype=np.int64), matched_target_ids)
        np.testing.assert_array_equal(np.array([1, 0, 2], dtype=np.int64), matched_predicted_ids)

    @pytest.mark.parametrize(
        "method",
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
    def test_match_instances_all_false_negatives(self, method: str):
        xyz = np.random.randn(6, 3)
        target = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        prediction = np.full(len(target), fill_value=-1, dtype=np.int64)

        matched_target_ids, matched_predicted_ids = match_instances(xyz, target, prediction, method=method)

        np.testing.assert_array_equal(np.array([], dtype=np.int64), matched_target_ids)
        np.testing.assert_array_equal(np.array([-1, -1, -1], dtype=np.int64), matched_predicted_ids)

    @pytest.mark.parametrize(
        "method",
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
    def test_match_instances_all_false_positives(self, method: str):
        xyz = np.random.randn(6, 3)
        prediction = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        target = np.full(len(prediction), fill_value=-1, dtype=np.int64)

        matched_target_ids, matched_predicted_ids = match_instances(xyz, target, prediction, method=method)

        np.testing.assert_array_equal(np.array([-1, -1, -1], dtype=np.int64), matched_target_ids)
        np.testing.assert_array_equal(np.array([], dtype=np.int64), matched_predicted_ids)

    @pytest.mark.parametrize(
        "method",
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
    def test_match_instances_non_continuous_target_ids(self, method: str):
        target = np.array([0, 2], dtype=np.int64)
        prediction = np.zeros_like(target)
        xyz = np.zeros((len(target), 3), dtype=np.float64)

        with pytest.raises(ValueError):
            match_instances(xyz, target, prediction, method=method)

    @pytest.mark.parametrize(
        "method",
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
    def test_match_instances_non_continuous_prediction_ids(self, method: str):
        prediction = np.array([0, 2], dtype=np.int64)
        target = np.zeros_like(prediction)
        xyz = np.zeros((len(target), 3), dtype=np.float64)

        with pytest.raises(ValueError):
            match_instances(xyz, target, prediction, method=method)

    def test_match_instances_invalid_method(self):
        xyz = np.random.randn(5, 3)
        target = np.zeros(5, dtype=np.int64)
        prediction = np.zeros(5, dtype=np.int64)

        with pytest.raises(ValueError):
            match_instances(xyz, target, prediction, method="test")

    @pytest.mark.parametrize("min_iou_treshold", [None, 0.2, 0.5])
    @pytest.mark.parametrize("accept_equal_iou", [True, False])
    @pytest.mark.parametrize("sort_by_target_height", [True, False])
    def test_match_instance_iou_sort_target_by_height(
        self, min_iou_treshold: float, accept_equal_iou: bool, sort_by_target_height: bool
    ):
        xyz = np.array(
            [
                [0, 0, 0],
                [0, 0, 10],
                [1, 1, 0],
                [1, 1, 1],
                [1, 1, 30],
                [2, 2, 0],
                [2, 2, 20],
            ],
            dtype=np.float64,
        )
        target = np.array([0, 0, 1, 1, 1, 2, 2], dtype=np.int64)
        prediction = np.array([0, 0, 0, 0, -1, 1, -1], dtype=np.int64)

        unique_target_ids = np.array([0, 1, 2], dtype=np.int64)
        unique_prediction_ids = np.array([0, 1], dtype=np.int64)

        matched_target_ids, matched_predicted_ids = match_instances_iou(
            xyz,
            target,
            unique_target_ids,
            prediction,
            unique_prediction_ids,
            min_iou_treshold=min_iou_treshold,
            accept_equal_iou=accept_equal_iou,
            sort_by_target_height=sort_by_target_height,
        )

        if min_iou_treshold is not None and min_iou_treshold >= 0.4:
            if accept_equal_iou:
                expected_matched_target_ids = np.array([0, 2], dtype=np.int64)
                expected_matched_predicted_ids = np.array([0, -1, 1], dtype=np.int64)
            else:
                expected_matched_target_ids = np.array([-1, -1], dtype=np.int64)
                expected_matched_predicted_ids = np.array([-1, -1, -1], dtype=np.int64)
        else:
            if sort_by_target_height:
                expected_matched_target_ids = np.array([1, 2], dtype=np.int64)
                expected_matched_predicted_ids = np.array([-1, 0, 1], dtype=np.int64)
            else:
                expected_matched_target_ids = np.array([0, 2], dtype=np.int64)
                expected_matched_predicted_ids = np.array([0, -1, 1], dtype=np.int64)

        np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids)
        np.testing.assert_array_equal(matched_predicted_ids, expected_matched_predicted_ids)

        if min_iou_treshold == 0.5 and accept_equal_iou and sort_by_target_height:
            matched_target_ids, matched_predicted_ids = match_instances(xyz, target, prediction, method="for_instance")

            np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids)
            np.testing.assert_array_equal(matched_predicted_ids, expected_matched_predicted_ids)

        if min_iou_treshold is None and sort_by_target_height:
            matched_target_ids, matched_predicted_ids = match_instances(xyz, target, prediction, method="point2tree")

            np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids)
            np.testing.assert_array_equal(matched_predicted_ids, expected_matched_predicted_ids)

        if min_iou_treshold == 0.5 and accept_equal_iou and not sort_by_target_height:
            matched_target_ids, matched_predicted_ids = match_instances(xyz, target, prediction, method="for_ai_net")

            np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids)
            np.testing.assert_array_equal(matched_predicted_ids, expected_matched_predicted_ids)

        if min_iou_treshold == 0.5 and not accept_equal_iou and not sort_by_target_height:
            matched_target_ids, matched_predicted_ids = match_instances(
                xyz, target, prediction, method="panoptic_segmentation"
            )

            np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids)
            np.testing.assert_array_equal(matched_predicted_ids, expected_matched_predicted_ids)

            matched_target_ids, matched_predicted_ids = match_instances(
                xyz, target, prediction, method="segment_any_tree"
            )

            np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids)
            np.testing.assert_array_equal(matched_predicted_ids, expected_matched_predicted_ids)

    @pytest.mark.parametrize("min_iou_treshold", [None, 0.2, 0.5])
    @pytest.mark.parametrize("accept_equal_iou", [True, False])
    def test_match_instances_tree_learn(self, min_iou_treshold: float, accept_equal_iou: bool):
        # test that Hungarian matching works as expected
        target = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2], dtype=np.int64)
        unique_target_ids = np.array([0, 1, 2], dtype=np.int64)
        prediction = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 2, -1], dtype=np.int64)
        unique_prediction_ids = np.array([0, 1, 2], dtype=np.int64)
        xyz = np.zeros((len(target), 3), dtype=np.float64)

        if min_iou_treshold is not None and min_iou_treshold == 0.5:
            if accept_equal_iou:
                expected_matched_target_ids = np.array([-1, -1, 2], dtype=np.int64)
                expected_matched_predicted_ids = np.array([-1, -1, 2], dtype=np.int64)
            else:
                expected_matched_target_ids = np.array([-1, -1, -1], dtype=np.int64)
                expected_matched_predicted_ids = np.array([-1, -1, -1], dtype=np.int64)
        else:
            expected_matched_target_ids = np.array([1, 0, 2], dtype=np.int64)
            expected_matched_predicted_ids = np.array([1, 0, 2], dtype=np.int64)

        matched_target_ids, matched_predicted_ids = match_instances_tree_learn(
            target,
            unique_target_ids,
            prediction,
            unique_prediction_ids,
            min_iou_treshold=min_iou_treshold,
            accept_equal_iou=accept_equal_iou,
        )

        np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids)
        np.testing.assert_array_equal(matched_predicted_ids, expected_matched_predicted_ids)

        if min_iou_treshold == 0.5 and not accept_equal_iou:
            matched_target_ids, matched_predicted_ids = match_instances(xyz, target, prediction, method="tree_learn")

            np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids)
            np.testing.assert_array_equal(matched_predicted_ids, expected_matched_predicted_ids)

    @pytest.mark.parametrize("min_iou_treshold", [None, 0.2, 0.4, 0.5])
    @pytest.mark.parametrize("accept_equal_iou", [True, False])
    def test_match_instances_for_ai_net_coverage(self, min_iou_treshold: float, accept_equal_iou: bool):
        target = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3], dtype=np.int64)
        prediction = np.array([0, 0, 0, 0, -1, 1, -1, -1, -1], dtype=np.int64)

        unique_target_id = np.array([0, 1, 2, 3], dtype=np.int64)
        unique_prediction_id = np.array([0, 1], dtype=np.int64)

        xyz = np.zeros((len(target), 3), dtype=np.float64)

        matched_target_ids, matched_predicted_ids = match_instances_for_ai_net_coverage(
            target,
            unique_target_id,
            prediction,
            unique_prediction_id,
            min_iou_treshold=min_iou_treshold,
            accept_equal_iou=accept_equal_iou,
        )

        expected_matched_target_ids = np.array([], dtype=np.int64)
        expected_matched_predicted_ids = np.array([], dtype=np.int64)
        if min_iou_treshold is None or min_iou_treshold == 0.2 or (min_iou_treshold == 0.4 and accept_equal_iou):
            expected_matched_target_ids = np.array([0, 2], dtype=np.int64)
            expected_matched_predicted_ids = np.array([0, 0, 1, -1], dtype=np.int64)
        elif (min_iou_treshold == 0.4 and not accept_equal_iou) or (min_iou_treshold == 0.5 and accept_equal_iou):
            expected_matched_target_ids = np.array([0, 2], dtype=np.int64)
            expected_matched_predicted_ids = np.array([0, -1, 1, -1], dtype=np.int64)
        elif min_iou_treshold == 0.5 and not accept_equal_iou:
            expected_matched_target_ids = np.array([-1, -1], dtype=np.int64)
            expected_matched_predicted_ids = np.array([-1, -1, -1, -1], dtype=np.int64)

        np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids)
        np.testing.assert_array_equal(matched_predicted_ids, expected_matched_predicted_ids)

        if min_iou_treshold is None:
            matched_target_ids, matched_predicted_ids = match_instances(
                xyz, target, prediction, method="for_ai_net_coverage"
            )

            np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids)
            np.testing.assert_array_equal(matched_predicted_ids, expected_matched_predicted_ids)
