"""Tests for pointtree.evaluation.match_instances."""

import numpy as np
import pytest


from pointtree.evaluation import match_instances


class TestMetrics:
    """Tests for pointtree.evaluation.match_instances."""

    @pytest.mark.parametrize(
        "method",
        ["panoptic_segmentation", "point2tree", "for_instance", "for_ai_net", "tree_learn"],
    )
    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_match_instances(self, method: str, invalid_instance_id: int):
        start_instance_id = invalid_instance_id + 1
        target = np.array([1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 1, 3, 3, 3, 3, -1], dtype=np.int64)
        prediction = np.array([2, 2, 2, 2, -1, 3, 1, 0, 0, 1, 2, -1, -1, -1, -1, -1], dtype=np.int64)

        target += start_instance_id
        prediction += start_instance_id

        xyz = np.zeros((len(target), 3), dtype=np.float64)
        xyz[:, 2] = [0, 1, 10, 0, 1, 2, 5, 1, 2, 3, 4, 0, 0, 0, 15, 0]

        if method == "point2tree":
            expected_matched_target_ids = np.array([0, -1, 1, 2], dtype=np.int64)
            expected_matched_predicted_ids = np.array([0, 2, 3, -1], dtype=np.int64)
        else:
            expected_matched_target_ids = np.array([0, -1, 1, -1], dtype=np.int64)
            expected_matched_predicted_ids = np.array([0, 2, -1, -1], dtype=np.int64)

        expected_matched_target_ids += start_instance_id
        expected_matched_predicted_ids += start_instance_id

        matched_target_ids, matched_predicted_ids, _ = match_instances(
            target, prediction, xyz=xyz, method=method, invalid_instance_id=invalid_instance_id
        )

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
            "tree_learn",
        ],
    )
    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_match_instances_all_correct(self, method: str, invalid_instance_id):
        start_instance_id = invalid_instance_id + 1
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
        target = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64) + start_instance_id
        prediction = np.array([1, 1, 0, 0, 2, 2], dtype=np.int64) + start_instance_id

        matched_target_ids, matched_predicted_ids, _ = match_instances(
            target, prediction, xyz=xyz, method=method, invalid_instance_id=invalid_instance_id
        )

        expected_matched_target_ids = np.array([1, 0, 2], dtype=np.int64) + start_instance_id
        expected_matched_predicted_ids = np.array([1, 0, 2], dtype=np.int64) + start_instance_id

        np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids)
        np.testing.assert_array_equal(expected_matched_predicted_ids, matched_predicted_ids)

    @pytest.mark.parametrize(
        "method",
        [
            "panoptic_segmentation",
            "point2tree",
            "for_instance",
            "for_ai_net",
            "for_ai_net_coverage",
            "tree_learn",
        ],
    )
    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_match_instances_all_false_negatives(self, method: str, invalid_instance_id: int):
        start_instance_id = invalid_instance_id + 1
        xyz = np.random.randn(6, 3)
        target = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64) + start_instance_id
        prediction = np.full(len(target), fill_value=invalid_instance_id, dtype=np.int64)

        matched_target_ids, matched_predicted_ids, _ = match_instances(
            target, prediction, xyz=xyz, method=method, invalid_instance_id=invalid_instance_id
        )

        np.testing.assert_array_equal(np.array([], dtype=np.int64), matched_target_ids)
        np.testing.assert_array_equal(
            np.full((3,), fill_value=invalid_instance_id, dtype=np.int64), matched_predicted_ids
        )

    @pytest.mark.parametrize(
        "method",
        [
            "panoptic_segmentation",
            "point2tree",
            "for_instance",
            "for_ai_net",
            "for_ai_net_coverage",
            "tree_learn",
        ],
    )
    @pytest.mark.parametrize("invalid_instance_id", [-1, 0])
    def test_match_instances_all_false_positives(self, method: str, invalid_instance_id: int):
        start_instance_id = invalid_instance_id + 1
        xyz = np.random.randn(6, 3)
        prediction = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64) + start_instance_id
        target = np.full(len(prediction), fill_value=-1, dtype=np.int64) + start_instance_id

        matched_target_ids, matched_predicted_ids, _ = match_instances(
            target, prediction, xyz=xyz, method=method, invalid_instance_id=invalid_instance_id
        )

        np.testing.assert_array_equal(np.full((3,), fill_value=invalid_instance_id, dtype=np.int64), matched_target_ids)
        np.testing.assert_array_equal(np.array([], dtype=np.int64), matched_predicted_ids)

    @pytest.mark.parametrize("pass_labeled_mask", [True, False])
    @pytest.mark.parametrize("invalid_instance_id, uncertain_instance_id", [(-1, -2), (0, -1)])
    def test_match_instances_labeled_mask(
        self, pass_labeled_mask: bool, invalid_instance_id: int, uncertain_instance_id: int
    ):
        start_instance_id = invalid_instance_id + 1
        target = np.array([1, 1, 1, 0, 0, 2, -1, -1, -1, -1, -1, -1], dtype=np.int64) + start_instance_id
        prediction = np.array([0, -1, 2, 2, 2, 1, 1, 1, -1, 3, 3, 3], dtype=np.int64) + start_instance_id
        labeled_mask = np.array([True] * 8 + [False] * 4, dtype=bool) if pass_labeled_mask else None
        xyz = np.random.randn(len(target), 3)

        matched_target_ids, matched_predicted_ids, _ = match_instances(
            target,
            prediction,
            xyz=xyz,
            method="panoptic_segmentation",
            invalid_instance_id=invalid_instance_id,
            uncertain_instance_id=uncertain_instance_id,
            min_precision_fp=0.5,
            labeled_mask=labeled_mask,
        )

        if labeled_mask is not None:
            expected_matched_predicted_ids = np.array([2, -1, -1], dtype=np.int64) + start_instance_id
            expected_matched_target_ids = np.array([-1, -1, 0, -2], dtype=np.int64) + start_instance_id
        else:
            expected_matched_predicted_ids = np.array([2, -1, -1], dtype=np.int64) + start_instance_id
            expected_matched_target_ids = np.array([-1, -2, 0, -2], dtype=np.int64) + start_instance_id

        np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids)
        np.testing.assert_array_equal(expected_matched_predicted_ids, matched_predicted_ids)

    @pytest.mark.parametrize("min_tree_height_fp", [0.0, 2.0])
    @pytest.mark.parametrize("invalid_instance_id, uncertain_instance_id", [(-1, -2), (0, -1)])
    def test_instance_detection_metrics_min_tree_height_fp(
        self, min_tree_height_fp: float, invalid_instance_id: int, uncertain_instance_id: int
    ):
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

        expected_matched_predicted_ids = np.array([0], dtype=np.int64) + start_instance_id
        if min_tree_height_fp > 1.5:
            expected_matched_target_ids = np.array([0, -2], dtype=np.int64) + start_instance_id
        else:
            expected_matched_target_ids = np.array([0, -1], dtype=np.int64) + start_instance_id

        matched_target_ids, matched_predicted_ids, _ = match_instances(
            target,
            prediction,
            xyz=xyz,
            method="panoptic_segmentation",
            invalid_instance_id=invalid_instance_id,
            uncertain_instance_id=uncertain_instance_id,
            min_tree_height_fp=min_tree_height_fp,
        )

        np.testing.assert_array_equal(expected_matched_target_ids, matched_target_ids)
        np.testing.assert_array_equal(expected_matched_predicted_ids, matched_predicted_ids)
