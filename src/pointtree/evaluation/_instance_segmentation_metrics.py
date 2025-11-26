"""Evaluation metrics for assesing the accuracy of instance segmentation methods."""

__all__ = [
    "instance_detection_metrics",
    "instance_segmentation_metrics",
    "instance_segmentation_metrics_per_partition",
    "evaluate_instance_segmentation",
]

from typing import Dict, Literal, Optional, Tuple, cast

from numba import njit, prange
import numpy as np
import pandas as pd

from pointtree.type_aliases import FloatArray, LongArray
from ._match_instances import match_instances


def instance_detection_metrics(  # pylint: disable=too-many-locals
    target: LongArray,
    prediction: LongArray,
    matched_predicted_ids: LongArray,
    matched_target_ids: LongArray,
    *,
    invalid_instance_id: int = -1,
    uncertain_instance_id: int = -2,
):
    r"""
    Computes metrics to measure the instance detection quality. Based on a given matching of ground-truth
    instances :math:`\mathcal{G}_i` and corresponding predicted instances :math:`\mathcal{P}_i`, the instances are
    categorized as true positives (:math:`TP`), false positives (:math:`FP`), or false negatives (:math:`FN`). As
    proposed in `Henrich, Jonathan, et al. "TreeLearn: A Deep Learning Method for Segmenting Individual Trees from \
    Ground-Based LiDAR Forest Point Clouds." Ecological Informatics 84 (2024): 102888. \
    <https://doi.org/10.1016/j.ecoinf.2024.102888>`__, unmatched predicted instances are not counted as false positives
    if less than :code:`min_precision_fp` of their points belong to labeled ground-truth instances. This is because
    such cases often correspond to instances that are correctly detected but not labeled in the ground truth.

    Based on the number of true positives, false positives, and false negatives the following instance detection metrics
    are calculated:

    .. math::
        \text{Precision} = \frac{TP}{TP + FP}

    .. math::
        \text{Commission error} = \frac{FP}{TP + FP}

    .. math::
        \text{Recall} = \frac{TP}{TP + FN}

    .. math::
        \text{Omission error} = \frac{FN}{TP + FN}

    .. math::
        \text{F$_1$-Score} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}

    Args:
        target: Ground truth instance ID for each point.
        prediction: Predicted instance ID for each point.
        matched_predicted_ids: ID of the matched predicted instance for each ground-truth instance.
        matched_target_ids: ID of the matched ground-truth instance for each predicted instance.
        invalid_instance_id: ID that is assigned to points not assigned to any instance / to instances that could not be
            matched and are considered to be false negative or false positive instances.
        uncertain_instance_id: ID that is assigned to predicted instances that could not be matched to any target
            instance but still should not be counted as false positive instances. Must be equal to or smaller than
            :code:`invalid_instance_id`.

    Raises:
        - ValueError: If :code:`uncertain_instance_id` is larger than :code:`invalid_instance_id`.
        - ValueError: If :code:`target` and :code:`prediction` have different lengths.
        - ValueError: If the length of :code:`matched_predicted_ids` is not equal to the number of ground-truth.
          instances
        - ValueError: If the length of :code:`matched_target_ids` is not equal to the number of predicted instances.
        - ValueError: If the unique target and predicted instance IDs don't start with the same number.

    Returns:
        :A dictionary with the following keys: `"TP"`, `"FP"`, `"FN"`, `"Precision"`, `"CommissionError"` `"Recall"`,
        `"OmissionError"`, `"F1Score"`.

    Shape:
        - :code:`target`: :math:`(N)`
        - :code:`prediction`: :math:`(N)`
        - :code:`matched_predicted_ids`: :math:`(G)`
        - :code:`matched_target_ids`: :math:`(P)`

        | where
        |
        | :math:`N = \text{ number of points}`
        | :math:`G = \text{ number of ground-truth instances}`
        | :math:`P = \text{ number of predicted instances}`
    """

    if uncertain_instance_id > invalid_instance_id:
        raise ValueError("uncertain_instance_id must be smaller or equal to invalid_instance_id.")

    if len(target) != len(prediction):
        raise ValueError("Target and prediction must have the same length.")

    target_ids = np.unique(target)
    target_ids = target_ids[target_ids != invalid_instance_id]

    if len(matched_predicted_ids) != len(target_ids):
        raise ValueError("The length of matched_predicted_ids must be equal to the number of target instances.")

    prediction_ids = np.unique(prediction)
    prediction_ids = prediction_ids[prediction_ids != invalid_instance_id]

    start_instance_id_target = target_ids.min() if len(target_ids) > 0 else None
    start_instance_id_prediction = prediction_ids.min() if len(prediction_ids) else None

    if (
        start_instance_id_target is not None
        and start_instance_id_prediction is not None
        and start_instance_id_target != start_instance_id_prediction
    ):
        raise ValueError("Start instance IDs for target and prediction must be identical.")

    if len(matched_target_ids) != len(prediction_ids):
        raise ValueError("The length of matched_target_ids must be equal to the number of predicted instances.")

    tp = (np.logical_not(np.isin(matched_predicted_ids, (invalid_instance_id, uncertain_instance_id)))).sum()
    fn = (matched_predicted_ids == invalid_instance_id).sum()
    fp = (matched_target_ids == invalid_instance_id).sum()

    metrics = {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        "CommissionError": fp / (tp + fp) if (tp + fp) > 0 else np.nan,
        "Recall": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        "OmissionError": fn / (tp + fn) if (tp + fn) > 0 else np.nan,
        "F1Score": 2 * tp / (2 * tp + fp + fn),
    }

    return metrics


@njit(parallel=True)
def _compute_instance_segmentation_metrics(
    target: LongArray,
    prediction: LongArray,
    matched_predicted_ids: LongArray,
    start_instance_id: int,
    invalid_instance_id: int,
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    r"""
    Computes metrics to measure the quality of the point-wise segmentation.

    Args:
        target: Ground truth instance ID for each point.
        prediction: Predicted instance ID for each point.
        matched_predicted_ids: ID of the matched predicted instance for each ground-truth instance.
        start_instance_id: Smallest valid instance ID. All instance IDs are expected to be consecutive.
        invalid_instance_id: ID that is assigned to points not assigned to any instance / to instances that could not be
            matched.

    Returns:
        :A tuple of three arrays containing the metrics for each instance pair:
            - IoU
            - Precision
            - Recall

            For ground-truth instances that have not been matched to any predicted instance, the metrics are set to
            zero.

    Shape:
        - :code:`target`: :math:`(N)`
        - :code:`prediction`: :math:`(N)`
        - :code:`matched_predicted_ids`: :math:`(G)`
        - Output: Three arrays of shape :math:`(N)`

        | where
        |
        | :math:`N = \text{ number of points}`
        | :math:`G = \text{ number of ground-truth instances}`
    """

    num_target_ids = len(matched_predicted_ids)

    iou = np.zeros(num_target_ids, dtype=np.float64)
    precision = np.zeros(num_target_ids, dtype=np.float64)
    recall = np.zeros(num_target_ids, dtype=np.float64)

    for target_idx in prange(num_target_ids):  # pylint: disable=not-an-iterable
        target_id = start_instance_id + target_idx
        predicted_id = matched_predicted_ids[target_idx]

        if predicted_id == invalid_instance_id:
            continue

        intersection = np.logical_and(target == target_id, prediction == predicted_id).sum()
        union = np.logical_or(target == target_id, prediction == predicted_id).sum()

        iou[target_idx] = intersection / union
        precision[target_idx] = intersection / (prediction == predicted_id).sum()
        recall[target_idx] = intersection / (target == target_id).sum()

    return iou, precision, recall


def instance_segmentation_metrics(  # pylint: disable=too-many-locals
    target: LongArray,
    prediction: LongArray,
    matched_predicted_ids: LongArray,
    invalid_instance_id: int = -1,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    r"""
    Given pairs of ground-truth instances :math:`\mathcal{G}_i` and matched predicted instances :math:`\mathcal{P}_i`,
    the following metrics are calculated to measure the quality of the point-wise segmentation:

    .. math::
        \text{IoU}(\mathcal{G}_i, \mathcal{P}_i) = \frac{|\mathcal{G}_i \cap \mathcal{P}_{i}|}{|\mathcal{G}_i \cup
        \mathcal{P}_{i}|}

    .. math::
        \text{Precision}(\mathcal{G}_i, \mathcal{P}_i) = \frac{|\mathcal{G}_i \cap \mathcal{P}_{i}|}{|\mathcal{P}_{i}|}

    .. math::
        \text{Recall}(\mathcal{G}_i, \mathcal{P}_i) = \frac{|\mathcal{G}_i \cap \mathcal{P}_{i}|}{|\mathcal{G}_i|}

    Then, the metrics are averaged over all instance pairs of target instances and matched predicted instances:

    .. math::
        \text{mIoU} = \frac{1}{N_G} \sum_{i=0}^{N_G} \text{IoU}(\mathcal{G}_i, \mathcal{P}_i)

    .. math::
        \text{mPrecision} = \frac{1}{N_G} \sum_{i=0}^{N_G} \text{Precision}(\mathcal{G}_i, \mathcal{P}_i)

    .. math::
        \text{mRecall} = \frac{1}{N_G} \sum_{i=0}^{N_G} \text{Recall}(\mathcal{G}_i, \mathcal{P}_i)

    Args:
        target: Ground truth instance ID for each point.
        prediction: Predicted instance ID for each point.
        matched_predicted_ids: ID of the matched predicted instance for each ground-truth instance.
        invalid_instance_id: ID that is assigned to points not assigned to any instance / to instances that could not be
            matched.

    Raises:
        - ValueError: if :code:`target` and :code:`prediction` have different lengths
        - ValueError: if the length of :code:`matched_predicted_ids` is not equal to the number of ground-truth
          instances.
        - ValueError: If the unique target and predicted instance IDs don't start with the same number.

    Returns:
        :A tuple with two elements:
            - A dictionary containing the segmentation metrics averaged over all instance pairs. The dictionary contains
              the following keys: :code:`"MeanIoU"`, :code:`"MeanPrecision"`, and :code:`"MeanRecall"`.
            - A pandas.DataFrame containing the segmentation metrics for each instance pair. The dataframe contains the
              following columns: :code:`"TargetID"`, :code:`"PredictionID"`, :code:`"IoU"`, :code:`"Precision"`,
              :code:`"Recall"`.

    Shape:
        - :code:`target`: :math:`(N)`
        - :code:`prediction`: :math:`(N)`
        - :code:`matched_predicted_ids`: :math:`(G)`

        | where
        |
        | :math:`N = \text{ number of points}`
        | :math:`G = \text{ number of ground-truth instances}`
    """

    if len(target) != len(prediction):
        raise ValueError("Target and prediction must have the same length.")

    target_ids = np.unique(target)
    target_ids = target_ids[target_ids != invalid_instance_id]

    if len(matched_predicted_ids) != len(target_ids):
        raise ValueError("The length of matched_predicted_ids must be equal to the number of target instances.")

    matched_instances_mask = matched_predicted_ids != invalid_instance_id

    if matched_instances_mask.sum() == 0:
        average_metrics = {
            "MeanIoU": np.nan,
            "MeanPrecision": np.nan,
            "MeanRecall": np.nan,
        }
        per_instance_metrics = pd.DataFrame([], columns=["TargetID", "PredictionID", "IoU", "Precision", "Recall"])

        return average_metrics, per_instance_metrics

    prediction_ids = np.unique(prediction)
    prediction_ids = prediction_ids[prediction_ids != invalid_instance_id]

    start_instance_id_target = target_ids.min()
    start_instance_id_prediction = prediction_ids.min()

    if start_instance_id_target != start_instance_id_prediction:
        raise ValueError("Start instance IDs for target and prediction must be identical.")

    num_target_ids = len(matched_predicted_ids)

    iou, precision, recall = _compute_instance_segmentation_metrics(
        target,
        prediction,
        matched_predicted_ids,
        start_instance_id_target,
        invalid_instance_id,
    )

    average_metrics = {
        "MeanIoU": iou[matched_instances_mask].mean(),
        "MeanPrecision": precision[matched_instances_mask].mean(),
        "MeanRecall": recall[matched_instances_mask].mean(),
    }

    target_ids = np.arange(
        start=start_instance_id_target, stop=start_instance_id_target + num_target_ids, dtype=np.int64
    )
    per_instance_metrics = pd.DataFrame(
        np.column_stack((target_ids, matched_predicted_ids))[matched_instances_mask],
        columns=["TargetID", "PredictionID"],
    )
    per_instance_metrics["IoU"] = iou[matched_instances_mask]
    per_instance_metrics["Precision"] = precision[matched_instances_mask]
    per_instance_metrics["Recall"] = recall[matched_instances_mask]

    return average_metrics, per_instance_metrics


@njit(parallel=True)
def _compute_instance_segmentation_metrics_per_partition(  # pylint: disable=too-many-locals
    xyz: FloatArray,
    target: LongArray,
    prediction: LongArray,
    matched_predicted_ids: LongArray,
    partition: Literal["xy", "z"],
    start_instance_id: int,
    invalid_instance_id: int,
    num_partitions: int = 10,
):
    r"""
    Calculates instance segmentation metrics for different spatial partitions of a tree instance.

    Args:
        xyz: Coordinates of all points.
        target: Ground truth instance ID for each point.
        prediction: Predicted instance ID for each point.
        matched_predicted_ids: ID of the matched predicted instance for each ground-truth instance.
        partition: Partioning schme to be used: `"xy"` | `"z"`.
        start_instance_id: Smallest valid instance ID. All instance IDs are expected to be consecutive.
        invalid_instance_id: ID that is assigned to points not assigned to any instance / to instances that could not be
            matched.
        num_partitions: Number of partitions.

    Raises:
        - ValueError: if :code:`partition` is set to an invalid value.
        - ValueError: if :code:`xyz` and :code:`target` have different lengths.
        - ValueError: if :code:`target` and :code:`prediction` have different lengths.
        - ValueError: if the length of :code:`matched_predicted_ids` is not equal to the number of ground-truth
          instances.

    Returns:
        :A tuple of three arrays containing the metrics for each instance pair:
            - IoU
            - Precision
            - Recall

            For ground-truth instances that have not been matched to any predicted instance, the metrics are set to
            zero.

    Shape:
        - :code:`xyz`: :math:`(N, 3)`
        - :code:`target`: :math:`(N)`
        - :code:`prediction`: :math:`(N)`
        - :code:`matched_predicted_ids`: :math:`(G)`
        - Output: Three arrays of shape :math:(G, P)

        | where
        |
        | :math:`N = \text{ number of points}`
        | :math:`G = \text{ number of ground-truth instances}`
        | :math:`P = \text{ number of partitions}`
    """
    intervals = np.linspace(0, 1, num_partitions + 1)

    num_target_ids = len(matched_predicted_ids)

    iou = np.full((num_target_ids, num_partitions), fill_value=np.nan, dtype=np.float64)
    precision = np.full((num_target_ids, num_partitions), fill_value=np.nan, dtype=np.float64)
    recall = np.full((num_target_ids, num_partitions), fill_value=np.nan, dtype=np.float64)

    for target_idx in prange(num_target_ids):  # pylint: disable=not-an-iterable
        target_id = start_instance_id + target_idx
        predicted_id = matched_predicted_ids[target_idx]

        if predicted_id == invalid_instance_id:
            continue

        target_mask = target == target_id

        tree_xyz = xyz[target_mask]
        min_z = np.min(tree_xyz[:, 2])

        distance = np.zeros(len(tree_xyz), dtype=tree_xyz.dtype)

        if partition == "xy":
            # calculate tree position and center xy-coordinates according to it
            z_threshold = min_z + 0.30
            lowest_points = tree_xyz[tree_xyz[:, 2] <= z_threshold]

            position = np.zeros(2, dtype=lowest_points.dtype)
            position[0] = np.mean(lowest_points[:, 0])
            position[1] = np.mean(lowest_points[:, 1])
            xy_centered = xyz[:, :2] - position

            # relative distance to tree center (0=seedpoint, 1=most distant point)
            distance = xy_centered**2
            distance = np.sqrt(distance[:, 0] + distance[:, 1])
            distance_target = distance[target_mask]
            regularized_max = np.quantile(distance_target, 0.95)
            distance = distance / regularized_max

        elif partition == "z":
            # get relative distance to lowest point (0=lowest point, 1=highest point)
            distance = xyz[:, 2] - min_z

            regularized_max = np.quantile(tree_xyz[:, 2], 0.95)
            distance = distance / (regularized_max - min_z)

        for i in range(num_partitions):
            partition_mask = np.logical_and(distance >= intervals[i], distance < intervals[i + 1])

            if partition_mask.sum() == 0:
                continue

            partition_target = target[partition_mask]
            partition_prediction = prediction[partition_mask]

            intersection = np.logical_and(partition_target == target_id, partition_prediction == predicted_id).sum()
            union = np.logical_or(partition_target == target_id, partition_prediction == predicted_id).sum()

            if union > 0:
                iou[target_idx, i] = intersection / union
            if (partition_prediction == predicted_id).sum() > 0:
                precision[target_idx, i] = intersection / (partition_prediction == predicted_id).sum()
            if (partition_target == target_id).sum() > 0:
                recall[target_idx, i] = intersection / (partition_target == target_id).sum()

    return iou, precision, recall


def instance_segmentation_metrics_per_partition(  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    xyz: FloatArray,
    target: LongArray,
    prediction: LongArray,
    matched_predicted_ids: LongArray,
    partition: Literal["xy", "z"],
    invalid_instance_id: int = -1,
    num_partitions: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r"""
    Calculates instance segmentation metrics for different spatial partitions of a tree instance as proposed in 
    `Henrich, Jonathan, et al. "TreeLearn: A Deep Learning Method for Segmenting Individual Trees from \
    Ground-Based LiDAR Forest Point Clouds." Ecological Informatics 84 (2024): 102888. \
    <https://doi.org/10.1016/j.ecoinf.2024.102888>`__ The reasoning behind this is that not all parts of a tree are
    equally difficult to segment. For example, points near the trunk are usually easier to assign than points at the
    crown boundary, where there are many interactions with other trees. To quantify how well different tree parts
    are segmented, the points of the ground-truth tree instances and the corresponding predicted instances are
    partitioned into :code:`num_partitions` subsets. The segmentation metrics are then calculated separately for each
    subset and averaged across all pairs of ground-truth instances and corresponding predicted instances.

    Henrich et al. propose two axes for partitioning: (1) horizontal distance to the trunk, and (2) vertical distance to
    the forest ground. For the horizontal partition, the :math:`i`-th subset contains all points with a horizontal
    distance to the ground truth trunk between :math:`\frac{i-1}{N_p}\cdot r` and :math:`\frac{i}{N_p}\cdot r` where
    :math:`N_p` is the number of partitions and :math:`r` is the maximum distance to the trunk of all points in the
    ground-truth instance. For the vertical partition, the :math:`i`-th subset contains all points with a vertical
    distance to the ground between :math:`\frac{i-1}{N_p}\cdot h` and :math:`\frac{i}{N_p}\cdot h` where :math:`N_p` is
    the number of partitions and :math:`h` is the is the height of the ground-truth tree instance. Points of a
    prediction that are farther away than :math:`r` and :math:`h` from the trunk and ground, respectively, are not taken
    into account in this part of the evaluation.

    Given pairs of ground-truth instances :math:`\mathcal{G}_i` and matched predicted instances :math:`\mathcal{P}_i`,
    the following metrics are calculated for each partition:

    .. math::
        \text{IoU}(\mathcal{G}_i, \mathcal{P}_i) = \frac{|\mathcal{G}_i \cap \mathcal{P}_{i}|}{|\mathcal{G}_i \cup
        \mathcal{P}_{i}|}

    .. math::
        \text{Precision}(\mathcal{G}_i, \mathcal{P}_i) = \frac{|\mathcal{G}_i \cap \mathcal{P}_{i}|}{|\mathcal{P}_{i}|}

    .. math::
        \text{Recall}(\mathcal{G}_i, \mathcal{P}_i) = \frac{|\mathcal{G}_i \cap \mathcal{P}_{i}|}{|\mathcal{G}_i|}

    Then, the metrics are averaged over all instance pairs of target instances and matched predicted instances:

    .. math::
        \text{mIoU} = \frac{1}{N_G} \sum_{i=0}^{N_G} \text{IoU}(\mathcal{G}_i, \mathcal{P}_i)

    .. math::
        \text{mPrecision} = \frac{1}{N_G} \sum_{i=0}^{N_G} \text{Precision}(\mathcal{G}_i, \mathcal{P}_i)

    .. math::
        \text{mRecall} = \frac{1}{N_G} \sum_{i=0}^{N_G} \text{Recall}(\mathcal{G}_i, \mathcal{P}_i)

    Args:
        xyz: Coordinates of all points.
        target: Ground truth instance ID for each point.
        prediction: Predicted instance ID for each point.
        matched_predicted_ids: ID of the matched predicted instance for each ground-truth instance.
        partition: Partioning schme to be used: `"xy"` | `"z"`.
        invalid_instance_id: ID that is assigned to points not assigned to any instance / to instances that could not be
            matched.
        num_partitions: Number of partitions.

    Raises:
        - ValueError: if :code:`partition` is set to an invalid value.
        - ValueError: if :code:`xyz` and :code:`target` have different lengths.
        - ValueError: if :code:`target` and :code:`prediction` have different lengths.
        - ValueErorr: if the length of :code:`matched_predicted_ids` is not equal to the number of ground-truth
            instances.
        - ValueError: If the unique target and predicted instance IDs don't start with the same number.

    Returns:
        :A tuple of two pandas.DataFrames:
            - Segmentation metrics for each partition averaged over all instance pairs. The dataframe contains the
              following columns: :code:`"Partition",` :code:`"MeanIoU"`, :code:`"MeanPrecision"`, and
              :code:`"MeanRecall"`.
            - Segmentation metrics for each partition and each instance pair. The dataframe contains the following keys:
              :code:`"Partition"`, :code:`"TargetID"`, :code:`"PredictionID"`, :code:`"IoU"`, :code:`"Precision"`,
              :code:`"Recall"`.

    Shape:
        - :code:`xyz`: :math:`(N, 3)`
        - :code:`target`: :math:`(N)`
        - :code:`prediction`: :math:`(N)`
        - :code:`matched_predicted_ids`: :math:`(G)`

        | where
        |
        | :math:`N = \text{ number of points}`
        | :math:`G = \text{ number of ground-truth instances}`
    """
    if partition not in ["xy", "z"]:
        raise ValueError(f"Invalid partition {partition}.")

    if len(xyz) != len(target):
        raise ValueError("xyz and target must have the same length.")

    if len(target) != len(prediction):
        raise ValueError("Target and prediction must have the same length.")

    target_ids = np.unique(target)
    target_ids = target_ids[target_ids != invalid_instance_id]

    if len(matched_predicted_ids) != len(target_ids):
        raise ValueError("The length of matched_predicted_ids must be equal to the number of target instances.")

    prediction_ids = np.unique(prediction)
    prediction_ids = prediction_ids[prediction_ids != invalid_instance_id]

    if len(target_ids) == 0 or len(prediction_ids) == 0:
        average_metrics = [
            {"Partition": i, "MeanIoU": np.nan, "MeanPrecision": np.nan, "MeanRecall": np.nan}
            for i in range(num_partitions)
        ]

        per_instance_metrics = pd.DataFrame(
            [], columns=["TargetID", "PredictionID", "Partition", "IoU", "Precision", "Recall"]
        )

        return pd.DataFrame(average_metrics), per_instance_metrics

    start_instance_id_target = target_ids.min()
    start_instance_id_prediction = prediction_ids.min()

    if start_instance_id_target != start_instance_id_prediction:
        raise ValueError("Start instance IDs for target and prediction must be identical.")

    iou, precision, recall = _compute_instance_segmentation_metrics_per_partition(
        xyz,
        target,
        prediction,
        matched_predicted_ids,
        partition,
        start_instance_id_target,
        invalid_instance_id,
        num_partitions,
    )

    average_metrics = []
    for i in range(num_partitions):
        average_metrics.append(
            {
                "Partition": i,
                "MeanIoU": np.nanmean(iou[:, i]) if (~np.isnan(iou[:, i])).sum() > 0 else np.nan,
                "MeanPrecision": np.nanmean(precision[:, i]) if (~np.isnan(precision[:, i])).sum() > 0 else np.nan,
                "MeanRecall": np.nanmean(recall[:, i]) if (~np.isnan(recall[:, i])).sum() > 0 else np.nan,
            }
        )

    matched_instances_mask = matched_predicted_ids != invalid_instance_id

    target_ids = np.arange(
        start=start_instance_id_target, stop=start_instance_id_target + len(target_ids), dtype=np.int64
    )
    per_instance_metrics = pd.DataFrame(
        np.repeat(
            np.column_stack((target_ids[matched_instances_mask], matched_predicted_ids[matched_instances_mask])),
            num_partitions,
            axis=0,
        ),
        columns=["TargetID", "PredictionID"],
    )
    per_instance_metrics["Partition"] = np.repeat(
        np.arange(num_partitions).reshape(-1, num_partitions), matched_instances_mask.sum(), axis=0
    ).flatten()
    per_instance_metrics["IoU"] = iou[matched_instances_mask].reshape(-1)
    per_instance_metrics["Precision"] = precision[matched_instances_mask].reshape(-1)
    per_instance_metrics["Recall"] = recall[matched_instances_mask].reshape(-1)

    return pd.DataFrame(average_metrics), pd.DataFrame(per_instance_metrics)


def evaluate_instance_segmentation(  # pylint: disable=too-many-branches,too-many-locals, too-many-arguments
    xyz: FloatArray,
    target: LongArray,
    prediction: LongArray,
    *,
    detection_metrics_matching_method: Literal[
        "panoptic_segmentation",
        "point2tree",
        "for_instance",
        "for_ai_net",
        "for_ai_net_coverage",
        "segment_any_tree",
        "tree_learn",
    ] = "panoptic_segmentation",
    segmentation_metrics_matching_method: Literal[
        "panoptic_segmentation",
        "point2tree",
        "for_instance",
        "for_ai_net",
        "for_ai_net_coverage",
        "segment_any_tree",
        "tree_learn",
    ] = "for_ai_net_coverage",
    invalid_instance_id: int = -1,
    uncertain_instance_id: int = -2,
    compute_partition_metrics: bool = True,
    num_partitions: int = 10,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
]:
    r"""
    Evaluates the quality of an instance segmentation by computing the following types of metrics:

    - Instance detection metrics (precision, commission error, recall, omission error, :math:`\text{F}_1`-score)
    - Instance segmentation metrics averaged over all pairs of ground-truth instances and corresponding predicted
      instances (mIoU, mPrecision, mRecall)
    - Instance segmentation metrics for each pair of a ground-truth instance and a corresponding predicted
      instance
    - Instance segmentation metrics for different spatial partitions of the instances, averaged over all pairs of
      ground-truth instances and corresponding predicted instances
    - Instance segmentation metrics for different spatial partitions of the instances for each pair of a ground-truth
      instance and a corresponding predicted instance.

    For more details on the individual metrics, see the documentation of
    :code:`pointtorch.evaluation.instance_detection_metrics`,
    :code:`pointtorch.evaluation.instance_segmentation_metrics`, and
    :code:`pointtorch.evaluation.instance_segmentation_metrics_per_partition`. The metric calculations are based on a
    matching of ground-truth instances and predicted instances. More details on this matching are provided in the
    documentation of :code:`pointtorch.evaluation.match_instances`.

    Args:
        xyz: Coordinates of all points.
        target: Ground truth instance ID for each point.
        prediction: Predicted instance ID for each point.
        detection_metrics_matching_method: Method to be used for matching ground-truth and predicted instances for
            computing the instance detection metrics.
        segmentation_metrics_matching_method: Method to be used for matching ground-truth and predicted instances for
            computing the instance segmentation metrics.
        invalid_instance_id: ID that is assigned to points not assigned to any instance / to instances that could not be
            matched and are considered to be false negative or false positive instances.
        uncertain_instance_id: ID that is assigned to predicted instances that could not be matched to any target
            instance but still should not be counted as false positive instances.  Must be equal to or smaller than
            :code:`invalid_instance_id`.
        compute_partition_metrics: Whether the metrics per partition should be computed.
        num_partitions: Number of partitions for the computation of instance segmentation metrics per partition.

    Returns:
        :Tuple of six pandas.DataFrames:
            - Instance detection metrics and the instance segmentation metrics averaged over all instance pairs. The
              dataframe has the following columns: :code:`"DetectionTP"`, :code:`"DetectionFP"`, :code:`"DetectionFN"`,
              :code:`"DetectionPrecision"`, :code:`"DetectionComissionError"`, :code:`"DetectionRecall"`,
              :code:`"DetectionOmissionError"`, :code:`"DetectionF1Score"`, :code:`"SegmentationMeanIoU"`,
              :code:`"SegmentationMeanPrecision"`, and :code:`"SegmentationMeanRecall"`.
            - Instance segmentation metrics for each instance pair.
            - Instance segmentation metrics for different horizontal partitions, averaged over all instance pairs.
            - Instance segmentation metrics for different horizontal partitions for each instance pair.
            - Instance segmentation metrics for different vertical partitions, averaged over all instance pairs.
            - Instance segmentation metrics for different vertical partitions for each instance pair.

            The elements containing the metrics per partition are :code:`None` when :code:`compute_partition_metrics` is
            set to :code:`False`.
    """

    matched_target_ids, matched_predicted_ids = match_instances(
        xyz,
        target,
        prediction,
        method=detection_metrics_matching_method,
        invalid_instance_id=invalid_instance_id,
    )

    instance_detect_metrics = instance_detection_metrics(
        target,
        prediction,
        matched_predicted_ids,
        matched_target_ids,
        invalid_instance_id=invalid_instance_id,
        uncertain_instance_id=uncertain_instance_id,
    )

    matched_target_ids, matched_predicted_ids = match_instances(
        xyz,
        target,
        prediction,
        method=segmentation_metrics_matching_method,
        invalid_instance_id=invalid_instance_id,
    )

    avg_segmentation_metrics, per_instance_segmentation_metrics = instance_segmentation_metrics(
        target, prediction, matched_predicted_ids, invalid_instance_id=invalid_instance_id
    )

    avg_segmentation_metrics_per_xy_partition = None
    per_instance_segmentation_metrics_per_xy_partition = None
    avg_segmentation_metrics_per_z_partition = None
    per_instance_segmentation_metrics_per_z_partition = None

    if compute_partition_metrics:
        avg_segmentation_metrics_per_xy_partition, per_instance_segmentation_metrics_per_xy_partition = (
            instance_segmentation_metrics_per_partition(
                xyz,
                target,
                prediction,
                matched_predicted_ids,
                partition="xy",
                invalid_instance_id=invalid_instance_id,
                num_partitions=num_partitions,
            )
        )

        avg_segmentation_metrics_per_z_partition, per_instance_segmentation_metrics_per_z_partition = (
            instance_segmentation_metrics_per_partition(
                xyz,
                target,
                prediction,
                matched_predicted_ids,
                partition="z",
                invalid_instance_id=invalid_instance_id,
                num_partitions=num_partitions,
            )
        )

    metrics = {}

    for key, value in instance_detect_metrics.items():
        metrics[f"Detection{key}"] = value
    for key, value in avg_segmentation_metrics.items():
        metrics[f"Segmentation{key}"] = value

    return (
        pd.DataFrame([metrics]),
        pd.DataFrame(per_instance_segmentation_metrics),
        avg_segmentation_metrics_per_xy_partition,
        per_instance_segmentation_metrics_per_xy_partition,
        avg_segmentation_metrics_per_z_partition,
        per_instance_segmentation_metrics_per_z_partition,
    )
