""" Evaluation metrics for assesing the accuracy of tree instance segmentation algorithms. """

__all__ = ["match_instances", "instance_segmentation_metrics", "semantic_segmentation_metrics"]

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def match_instances(  # pylint: disable=too-many-locals
    target: np.ndarray, prediction: np.ndarray, invalid_tree_id: int = -1, iou_treshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    r"""
    Matches tree instances from the ground truth and the prediction based on their intersection over union (IoU).

    Args:
        target: Ground truth instance ID for each point.
        prediction: Predicted instance ID for each point.
        invalid_tree_id: Instance ID that is assigned to points not belonging to any tree instance. Defaults to -1.
        iou_treshold: Minimum IoU score that a ground truth and a predicted tree instance must exceed in order to be
            matched. The value must be greater than 0.5 in order to achieve a unique matching. Defaults to 0.5.

    Returns:
        Tuple of two numpy arrays and one pandas.DataFrame. The first array contains the ID of the matching target
        instance for each predicted instance. Predicted instances that are not matched to a target instance are
        assigned :code:`invalid_tree_id`. The second array contains the ID of the matching predicted instance for
        each target instance. Target instances that are not matched to a predicted instance are assigned
        :code:`invalid_tree_id`. The DataFrame contains one row per matched instance. It contains five columns
        `"Target"`, `"Prediction"`, `"IoU"`, `"Precision"`, `"Recall"` that contain the target ID and the predicted ID
        that match, and the IoU, precision, and recall between the two instances.

    Shape:
        - :code:`target`: :math:`(N)`
        - :code:`prediction`: :math:`(N)`
        - Output: The first array in the output tuple has shape :math:`(P)` and the second shape :math:`(T)`.

        | where
        |
        | :math:`N = \text{ number of tree points}`
        | :math:`P = \text{ number of predicted tree instances}`
        | :math:`T = \text{ number of target tree instances}`
    """

    target_tree_ids = np.unique(target)
    target_tree_ids = target_tree_ids[target_tree_ids != invalid_tree_id]
    predicted_tree_ids = np.unique(prediction)
    predicted_tree_ids = predicted_tree_ids[predicted_tree_ids != invalid_tree_id]

    if target_tree_ids.min() != 0 or target_tree_ids.max() != len(target_tree_ids) - 1:
        raise ValueError("The target instance IDs must be continuous, starting with zero.")

    if predicted_tree_ids.min() != 0 or predicted_tree_ids.max() != len(predicted_tree_ids) - 1:
        raise ValueError("The predicted instance IDs must be continuous, starting with zero.")

    matched_target_ids = np.full(len(predicted_tree_ids), fill_value=invalid_tree_id, dtype=np.int64)
    matched_predicted_ids = np.full(len(target_tree_ids), fill_value=invalid_tree_id, dtype=np.int64)
    metrics = []

    for target_id in target_tree_ids:
        if target_id == invalid_tree_id:
            continue
        predicted_ids = prediction[np.logical_and(target == target_id, prediction != invalid_tree_id)]
        if len(predicted_ids) == 0:
            continue
        values, counts = np.unique(predicted_ids, return_counts=True)
        predicted_id = values[np.argmax(counts)]
        intersection = np.logical_and(target == target_id, prediction == predicted_id).sum()
        union = np.logical_or(target == target_id, prediction == predicted_id).sum()
        iou = intersection / union
        precision = intersection / (prediction == predicted_id).sum()
        recall = intersection / (target == target_id).sum()

        if iou > iou_treshold:
            matched_target_ids[predicted_id] = target_id
            matched_predicted_ids[target_id] = predicted_id
            metrics.append(
                {"Target": target_id, "Prediction": predicted_id, "IoU": iou, "Precision": precision, "Recall": recall}
            )

    return matched_target_ids, matched_predicted_ids, pd.DataFrame(metrics)


def instance_segmentation_metrics(
    matched_target_ids: np.ndarray,
    matched_predicted_ids: np.ndarray,
    segmentation_metrics: pd.DataFrame,
    invalid_tree_id: int = -1,
) -> Dict[str, float]:
    """
    Calculates instance segmentation metrics.

    Args:
        matched_target_ids: ID of the matching target instance for each predicted instance.
        matched_predicted_ids: ID of the matching predicted instance for each target instance.
        segmentation_metrics: Segmentation metrics for each pair of matched target and predicted instance.
        invalid_tree_id: ID that is assigned to instances that could not be matched. Defaults to -1.

    Returns:
        A dictionary containing the following instance segmentation keys: `"tp"`, `"fp"`, `"fn"`, `"panoptic_quality"`,
        `"detection_f1_score"`, `"detection_precision"`, `"detection_recall"`, `"segmentation_m_iou"`,
        `"segmentation_m_precision"`, `"segmentation_m_recall"`.
    """

    tp = (matched_target_ids != invalid_tree_id).sum()

    if tp != (matched_predicted_ids != invalid_tree_id).sum():
        raise ValueError("The number of matched instances is not the same for target and prediction.")

    fp = (matched_target_ids == invalid_tree_id).sum()
    fn = (matched_predicted_ids == invalid_tree_id).sum()

    detection_precision = tp / (tp + fp)
    detection_recall = tp / (tp + fn)
    detection_f1_score = 2 * detection_precision * detection_recall / (detection_precision + detection_recall)
    segmentation_m_iou = segmentation_metrics["IoU"].to_numpy().mean()
    segmentation_m_precision = segmentation_metrics["Precision"].to_numpy().mean()
    segmentation_m_recall = segmentation_metrics["Recall"].to_numpy().mean()
    panoptic_quality = detection_f1_score * segmentation_m_iou

    metrics = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "panoptic_quality": panoptic_quality,
        "detection_f1_score": detection_f1_score,
        "detection_precision": detection_precision,
        "detection_recall": detection_recall,
        "segmentation_m_iou": segmentation_m_iou,
        "segmentation_m_precision": segmentation_m_precision,
        "segmentation_m_recall": segmentation_m_recall,
    }

    return metrics


def semantic_segmentation_metrics(  # pylint: disable=too-many-locals, too-many-statements
    target: np.ndarray,
    prediction: np.ndarray,
    class_map: Dict[str, int],
    aggregate_classes: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, float]:
    """
    Calculates semantic segmentation metrics.

    Args:
        target: Ground truth semantic class IDs for each point.
        prediction: Predicted semantic class IDs for each point.
        class_map: A dictionary mapping class names to numeric class IDs.
        aggregate_classes: A dictionary with which aggregations of classes can be defined. The keys are the names of the
            aggregated classes and the values are lists of the IDs of the classes to be aggregated.

    Returns:
        A dictionary containing the following keys for each semantic class: `"<class_name>_iou"`,
        `"<class_name>_precision"`, `"<class_name>_recall"`. For each aggregated class, the keys
        `"<class_name>_iou_aggregated"`, `"<class_name>_precision_aggregated"`, `"<class_name>_recall_aggregated"` are
        provided.
    """

    if len(target) != len(prediction):
        raise ValueError("Target and prediction must have the same shape.")

    aggregated_class_ids = []
    if aggregate_classes is not None:
        for class_ids in aggregate_classes.values():
            aggregated_class_ids.extend(class_ids)

    metrics = {}

    ious = []
    ious_aggregated = []

    precisions = []
    precisions_aggregated = []

    recalls = []
    recalls_aggregated = []

    for class_name, class_id in class_map.items():
        tp = np.logical_and(target == class_id, prediction == class_id).sum()
        fp = np.logical_and(target != class_id, prediction == class_id).sum()
        fn = np.logical_and(target == class_id, prediction != class_id).sum()

        iou = tp / (tp + fp + fn)
        metrics[f"{class_name}_iou"] = iou
        ious.append(iou)

        precision = tp / (tp + fp)
        metrics[f"{class_name}_precision"] = precision
        precisions.append(precision)

        recall = tp / (tp + fn)
        metrics[f"{class_name}_recall"] = recall
        recalls.append(recall)

        if class_id not in aggregated_class_ids:
            ious_aggregated.append(iou)
            precisions_aggregated.append(precision)
            recalls_aggregated.append(recall)

    metrics["m_iou"] = np.array(ious).mean()
    metrics["m_precision"] = np.array(precisions).mean()
    metrics["m_recall"] = np.array(recalls).mean()

    if aggregate_classes is not None:
        for class_name, aggregate_class_ids in aggregate_classes.items():
            target_mask = np.zeros(len(target), dtype=bool)
            prediction_mask = np.zeros(len(prediction), dtype=bool)
            for class_id in aggregate_class_ids:
                target_mask = np.logical_or(target_mask, target == class_id)
                prediction_mask = np.logical_or(prediction_mask, prediction == class_id)

            tp = np.logical_and(target_mask, prediction_mask).sum()
            fp = np.logical_and(np.logical_not(target_mask), prediction_mask).sum()
            fn = np.logical_and(target_mask, np.logical_not(prediction_mask)).sum()

            iou = tp / (tp + fp + fn)
            metrics[f"{class_name}_iou"] = iou
            ious_aggregated.append(iou)

            precision = tp / (tp + fp)
            metrics[f"{class_name}_precision"] = precision
            precisions_aggregated.append(precision)

            recall = tp / (tp + fn)
            metrics[f"{class_name}_recall"] = recall
            recalls_aggregated.append(recall)

    metrics["m_iou_aggregated"] = np.array(ious_aggregated).mean()
    metrics["m_precision_aggregated"] = np.array(precisions_aggregated).mean()
    metrics["m_recall_aggregated"] = np.array(recalls_aggregated).mean()

    return metrics
