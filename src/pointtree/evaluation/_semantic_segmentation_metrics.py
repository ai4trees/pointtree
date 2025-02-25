""" Evaluation metrics for assesing the accuracy of semantic segmentation methods. """

__all__ = ["semantic_segmentation_metrics"]

from typing import Dict, List, Optional

import numpy as np


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
        :A dictionary containing the following keys for each semantic class: `"<class_name>IoU"`,
        `"<class_name>Precision"`, `"<class_name>Recall"`. For each aggregated class, the keys
        `"<class_name>IoUAggregated"`, `"<class_name>PrecisionAggregated"`, `"<class_name>RecallAggregated"` are
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
        metrics[f"{class_name}IoU"] = iou
        ious.append(iou)

        precision = tp / (tp + fp)
        metrics[f"{class_name}Precision"] = precision
        precisions.append(precision)

        recall = tp / (tp + fn)
        metrics[f"{class_name}Recall"] = recall
        recalls.append(recall)

        if class_id not in aggregated_class_ids:
            ious_aggregated.append(iou)
            precisions_aggregated.append(precision)
            recalls_aggregated.append(recall)

    metrics["MeanIoU"] = np.array(ious).mean()
    metrics["MeanPrecision"] = np.array(precisions).mean()
    metrics["MeanRecall"] = np.array(recalls).mean()

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
            metrics[f"{class_name}IoU"] = iou
            ious_aggregated.append(iou)

            precision = tp / (tp + fp)
            metrics[f"{class_name}Precision"] = precision
            precisions_aggregated.append(precision)

            recall = tp / (tp + fn)
            metrics[f"{class_name}Recall"] = recall
            recalls_aggregated.append(recall)

    metrics["MeanIoUAggregated"] = np.array(ious_aggregated).mean()
    metrics["MeanPrecisionAggregated"] = np.array(precisions_aggregated).mean()
    metrics["MeanRecallAggregated"] = np.array(recalls_aggregated).mean()

    return metrics
