"""Methods for matching ground-truth instances and predicted instances."""

__all__ = [
    "match_instances",
    "match_instances_iou",
    "match_instances_tree_learn",
    "match_instances_for_ai_net_coverage",
]

from typing import Literal, Optional, Tuple

from numba import njit, prange
import numpy as np
import numpy.typing as npt
import scipy


def match_instances(
    xyz: npt.NDArray,
    target: npt.NDArray[np.int64],
    prediction: npt.NDArray[np.int64],
    method: Literal[
        "panoptic_segmentation",
        "point2tree",
        "for_instance",
        "for_ai_net",
        "for_ai_net_coverage",
        "segment_any_tree",
        "tree_learn",
    ],
    invalid_instance_id: int = -1,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    r"""
    This method implements the instance matching methods proposed in the following works:

    - :code:`panoptic_segmentation`: `Kirillov, Alexander, et al. "Panoptic segmentation." Proceedings of the IEEE/CVF \
      Conference on Computer Vision and Pattern Recognition. 2019. <https://doi.org/10.1109/CVPR.2019.00963>`__
      
      This method matches predicted and ground-truth instances if their IoU is striclty greater than 0.5, which results
      in an unambigous matching.

    - :code:`point2tree`: `Wielgosz, Maciej, et al. "Point2Tree (P2T)—Framework for Parameter Tuning of Semantic and \
      Instance Segmentation Used with Mobile Laser Scanning Data in Coniferous Forest." Remote Sensing 15.15 (2023): \
      3737. <https://doi.org/10.3390/rs15153737>`__

      This method processes the ground-truth instances sorted according to their height. Starting with the highest
      ground-truth instance, each ground-truth instance is matched with the predicted instance with which it has the
      highest IoU. Predicted instances that were already matched to a ground-truth instance before, are excluded from
      the matching.

    - :code:`for_instance`: `Puliti, Stefano, et al. "For-Instance: a UAV Laser Scanning Benchmark Dataset for \
      Semantic and Instance Segmentation of Individual Trees." arXiv preprint arXiv:2309.01279 (2023). \
      <https://arxiv.org/pdf/2309.01279>`__

      This method is based on the method proposed by Wielgosz et al. (2023) but additionally introduces the criterion
      that ground-truth and predicted instances must have an IoU of a least 0.5 to be matched.

    - :code:`for_ai_net`: `Xiang, Binbin, et al. "Automated Forest Inventory: Analysis of High-Density Airborne LiDAR \
      Point Clouds with 3D Deep Learning." Remote Sensing of Environment 305 (2024): 114078. \
      <https://doi.org/10.1016/j.rse.2024.114078>`__

      This method is similar to the method proposed by Kirillov et al., with the difference that ground-truth and
      predicted instances are also matched if their IoU is equal to 0.5.

    - :code:`for_ai_net_coverage`: `Xiang, Binbin, et al. "Automated Forest Inventory: Analysis of High-Density \
      Airborne LiDAR Point Clouds with 3D Deep Learning." Remote Sensing of Environment 305 (2024): 114078. \
      <https://doi.org/10.1016/j.rse.2024.114078>`__

      This method matches each ground truth instance with the predicted instance with which it has the highest IoU. This
      means that predicted instances can be matched with multiple ground-truth instances. Such a matching approach is
      useful for the calculation of segmentation metrics (e.g., coverage) that should be independent from the instance
      detection rate.
  
    - :code:`segment_any_tree`: `Wielgosz, Maciej, et al. "SegmentAnyTree: A Sensor and Platform Agnostic Deep \
      Learning Model for Tree Segmentation Using Laser Scanning Data." Remote Sensing of Environment 313 (2024): \
      114367. <https://doi.org/10.1016/j.rse.2024.114367>`__
      
      This method is the same as the method proposed by Kirillov et al.

    - :code:`tree_learn`: `Henrich, Jonathan, et al. "TreeLearn: A Deep Learning Method for Segmenting Individual Trees
      from Ground-Based LiDAR Forest Point Clouds." Ecological Informatics 84 (2024): 102888.
      <https://doi.org/10.1016/j.ecoinf.2024.102888>`__

      This method uses Hungarian matching to match predicted and ground truth instances in such a way that the sum of
      the IoU scores of all matched instance pairs is maximized. Subsequently, matches with an IoU score less than or
      equal to 0.5 are discarded.

    Args:
        xyz: Coordinates of all points.
        target: Ground truth instance ID for each point.
        prediction: Predicted instance ID for each point.
        method: Instance matching method to be used.
        invalid_instance_id: ID that is assigned to points not assigned to any instance. Defaults to -1.

    Returns:
        Tuple of two arrays. The first contains the ID of the matched ground-truth instance for each predicted instance.
        The second contains the ID of the matched predicted instance for each ground-truth instance.

    Raises:
        ValueError: If the unique instance IDs are not continuous, starting with zero.

    Shape:
        - :code:`xyz`: :math:`(N, 3)`
        - :code:`target`: :math:`(N)`
        - :code:`prediction`: :math:`(N)`
        - Output: Tuple of two arrays. The first has shape :math:`(P)`, the second has shape :math:`(G)`.

        | where
        |
        | :math:`N = \text{ number of points}`
        | :math:`G = \text{ number of ground-truth instances}`
        | :math:`P = \text{ number of predicted instances}`
    """

    unique_target_ids = np.unique(target)
    unique_target_ids = unique_target_ids[unique_target_ids != invalid_instance_id]
    unique_prediction_ids = np.unique(prediction)
    unique_prediction_ids = unique_prediction_ids[unique_prediction_ids != invalid_instance_id]

    if len(unique_target_ids) == 0 or len(unique_prediction_ids) == 0:
        matched_target_ids = np.full(len(unique_prediction_ids), fill_value=invalid_instance_id, dtype=np.int64)
        matched_predicted_ids = np.full(len(unique_target_ids), fill_value=invalid_instance_id, dtype=np.int64)
        return matched_target_ids, matched_predicted_ids

    if unique_target_ids.min() != 0 or unique_target_ids.max() != len(unique_target_ids) - 1:
        raise ValueError("The target instance IDs must be continuous, starting with zero.")

    if unique_prediction_ids.min() != 0 or unique_prediction_ids.max() != len(unique_prediction_ids) - 1:
        raise ValueError("The predicted instance IDs must be continuous, starting with zero.")

    if method in ["panoptic_segmentation", "segment_any_tree"]:
        matching_result = match_instances_iou(
            xyz,
            target,
            unique_target_ids,
            prediction,
            unique_prediction_ids,
            invalid_instance_id=invalid_instance_id,
            min_iou_treshold=0.5,
            accept_equal_iou=False,
            sort_by_target_height=False,
        )
    elif method == "point2tree":
        matching_result = match_instances_iou(
            xyz,
            target,
            unique_target_ids,
            prediction,
            unique_prediction_ids,
            invalid_instance_id=invalid_instance_id,
            min_iou_treshold=None,
            sort_by_target_height=True,
        )
    elif method == "for_instance":
        matching_result = match_instances_iou(
            xyz,
            target,
            unique_target_ids,
            prediction,
            unique_prediction_ids,
            invalid_instance_id=invalid_instance_id,
            min_iou_treshold=0.5,
            accept_equal_iou=True,
            sort_by_target_height=True,
        )
    elif method == "for_ai_net":
        matching_result = match_instances_iou(
            xyz,
            target,
            unique_target_ids,
            prediction,
            unique_prediction_ids,
            invalid_instance_id=invalid_instance_id,
            min_iou_treshold=0.5,
            accept_equal_iou=True,
            sort_by_target_height=False,
        )
    elif method == "for_ai_net_coverage":
        matching_result = match_instances_for_ai_net_coverage(
            target,
            unique_target_ids,
            prediction,
            unique_prediction_ids,
            invalid_instance_id=invalid_instance_id,
            min_iou_treshold=None,
        )
    elif method == "tree_learn":
        matching_result = match_instances_tree_learn(
            target,
            unique_target_ids,
            prediction,
            unique_prediction_ids,
            invalid_instance_id=invalid_instance_id,
            min_iou_treshold=0.5,
        )
    else:
        raise ValueError(f"Invalid matching method: {method}.")

    matched_target_ids, matched_predicted_ids = matching_result

    return matched_target_ids, matched_predicted_ids


@njit(parallel=True)
def match_instances_iou(  # pylint: disable=too-many-locals, too-many-positional-arguments
    xyz: npt.NDArray,
    target: npt.NDArray[np.int64],
    unique_target_ids: npt.NDArray[np.int64],
    prediction: npt.NDArray[np.int64],
    unique_prediction_ids: npt.NDArray[np.int64],
    invalid_instance_id: int = -1,
    min_iou_treshold: Optional[float] = 0.5,
    accept_equal_iou: bool = False,
    sort_by_target_height: bool = False,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    r"""
    This method implements the instance matching methods proposed in the following works:

    - `Kirillov, Alexander, et al. "Panoptic segmentation." Proceedings of the IEEE/CVF Conference on Computer Vision \
       and Pattern Recognition. 2019. <https://doi.org/10.1109/CVPR.2019.00963>`__
      
      This method matches predicted and ground-truth instances if their IoU is striclty greater than 0.5, which results
      in an unambigous matching.

      To apply this method, use the following settings: 

      - :code:`min_iou_treshold` = 0.5
      - :code:`accept_equal_iou` = :code:`False`
      - :code:`sort_by_target_height` = :code:`False`

    - `Wielgosz, Maciej, et al. "Point2Tree (P2T)—Framework for Parameter Tuning of Semantic and Instance Segmentation \
      Used with Mobile Laser Scanning Data in Coniferous Forest." Remote Sensing 15.15 (2023): 3737. \
      <https://doi.org/10.3390/rs15153737>`__

      This method processes the ground-truth instances sorted according to their height. Starting with the highest
      ground-truth instance, each ground-truth instance is matched with the predicted instance with which it has the
      highest IoU. Predicted instances that were already matched to a ground-truth instance before, are excluded from
      the matching.

      To apply this method, use the following settings: 

      - :code:`min_iou_treshold` = :code:`None`
      - :code:`sort_by_target_height` = :code:`True`

    - `Puliti, Stefano, et al. "For-Instance: a UAV Laser Scanning Benchmark Dataset for Semantic and Instance
      Segmentation of Individual Trees." arXiv preprint arXiv:2309.01279 (2023). <https://arxiv.org/pdf/2309.01279>`__

      This method is based on the method proposed by Wielgosz et al. (2023) but additionally introduces the criterion
      that ground-truth and predicted instances must have an IoU of a least 0.5 to be matched.

      To apply this method, use the following settings: 

      - :code:`min_iou_treshold` = 0.5
      - :code:`accept_equal_iou` = :code:`True`
      - :code:`sort_by_target_height` = :code:`True`

    - `Xiang, Binbin, et al. "Automated Forest Inventory: Analysis of High-Density Airborne LiDAR Point Clouds with 3D
      Deep Learning." Remote Sensing of Environment 305 (2024): 114078. <https://doi.org/10.1016/j.rse.2024.114078>`__
      (with :code:`min_iou_treshold` = 0.5, :code:`accept_equal_iou` = :code:`True`, and
      :code:`sort_by_target_height` = :code:`False`)

      This method is similar to the method proposed by Kirillov et al., with the difference that ground-truth and
      predicted instances are also matched if their IoU is equal to 0.5.

      To apply this method, use the following settings: 

      - :code:`min_iou_treshold` = 0.5
      - :code:`accept_equal_iou` = :code:`True`
      - :code:`sort_by_target_height` = :code:`False`      

    - `Wielgosz, Maciej, et al. "SegmentAnyTree: A Sensor and Platform Agnostic Deep Learning Model for Tree \
      Segmentation Using Laser Scanning Data." Remote Sensing of Environment 313 (2024): 114367. \
      <https://doi.org/10.1016/j.rse.2024.114367>`__
      
      This method is the same as the method proposed by Kirillov et al.

    Args:
        xyz: Coordinates of all points.
        target: Ground truth instance ID for each point.
        unique_target_ids: Unique ground-truth instance IDs excluding :code:`invalid_instance_id`.
        prediction: Predicted instance ID for each point.
        unique_prediction_ids: Unique predicted instance IDs excluding :code:`invalid_instance_id`.
        invalid_instance_id: ID that is assigned to points not assigned to any instance. Defaults to -1.
        min_iou_treshold: IoU threshold for instance matching. If set to a value that is not :code:`None`,
            instances are only matched if their IoU is equal to (only if :code:`accept_equal_iou` is :code:`True`) or
            stricly greater than this threshold. Defaults to :code:`0.5`.
        accept_equal_iou: Whether matched pairs of instances should be accepted if their IoU is equal to
            :code:`min_iou_treshold`. Defaults to :code:`False`.
        sort_by_target_height: Whether the instance matching should process the target instances sorted according to
            their height. This corresponds to the matching approach proposed by Wielgosz et al. The processing order of
            the target instances is only relevant if the matching can be ambiguous, i.e. if matches with an IoU <= 0.5
            are accepted. Defaults to :code:`False`.

    Returns:
        Tuple of two arrays. The first contains the ID of the matched ground-truth instance for each predicted instance.
        The second contains the ID of the matched predicted instance for each ground-truth instance.

    Shape:
        - :code:`xyz`: :math:`(N, 3)`
        - :code:`target`: :math:`(N)`
        - :code:`unique_target_ids`: math:`(G)`
        - :code:`prediction`: :math:`(N)`
        - :code:`unique_prediction_ids`: math:`(P)`
        - Output: Tuple of two arrays. The first has shape :math:`(P)`, the second has shape :math:`(G)`.

        | where
        |
        | :math:`N = \text{ number of points}`
        | :math:`G = \text{ number of ground-truth instances}`
        | :math:`P = \text{ number of predicted instances}`
    """
    if min_iou_treshold is None:
        min_iou_treshold = -1

    matched_target_ids = np.full(len(unique_prediction_ids), fill_value=invalid_instance_id, dtype=np.int64)
    matched_predicted_ids = np.full(len(unique_target_ids), fill_value=invalid_instance_id, dtype=np.int64)

    if sort_by_target_height:
        instance_heights = np.zeros(len(unique_target_ids), dtype=np.float64)
        for target_id in unique_target_ids:
            z_vals = xyz[target == target_id, 2]
            instance_heights[target_id] = z_vals.max() - z_vals.min()

        sorting_indices = np.argsort(-1 * instance_heights)
        unique_target_ids = unique_target_ids[sorting_indices]

    for i in (
        range(len(unique_target_ids))
        if (accept_equal_iou and min_iou_treshold <= 0.5) or (min_iou_treshold < 0.5)
        else prange(len(unique_target_ids))
    ):
        target_id = unique_target_ids[i]
        predicted_instances_intersecting_with_target = prediction[target == target_id]
        predicted_instances_intersecting_with_target = predicted_instances_intersecting_with_target[
            predicted_instances_intersecting_with_target != invalid_instance_id
        ]

        if len(predicted_instances_intersecting_with_target) == 0:
            continue
        values = np.unique(predicted_instances_intersecting_with_target)

        if (accept_equal_iou and min_iou_treshold <= 0.5) or (min_iou_treshold < 0.5):
            available_for_matching = matched_target_ids[values] == invalid_instance_id

            values = values[available_for_matching]

            if len(values) == 0:
                continue
        counts = np.array([(predicted_instances_intersecting_with_target == x).sum() for x in values])

        predicted_id = values[np.argmax(counts)]
        intersection = np.logical_and(target == target_id, prediction == predicted_id).sum()
        union = np.logical_or(target == target_id, prediction == predicted_id).sum()
        iou = intersection / union

        if (accept_equal_iou and iou >= min_iou_treshold) or iou > min_iou_treshold:
            matched_target_ids[predicted_id] = target_id
            matched_predicted_ids[target_id] = predicted_id

    return matched_target_ids, matched_predicted_ids


def match_instances_tree_learn(  # pylint: disable=too-many-locals
    target: npt.NDArray[np.int64],
    unique_target_ids: npt.NDArray[np.int64],
    prediction: npt.NDArray[np.int64],
    unique_prediction_ids: npt.NDArray[np.int64],
    invalid_instance_id: int = -1,
    min_iou_treshold: Optional[float] = 0.5,
    accept_equal_iou: bool = False,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    r"""
    Instance matching method that is proposed in `Henrich, Jonathan, et al. "TreeLearn: A Deep Learning Method for
    Segmenting Individual Trees from Ground-Based LiDAR Forest Point Clouds." Ecological Informatics 84 (2024): 102888.
    <https://doi.org/10.1016/j.ecoinf.2024.102888>`__ The method uses Hungarian matching to match predicted and ground
    truth instances in such a way that the sum of the IoU scores of all matched instance pairs is maximized.
    Subsequently, matches with an IoU score less than or equal to :code:`min_iou_treshold` are discarded.

    Args:
        target: Ground truth instance ID for each point.
        unique_target_ids: Unique ground-truth instance IDs excluding :code:`invalid_instance_id`.
        prediction: Predicted instance ID for each point.
        unique_prediction_ids: Unique predicted instance IDs excluding :code:`invalid_instance_id`.
        invalid_instance_id: ID that is assigned to points not assigned to any instance. Defaults to -1.
        min_iou_treshold: IoU threshold for instance matching. If set to a value that is not :code:`None`,
            instances are only matched if their IoU is strictly greater than this threshold. Defaults to :code:`0.5`,
            which corresponds to the setting proposed by Henrich et al.
        accept_equal_iou: Whether matched pairs of instances should be accepted if their IoU is equal to
            :code:`min_iou_treshold`. Defaults to :code:`False`.

    Returns:
        Tuple of two arrays. The first contains the ID of the matched ground-truth instance for each predicted instance.
        The second contains the ID of the matched predicted instance for each ground-truth instance.

    Shape:
        - :code:`target`: :math:`(N)`
        - :code:`unique_target_ids`: math:`(G)`
        - :code:`prediction`: :math:`(N)`
        - :code:`unique_prediction_ids`: math:`(P)`
        - Output: Tuple of two arrays. The first has shape :math:`(P)`, the second has shape :math:`(G)`.

        | where
        |
        | :math:`N = \text{ number of points}`
        | :math:`G = \text{ number of ground-truth instances}`
        | :math:`P = \text{ number of predicted instances}`
    """
    if min_iou_treshold is None:
        min_iou_treshold = -1

    matched_target_ids = np.full(len(unique_prediction_ids), fill_value=invalid_instance_id, dtype=np.int64)
    matched_predicted_ids = np.full(len(unique_target_ids), fill_value=invalid_instance_id, dtype=np.int64)

    iou_matrix = np.zeros((len(unique_prediction_ids), len(unique_target_ids)))

    for predicted_id in unique_prediction_ids:
        target_instances_intersecting_with_prediction = np.unique(target[prediction == predicted_id])
        target_instances_intersecting_with_prediction = target_instances_intersecting_with_prediction[
            target_instances_intersecting_with_prediction != invalid_instance_id
        ]

        for target_id in target_instances_intersecting_with_prediction:
            intersection = np.logical_and(target == target_id, prediction == predicted_id).sum()
            union = np.logical_or(target == target_id, prediction == predicted_id).sum()
            iou = intersection / union

            iou_matrix[predicted_id, target_id] = iou

    # Hungarian matching between predicted instances and ground-truth instances
    matched_preds, matched_gts = scipy.optimize.linear_sum_assignment(iou_matrix, maximize=True)

    if min_iou_treshold > 0:
        if accept_equal_iou:
            mask_satisfies_match_condition = iou_matrix[matched_preds, matched_gts] >= min_iou_treshold
        else:
            mask_satisfies_match_condition = iou_matrix[matched_preds, matched_gts] > min_iou_treshold
        matched_preds = matched_preds[mask_satisfies_match_condition]
        matched_gts = matched_gts[mask_satisfies_match_condition]

    matched_target_ids[matched_preds] = matched_gts
    matched_predicted_ids[matched_gts] = matched_preds

    return matched_target_ids, matched_predicted_ids


@njit(parallel=True)
def match_instances_for_ai_net_coverage(  # pylint: disable=too-many-locals
    target: npt.NDArray[np.int64],
    unique_target_ids: npt.NDArray[np.int64],
    prediction: npt.NDArray[np.int64],
    unique_prediction_ids: npt.NDArray[np.int64],
    invalid_instance_id: int = -1,
    min_iou_treshold: Optional[float] = None,
    accept_equal_iou: bool = False,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    r"""
    Instance matching method that is proposed in `Xiang, Binbin, et al. "Automated Forest Inventory: Analysis of
    High-Density Airborne LiDAR Point Clouds with 3D Deep Learning." Remote Sensing of Environment 305 (2024): 114078.
    <https://doi.org/10.1016/j.rse.2024.114078>`__ for computing the coverage metric. The method matches each ground
    truth instance with the predicted instance with which it has the highest IoU. This means that predicted instances
    can be matched with multiple ground-truth instances. Such a matching approach is useful for the calculation of
    segmentation metrics (e.g., coverage) that should be independent from the instance detection rate.

    Args:
        target: Ground truth instance ID for each point.
        unique_target_ids: Unique ground-truth instance IDs excluding :code:`invalid_instance_id`.
        prediction: Predicted instance ID for each point.
        unique_prediction_ids: Unique predicted instance IDs excluding :code:`invalid_instance_id`.
        invalid_instance_id: ID that is assigned to points not assigned to any instance. Defaults to -1.
        min_iou_treshold: IoU threshold for instance matching. If set to a value that is not :code:`None`,
            instances are only matched if their IoU is strictly greater than this threshold. Defaults to :code:`None`,
            which corresponds to the setting proposed by Xiang et al.
        accept_equal_iou: Whether matched pairs of instances should be accepted if their IoU is equal to
            :code:`min_iou_treshold`. Defaults to :code:`False`.

    Returns:
        Tuple of two arrays. The first contains the ID of the matched ground-truth instance for each predicted instance.
        If a predicted instance is matched with multiple ground-truth instances, the ID of the matched ground-truth
        instance with which it has the highest IoU is returned. The second contains the ID of the matched predicted
        instance for each ground-truth instance.

    Shape:
        - :code:`target`: :math:`(N)`
        - :code:`unique_target_ids`: math:`(G)`
        - :code:`prediction`: :math:`(N)`
        - :code:`unique_prediction_ids`: math:`(P)`
        - Output: Tuple of two arrays. The first has shape :math:`(P)`, the second has shape :math:`(G)`.

        | where
        |
        | :math:`N = \text{ number of points}`
        | :math:`G = \text{ number of ground-truth instances}`
        | :math:`P = \text{ number of predicted instances}`
    """
    if min_iou_treshold is None:
        min_iou_treshold = -1

    ious = np.zeros(len(unique_prediction_ids), dtype=np.float64)
    matched_target_ids = np.full(len(unique_prediction_ids), fill_value=invalid_instance_id, dtype=np.int64)
    matched_predicted_ids = np.full(len(unique_target_ids), fill_value=invalid_instance_id, dtype=np.int64)

    for i in prange(len(unique_target_ids)):  # pylint: disable=not-an-iterable
        target_id = unique_target_ids[i]

        predicted_instances_intersecting_with_target = np.unique(prediction[target == target_id])
        predicted_instances_intersecting_with_target = predicted_instances_intersecting_with_target[
            predicted_instances_intersecting_with_target != invalid_instance_id
        ]
        if len(predicted_instances_intersecting_with_target) == 0:
            continue

        values = np.unique(predicted_instances_intersecting_with_target)
        counts = np.array([(predicted_instances_intersecting_with_target == x).sum() for x in values])
        predicted_id = values[np.argmax(counts)]

        intersection = np.logical_and(target == target_id, prediction == predicted_id).sum()
        union = np.logical_or(target == target_id, prediction == predicted_id).sum()
        iou = intersection / union

        if (accept_equal_iou and iou >= min_iou_treshold) or iou > min_iou_treshold:
            if iou > ious[predicted_id] and (
                min_iou_treshold > 0.5 or min_iou_treshold == 0.5 and not accept_equal_iou
            ):
                ious[predicted_id] = iou
                matched_target_ids[predicted_id] = target_id
            matched_predicted_ids[target_id] = predicted_id

    if (accept_equal_iou and min_iou_treshold <= 0.5) or (min_iou_treshold < 0.5):
        for i in prange(len(unique_prediction_ids)):  # pylint: disable=not-an-iterable
            predicted_id = unique_prediction_ids[i]
            target_instances_intersecting_with_prediction = np.unique(target[prediction == predicted_id])
            target_instances_intersecting_with_prediction = target_instances_intersecting_with_prediction[
                target_instances_intersecting_with_prediction != invalid_instance_id
            ]
            if len(target_instances_intersecting_with_prediction) == 0:
                continue

            values = np.unique(target_instances_intersecting_with_prediction)
            counts = np.array([(target_instances_intersecting_with_prediction == x).sum() for x in values])
            target_id = values[np.argmax(counts)]
            matched_target_ids[predicted_id] = target_id

    return matched_target_ids, matched_predicted_ids
