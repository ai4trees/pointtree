"""Methods for matching ground-truth instances and predicted instances."""

__all__ = ["match_instances"]

from typing import Dict, Literal, Optional, Tuple

import numpy as np
from pointtorch.metrics.instance_segmentation import match_instances as match_instances_pointtorch
import torch

from pointtree.operations import cloth_simulation_filtering, create_digital_terrain_model, distance_to_dtm
from pointtree.type_aliases import FloatArray, LongArray


def match_instances(  # pylint: disable=too-many-locals
    target: LongArray,
    prediction: LongArray,
    xyz: FloatArray,
    method: Literal[
        "panoptic_segmentation",
        "point2tree",
        "for_instance",
        "for_ai_net",
        "for_ai_net_coverage",
        "tree_learn",
    ],
    *,
    invalid_instance_id: int = -1,
    uncertain_instance_id: int = -2,
    min_tree_height_fp: float = 0.0,
    min_precision_fp: float = 0.0,
    labeled_mask: Optional[np.ndarray] = None,
) -> Tuple[LongArray, LongArray, Dict[str, LongArray]]:
    r"""
    This method implements the instance matching methods proposed in the following works:

    - :code:`panoptic_segmentation`: `Kirillov, Alexander, et al. "Panoptic segmentation." Proceedings of the IEEE/CVF \
      Conference on Computer Vision and Pattern Recognition. 2019. <https://doi.org/10.1109/CVPR.2019.00963>`__
      
      This method matches predicted and target instances if their IoU is striclty greater than 0.5, which results
      in an unambigous matching. This method is also used in `Wielgosz, Maciej, et al. "SegmentAnyTree: A Sensor and \
      Platform Agnostic Deep Learning Model for Tree Segmentation Using Laser Scanning Data." Remote Sensing of \
      Environment 313 (2024): 114367. <https://doi.org/10.1016/j.rse.2024.114367>`__

    - :code:`point2tree`: `Wielgosz, Maciej, et al. "Point2Tree (P2T)—Framework for Parameter Tuning of Semantic and \
      Instance Segmentation Used with Mobile Laser Scanning Data in Coniferous Forest." Remote Sensing 15.15 (2023): \
      3737. <https://doi.org/10.3390/rs15153737>`__

      This method processes the target instances sorted according to their height. Starting with the highest target
      instance, each target instance is matched with the predicted instance with which it has the highest IoU. Predicted
      instances that were already matched to a target instance before, are excluded from the matching.

    - :code:`for_instance`: `Puliti, Stefano, et al. "For-Instance: a UAV Laser Scanning Benchmark Dataset for \
      Semantic and Instance Segmentation of Individual Trees." arXiv preprint arXiv:2309.01279 (2023). \
      <https://arxiv.org/pdf/2309.01279>`__

      This method is based on the method proposed by Wielgosz et al. (2023) but additionally introduces the criterion
      that target and predicted instances must have an IoU of a least 0.5 to be matched.

    - :code:`for_ai_net`: `Xiang, Binbin, et al. "Automated Forest Inventory: Analysis of High-Density Airborne LiDAR \
      Point Clouds with 3D Deep Learning." Remote Sensing of Environment 305 (2024): 114078. \
      <https://doi.org/10.1016/j.rse.2024.114078>`__

      This method is similar to the method proposed by Kirillov et al., with the difference that target and predicted
      instances are also matched if their IoU is equal to 0.5.

    - :code:`for_ai_net_coverage`: `Xiang, Binbin, et al. "Automated Forest Inventory: Analysis of High-Density \
      Airborne LiDAR Point Clouds with 3D Deep Learning." Remote Sensing of Environment 305 (2024): 114078. \
      <https://doi.org/10.1016/j.rse.2024.114078>`__

      This method matches each target instance with the predicted instance with which it has the highest IoU. This
      means that predicted instances can be matched with multiple target instances. Such a matching approach is useful
      for the calculation of segmentation metrics (e.g., coverage) that should be independent from the instance
      detection rate.
  
    - :code:`tree_learn`: `Henrich, Jonathan, et al. "TreeLearn: A Deep Learning Method for Segmenting Individual Trees
      from Ground-Based LiDAR Forest Point Clouds." Ecological Informatics 84 (2024): 102888.
      <https://doi.org/10.1016/j.ecoinf.2024.102888>`__

      This method uses Hungarian matching to match predicted and target instances in such a way that the sum of the IoU
      scores of all matched instance pairs is maximized. Subsequently, matches with an IoU score less than o equal to
      0.5 are discarded.

    Args:
        target: Target instance ID for each point. Instance IDs must be integers forming a continuous range. The
            smallest instance ID in :code:`target` must be equal to the smallest instance ID in :code:`prediction`.
        prediction: Predicted instance ID for each point. Instance IDs must be integers forming a continuous range. The
            smallest instance ID in :code:`prediction` must be equal to the smallest instance ID in :code:`target`.
        xyz: Coordinates of all points.
        method: Instance matching method to be used.
        invalid_instance_id: ID that is used as label for points not assigned to any instance in :code:`target` and
            :code:`prediction`. In the returned instance matchings, the matched instance ID is set to
            :code:`invalid_instance_id` for target instances that were not matched with any predicted instance and for
            predicted instances that were not matched with any target instance and are considered as false positives
            according to :code:`min_tree_height_fp` and :code:`min_precision_fp`.
        uncertain_instance_id: ID that is used to mark predicted instances that were not matched with any target
            instance but are not counted as false positives according to :code:`min_tree_height_fp` or
            :code:`min_precision_fp`. Must be equal to or smaller than :code:`invalid_instance_id`.
        min_tree_height_fp: Minimum height an unmatched predicted tree instance must have in order to be counted as a
            false positive. The height of a tree is defined as the maximum distance between its points and a digital
            terrain model. If a predicted tree instance could not be matched with any target instance but its height is
            below :code:`min_tree_height_fp`, its matched instance ID is set to :code:`uncertain_instance_id`.
        min_precision_fp: Minimum percentage of points of an unmatched predicted instance that must be labeled in order
            to count the predicted instance as a false positive. If :code:`labeled_mask` is not :code:`None`, the points
            for which the mask is :code:`True` are considered as labeled points. If :code:`labeled_mask` is
            :code:`None`, all points that are labeled as instances are considered as labeled points (i.e., points
            labeled with :code:`invalid_instance_id` are considered unlabeled). If a predicted instance could not be
            matched with any target instance but its percentage of labeled points is below :code:`min_precision_fp`, its
            matched instance ID is set to :code:`uncertain_instance_id`.
        labeled_mask: Boolean mask indicating which points are labeled. This mask is used to mark false positive
            instances that mainly consist of unlabeled points.

    Returns: A tuple with the following elements:
        - :code:`matched_target_ids`: IDs of the matched target instance for each predicted instance. If the predicted
          instance is not matched to any target instance, its entry is set to either :code:`invalid_instance_id` (false
          positive) or :code:`uncertain_instance_id` (uncertain predicted instance).
        - :code:`matched_predicted_ids`: IDs of the matched predicted instance for each target instance. If the target
          instance is not matched to any predicted instance, its entry is set to :code:`invalid_instance_id`.
        - :code:`metrics`: Dictionary with the keys :code:`"tp"`, :code:`"fp"`, :code:`"fn"`. The values are
          tensors whose length is equal to the number of target instances and that contain the number of true positive,
          false positive, and false negative points between the matched instances. For target instances not matched to
          any prediction, the true and false posiitves are set to zero and the false negatives to the number of target
          points.

    Shape:
        - :code:`xyz`: :math:`(N, 3)`
        - :code:`target`: :math:`(N)`
        - :code:`prediction`: :math:`(N)`
        - Output:
            - :code:`matched_target_ids`: :math:`(P)`
            - :code:`matched_predicted_ids`: :math:`(T)`
            - :code:`metrics`: Dictionary whose values are tensors of length :math:`(T)`

        | where
        |
        | :math:`N` = number of points
        | :math:`P` = number of predicted instances
        | :math:`T` = number of target instances
    """

    matched_target_ids_torch, matched_predicted_ids_torch, segmentation_metrics_torch = match_instances_pointtorch(
        torch.from_numpy(target),
        torch.from_numpy(prediction),
        xyz=torch.from_numpy(xyz),
        method=method,
        invalid_instance_id=invalid_instance_id,
    )

    matched_target_ids = matched_target_ids_torch.numpy()
    matched_predicted_ids = matched_predicted_ids_torch.numpy()
    segmentation_metrics = {key: metric.numpy() for key, metric in segmentation_metrics_torch.items()}

    if len(matched_target_ids) > 0:
        start_instance_id = prediction[prediction != invalid_instance_id].min()

        dists_to_dtm = None
        if min_tree_height_fp > 0:
            terrain_classification = cloth_simulation_filtering(
                xyz, classification_threshold=0.5, resolution=0.5, rigidness=2
            )
            dtm_resolution = 0.25
            dtm, dtm_offset = create_digital_terrain_model(
                xyz[terrain_classification == 0],
                grid_resolution=dtm_resolution,
                k=400,
                p=1,
                voxel_size=0.05,
                num_workers=-1,
            )
            dists_to_dtm = distance_to_dtm(xyz, dtm, dtm_offset, dtm_resolution=dtm_resolution)

        if min_precision_fp <= 0 and min_tree_height_fp <= 0:
            return matched_target_ids, matched_predicted_ids, segmentation_metrics

        if min_precision_fp > 0 and labeled_mask is None:
            labeled_mask = target != invalid_instance_id

        for predicted_idx, matched_target_id in enumerate(matched_target_ids):
            if matched_target_id == invalid_instance_id:
                predicted_id = start_instance_id + predicted_idx
                if min_precision_fp > 0:
                    # count percentage of points belonging to labeled ground-truth instances
                    intersection = np.logical_and(  # type: ignore[arg-type]
                        labeled_mask, prediction == predicted_id
                    ).sum()
                    precision = intersection / (prediction == predicted_id).sum()
                else:
                    precision = 1.0

                tree_height = np.inf
                if dists_to_dtm is not None:
                    tree_height = max(0, dists_to_dtm[prediction == predicted_id].max())

                if precision < min_precision_fp or tree_height < min_tree_height_fp:
                    matched_target_ids[predicted_idx] = uncertain_instance_id

    return matched_target_ids, matched_predicted_ids, segmentation_metrics
