"""Computation of tree attributes from individual tree point clouds."""

__all__ = [
    "tree_attributes",
    "crown_base_height",
    "crown_volume",
    "crown_width",
    "tree_height",
    "tree_position",
    "under_branch_height",
    "stem_diameter",
    "stem_direction",
]


from typing import Any, Dict, List, Literal, Optional, Tuple

from circle_detection.operations import circumferential_completeness_index
import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA

from .stem_diameter import (
    estimate_stem_diameter,
    fit_circles_and_ellipses_to_stem_layers,
    select_best_stem_layer_combination,
    StemDiameterAllometricModel,
)


def tree_attributes(  # pylint: disable=too-many-arguments, too-many-branches, too-many-locals, too-many-positional-arguments
    tree_xyz: npt.NDArray,
    attributes: Optional[
        List[
            Literal[
                "crown_base_height",
                "crown_volume",
                "crown_width",
                "tree_height",
                "tree_position",
                "under_branch_height",
                "stem_diameter",
                "stem_direction",
            ]
        ]
    ] = None,
    classification: Optional[npt.NDArray] = None,
    ground_height: Optional[float] = None,
    stem_class_ids: Optional[List[int]] = None,
    branch_class_ids: Optional[List[int]] = None,
    leaf_class_ids: Optional[List[int]] = None,
    attribute_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    allometric_model: Optional[StemDiameterAllometricModel] = None,
) -> Dict[str, Any]:
    """
    Computes attributes for a single tree.

    Args:
        tree_xyz: Coordinates of all points belonging to the tree.
        attributes: Names of the attributes to compute. If :code:`None`, all supported attributes are computed.
        classification: Semantic class ID for each point in :code:`tree_xyz`. Used together with
            :code:`stem_class_ids`, :code:`branch_class_ids`, and :code:`leaf_class_ids` to restrict the points used
            to compute the stem, branch, and crown attributes, respectively. If :code:`None`, all points of
            :code:`tree_xyz` are used for every attribute.
        ground_height: Height of the ground surface underneath the tree, used to compute the tree height, crown
            base height, and under branch height. If :code:`None`, the minimum z-coordinate of :code:`tree_xyz` is
            used instead.
        stem_class_ids: Class IDs that identify stem points in :code:`classification`.
        branch_class_ids: Class IDs that identify branch points in :code:`classification`.
        leaf_class_ids: Class IDs that identify leaf points in :code:`classification`.
        attribute_kwargs: Additional keyword arguments to pass to the computation of individual attributes. The keys
            are attribute names (as in :code:`attributes`) and the values are dictionaries of keyword arguments to
            pass to the respective attribute computation function (e.g., :code:`{"stem_diameter": {"num_layers": 8,
            "target_heights": np.array([1.3, 2.0])}}`). If :code:`None`, or if an attribute's name is not contained
            in :code:`attribute_kwargs`, the respective function's default arguments are used.
        allometric_model: Fitted allometric model (see :code:`pointtree.tree_attributes.stem_diameter.\
            StemDiameterAllometricModel`) used as a fallback to estimate the stem diameter at breast height (1.3 m
            above the ground) from the tree height and crown width, if 1.3 m is among the requested target heights
            and the diameter at that height cannot be estimated using the circle / ellipse fitting approach (see
            :code:`stem_diameter`). Diameters at other target heights are not affected. If :code:`None`, no
            fallback is used and the stem diameter remains :code:`NaN` in that case.

    Returns:
        Dictionary mapping the name of each computed attribute to its value. Since the stem diameter can be
        estimated at multiple heights, the diameter estimated at each target height is stored under the key
        :code:`"stem_diameter_<target height>"`, where the target height is rounded to two decimal places (e.g.,
        :code:`"stem_diameter_1.3"`). If :code:`"stem_diameter"` is computed, the dictionary additionally contains the
        keys :code:`"stem_diameter_layer_completeness"` and :code:`"stem_diameter_layer_std"`, which hold, respectively,
        the circumferential completeness indices and the standard deviation of the diameters of the layers that
        were used to estimate the stem diameter (see :code:`stem_diameter`). The diameter at breast height (1.3 m
        above the ground) is set to the prediction of :code:`allometric_model`, if one is provided, if it could
        not be estimated using the circle / ellipse fitting approach. Diameters at other target heights, and the
        diameter at breast height if no :code:`allometric_model` is provided, are :code:`NaN` if they could not be
        estimated using the circle / ellipse fitting approach.
    """

    tree_attributes_dict: Dict[str, Any] = {}
    attribute_kwargs = attribute_kwargs if attribute_kwargs is not None else {}

    stem_xyz = tree_xyz
    branch_xyz = tree_xyz
    crown_xyz = tree_xyz
    if classification is not None and stem_class_ids is not None:
        stem_xyz = tree_xyz[np.isin(classification, stem_class_ids)]
    if classification is not None and branch_class_ids is not None:
        branch_xyz = tree_xyz[np.isin(classification, branch_class_ids)]
    if classification is not None and leaf_class_ids is not None:
        crown_xyz = tree_xyz[np.isin(classification, leaf_class_ids)]

    ground_height = ground_height if ground_height is not None else float(tree_xyz[:, 2].min())

    if attributes is None or "crown_volume" in attributes:
        tree_attributes_dict["crown_volume"] = crown_volume(crown_xyz, **attribute_kwargs.get("crown_volume", {}))

    if attributes is None or "crown_width" in attributes:
        tree_attributes_dict["crown_width"] = crown_width(crown_xyz, **attribute_kwargs.get("crown_width", {}))

    if attributes is None or "tree_height" in attributes:
        tree_attributes_dict["tree_height"] = tree_height(
            tree_xyz, ground_height=ground_height, **attribute_kwargs.get("tree_height", {})
        )

    if attributes is None or "tree_position" in attributes:
        tree_attributes_dict["tree_position"] = tree_position(
            stem_xyz, tree_xyz, **attribute_kwargs.get("tree_position", {})
        )

    if attributes is None or "crown_base_height" in attributes:
        tree_attributes_dict["crown_base_height"] = crown_base_height(
            crown_xyz, ground_height, **attribute_kwargs.get("crown_base_height", {})
        )

    if attributes is None or "under_branch_height" in attributes:
        tree_attributes_dict["under_branch_height"] = under_branch_height(
            branch_xyz, ground_height, **attribute_kwargs.get("under_branch_height", {})
        )

    if attributes is None or "stem_diameter" in attributes:
        stem_diameter_kwargs = dict(attribute_kwargs.get("stem_diameter", {}))
        target_heights = stem_diameter_kwargs.pop("target_heights", None)
        if target_heights is None:
            target_heights = np.array([1.3])
        diameters, completeness_indices, layer_diameter_std = stem_diameter(
            stem_xyz, ground_height=ground_height, target_heights=target_heights, **stem_diameter_kwargs
        )

        missing_breast_height_diameter = np.isnan(diameters) & np.isclose(target_heights, 1.3)
        if allometric_model is not None and missing_breast_height_diameter.any():
            fallback_tree_height = tree_attributes_dict.get("tree_height")
            if fallback_tree_height is None:
                fallback_tree_height = tree_height(
                    tree_xyz, ground_height=ground_height, **attribute_kwargs.get("tree_height", {})
                )
            fallback_crown_width = tree_attributes_dict.get("crown_width")
            if fallback_crown_width is None:
                fallback_crown_width = crown_width(crown_xyz, **attribute_kwargs.get("crown_width", {}))
            fallback_diameter = allometric_model.predict(
                np.array([fallback_tree_height]), np.array([fallback_crown_width])
            )[0]
            diameters = diameters.copy()
            diameters[missing_breast_height_diameter] = fallback_diameter

        for height, diameter in zip(target_heights, diameters):
            tree_attributes_dict[f"stem_diameter_{round(float(height), 2)}"] = float(diameter)
        tree_attributes_dict["stem_diameter_layer_completeness"] = completeness_indices
        tree_attributes_dict["stem_diameter_layer_std"] = layer_diameter_std

    if attributes is None or "stem_direction" in attributes:
        tree_attributes_dict["stem_direction"] = stem_direction(stem_xyz, **attribute_kwargs.get("stem_direction", {}))

    return tree_attributes_dict


def crown_base_height(crown_xyz: npt.NDArray, ground_height: float) -> float:
    """
    Computes the height of the crown base above the ground. The crown base height is estimated as the height of the
    of the lowest leaf point above the ground.

    Args:
        crown_xyz: Coordinates of the points belonging to the tree crown.
        ground_height: Height of the ground surface underneath the tree.

    Returns:
        Height of the crown base above the ground. :code:`NaN` if :code:`crown_xyz` is empty.
    """

    if len(crown_xyz) == 0:
        return float("nan")

    crown_base = float(crown_xyz[:, 2].min()) - ground_height

    return max(crown_base, 0.0)


def crown_volume(crown_xyz: npt.NDArray, voxel_size: float = 0.5) -> float:
    """
    Computes the crown volume by summing the volumes of the 2.5D voxel columns spanned by the crown points.

    Args:
        crown_xyz: Coordinates of the points belonging to the tree crown.
        voxel_size: Edge length of the (square) voxel columns in meters.

    Returns:
        Crown volume in cubic meters. :code:`0.0` if :code:`crown_xyz` is empty.
    """

    if len(crown_xyz) == 0:
        return 0.0

    bins = np.floor(crown_xyz / voxel_size).astype(np.int64)
    volume = 0.0

    sorting_indices = np.lexsort((bins[:, 1], bins[:, 0]))
    bins_sorted = bins[sorting_indices]
    z_sorted = crown_xyz[sorting_indices, 2]

    # split into contiguous (bin_x, bin_y) columns
    column_keys = bins_sorted[:, :2]
    column_boundaries = np.any(column_keys[1:] != column_keys[:-1], axis=1)
    split_idx = np.flatnonzero(column_boundaries) + 1

    for column in np.split(z_sorted, split_idx):
        volume += voxel_size * voxel_size * (column.max() - column.min())

    return float(volume)


def crown_width(crown_xyz: npt.NDArray) -> float:
    """
    Computes the crown width as the largest extent of the crown points along the x- or y-axis.

    Args:
        crown_xyz: Coordinates of the points belonging to the tree crown.

    Returns:
        Crown width in meters. :code:`0.0` if :code:`crown_xyz` is empty.
    """

    if len(crown_xyz) == 0:
        return 0.0

    return (crown_xyz[:, :2].max(axis=0) - crown_xyz[:, :2].min(axis=0)).max()


def tree_height(xyz: npt.NDArray, ground_height: Optional[float] = None) -> float:
    """
    Computes the tree height.

    Args:
        xyz: Coordinates of all points belonging to the tree.
        ground_height: Height of the ground surface underneath the tree. If provided, the tree height is computed
            as the difference between the maximum z-coordinate of :code:`xyz` and :code:`ground_height`. If
            :code:`None`, the tree height is computed as the difference between the maximum and minimum
            z-coordinate of :code:`xyz`.

    Returns:
        Tree height in meters. :code:`0.0` if :code:`xyz` is empty.
    """

    if len(xyz) == 0:
        return 0.0

    if ground_height is not None:
        return max(xyz[:, 2].max() - ground_height, 0.0)

    return xyz[:, 2].max() - xyz[:, 2].min()


def tree_position(stem_xyz: npt.NDArray, tree_xyz: npt.NDArray) -> Tuple[float, float]:
    """
    Computes the position of a tree as the mean x- and y-coordinate of its stem points.

    Args:
        stem_xyz: Coordinates of the points belonging to the tree stem.
        tree_xyz: Coordinates of the points belonging to the tree.

    Returns:
        X- and y-coordinate of the tree position. :code:`(NaN, NaN)` if :code:`stem_xyz` and :code:`tree_xyz` are empty.
    """

    if len(stem_xyz) == 0 and len(tree_xyz) == 0:
        return float("nan"), float("nan")

    if len(stem_xyz) > 0:
        mean_xy = stem_xyz[:, :2].mean(axis=0)
    else:
        mean_xy = tree_xyz[:, :2].mean(axis=0)

    return float(mean_xy[0]), float(mean_xy[1])


def stem_diameter(  # pylint: disable=too-many-locals, too-many-arguments, too-many-positional-arguments
    stem_xyz: npt.NDArray,
    ground_height: Optional[float] = None,
    target_heights: Optional[npt.NDArray] = None,
    circle_fitting_method: Literal["ransac", "m-estimator"] = "ransac",
    num_layers: int = 15,
    layer_height: float = 0.225,
    layer_overlap: float = 0.025,
    layer_start: float = 1.0,
    std_num_layers: int = 6,
    max_std_diameter: float = 0.04,
    bandwidth: float = 0.01,
    min_points: int = 15,
    min_fitting_score: float = 100.0,
    min_stem_diameter: float = 0.02,
    max_stem_diameter: float = 1.0,
    min_completeness_idx: Optional[float] = 0.3,
    fit_ellipses: bool = False,
    ellipse_filter_threshold: float = 0.6,
    gam_max_radius_diff: Optional[float] = 0.3,
    random_seed: Optional[int] = None,
) -> Tuple[npt.NDArray, npt.NDArray, float]:
    r"""
    Estimates the stem diameter at the target height using the circle / ellipse fitting approach of the treeX algorithm
    (see :code:`pointtree.instance_segmentation.TreeXAlgorithm`). :code:`num_layers` horizontal layers of height
    :code:`layer_height` are extracted from the stem points, starting at :code:`layer_start` meters above the
    ground and overlapping adjacent layers by :code:`layer_overlap`. A circle (or, if :code:`fit_ellipses` is
    :code:`True`, alternatively an ellipse) is fitted to the points of each layer. From all combinations of
    :code:`std_num_layers` layers with a valid circle fit, the combination with the lowest standard deviation of the
    fitted diameters is selected, provided that this standard deviation does not exceed :code:`max_std_diameter`. If
    no such combination of circles exists, the same selection is repeated using the fitted ellipses (if
    :code:`fit_ellipses` is :code:`True`). For each layer of the selected combination, the stem diameter is refined
    by fitting a generalized additive model (GAM) to the points of that layer, using the center of the respective
    circle or ellipse to normalize the points (if the GAM fit is invalid, the diameter of the circle or ellipse is
    used instead). Finally, a linear model is fitted to the diameters of the selected layers to predict the stem
    diameter as a function of the height above the ground, and the predictions of this model for :code:`target_heights`
    are returned as the estimated diameters.

    Args:
        stem_xyz: Coordinates of the points belonging to the tree stem.
        ground_height: Height of the ground surface underneath the tree. If :code:`None`, the minimum z-coordinate
            of :code:`stem_xyz` is used instead.
        target_heights: Heights above the ground at which the stem diameter is to be estimated. Defaults to
            :code:`None`, which means that the stem diameter at 1.3 m above the ground is estimated.
        circle_fitting_method: Circle fitting method to use: :code:`"ransac"` or :code:`"m-estimator"`.
        num_layers: Number of horizontal layers used for the circle / ellipse fitting.
        layer_height: Height of the horizontal layers used for circle / ellipse fitting.
        layer_overlap: Overlap between adjacent horizontal layers used for circle / ellipse fitting.
        layer_start: Height above the ground at which the lowest layer used for circle / ellipse fitting starts.
        std_num_layers: Number of horizontal layers to consider when selecting the combination of layers with the
            lowest standard deviation of the fitted diameters.
        max_std_diameter: Maximum standard deviation of the fitted diameters within a combination of
            :code:`std_num_layers` layers for that combination to be considered valid.
        bandwidth: Bandwidth for circle fitting. It is used in the calculation of the circle fitting score and, for
            the M-estimator method, also for kernel density estimation.
        min_points: Minimum number of points that a horizontal layer must contain in order for a circle / ellipse to
            be fitted to it.
        min_fitting_score: Minimum fitting score that circles must achieve in the circle fitting to be considered
            valid. Only used when :code:`circle_fitting_method` is set to :code:`"ransac"`.
        min_stem_diameter: Minimum circle / ellipse diameter to be considered a valid fit.
        max_stem_diameter: Maximum circle / ellipse diameter to be considered a valid fit.
        min_completeness_idx: Minimum circumferential completeness index that circles must have to be considered
            valid. If :code:`None`, the circumferential completeness index is not used for filtering.
        fit_ellipses: Whether ellipses should additionally be fitted to the layers and used as a fallback if no
            valid combination of circles is found.
        ellipse_filter_threshold: Ellipses are only kept if the ratio of the radius along the semi-minor axis to the
            radius along the semi-major axis is greater than or equal to this threshold. Only used when
            :code:`fit_ellipses` is :code:`True`.
        gam_max_radius_diff: If the difference between the minimum and the maximum of the radii predicted by the GAM
            is greater than this value, the GAM fit is considered invalid and the diameter of the fitted circle or
            ellipse is used instead. If :code:`None`, the GAM fit is never considered invalid based on this
            criterion.
        random_seed: Seed for the random number generator used for circle fitting and for the small random offset
            added to the point radii before fitting the GAM (to avoid perfect separation). If :code:`None`, the
            random number generator is not seeded.

    Returns:
        :Tuple of three elements:
            - Estimated stem diameters at the target heights. :code:`NaN` if the diameter could not be estimated,
              e.g., because :code:`stem_xyz` does not contain enough points in at least :code:`std_num_layers`
              valid layers.
            - Circumferential completeness indices of the circles fitted to the :code:`std_num_layers` layers that
              were used to estimate the stem diameter. :code:`NaN` for a layer if the diameter could not be
              estimated, or if the ellipse fallback was used for the respective layer combination instead of
              circles (in which case no circumferential completeness index is available).
            - Standard deviation of the diameters of the :code:`std_num_layers` layers that were used to estimate
              the stem diameter (see :code:`select_best_stem_layer_combination`). :code:`NaN` if the diameter could
              not be estimated.

    Shape:
        - Output: :math:`(T)`, :math:`(L)`, scalar

        | where
        |
        | :math:`T` = number of target heights
        | :math:`L` = number of layers used to estimate the stem diameters
    """
    if target_heights is None:
        target_heights = np.array([1.3])

    no_completeness_indices = np.full(std_num_layers, fill_value=np.nan)

    if len(stem_xyz) < min_points:
        return np.full_like(target_heights, fill_value=np.nan), no_completeness_indices, float("nan")

    if ground_height is not None:
        height_above_ground = stem_xyz[:, 2] - ground_height
    else:
        height_above_ground = stem_xyz[:, 2] - stem_xyz[:, 2].min()

    layer_starts = layer_start + np.arange(num_layers) * (layer_height - layer_overlap)
    layer_ends = layer_starts + layer_height
    layer_heights = layer_starts + layer_height / 2

    xy_batches = []
    batch_lengths = np.zeros(num_layers, dtype=np.int64)
    for layer in range(num_layers):
        mask = (height_above_ground >= layer_starts[layer]) & (height_above_ground < layer_ends[layer])
        if mask.sum() >= min_points:
            xy_batches.append(stem_xyz[mask, :2])
            batch_lengths[layer] = mask.sum()

    if len(xy_batches) == 0:
        return np.full_like(target_heights, fill_value=np.nan), no_completeness_indices, float("nan")

    stem_layer_xy = np.concatenate(xy_batches, axis=0).astype(np.float64)
    stem_layer_xy = np.asfortranarray(stem_layer_xy)

    random_generator = np.random.default_rng(seed=random_seed)

    layer_circles, layer_ellipses = fit_circles_and_ellipses_to_stem_layers(
        stem_layer_xy,
        batch_lengths,
        circle_fitting_method=circle_fitting_method,
        bandwidth=bandwidth,
        min_fitting_score=min_fitting_score,
        min_stem_diameter=min_stem_diameter,
        max_stem_diameter=max_stem_diameter,
        min_completeness_idx=min_completeness_idx,
        fit_ellipses=fit_ellipses,
        ellipse_filter_threshold=ellipse_filter_threshold,
        seed=random_seed,
    )

    existing_circle_layers = np.flatnonzero(layer_circles[:, 2] != -1)
    best_combination, layer_diameter_std = select_best_stem_layer_combination(
        existing_circle_layers, layer_circles[:, 2] * 2, std_num_layers, max_std_diameter
    )
    use_circles = True

    if best_combination is None and fit_ellipses:
        existing_ellipse_layers = np.flatnonzero(layer_ellipses[:, 2] != -1)
        best_combination, layer_diameter_std = select_best_stem_layer_combination(
            existing_ellipse_layers, layer_ellipses[:, 2:4].sum(axis=-1), std_num_layers, max_std_diameter
        )
        use_circles = False

    if best_combination is None:
        return np.full_like(target_heights, fill_value=np.nan), no_completeness_indices, float("nan")

    circles_or_ellipses = layer_circles[best_combination] if use_circles else layer_ellipses[best_combination]
    centers = circles_or_ellipses[:, :2]
    fallback_diameters = circles_or_ellipses[:, 2] * 2 if use_circles else circles_or_ellipses[:, 2:4].sum(axis=-1)
    combination_heights = layer_heights[best_combination]

    batch_starts = np.cumsum(np.concatenate((np.array([0], dtype=np.int64), batch_lengths)))[:-1]
    combination_batch_lengths = batch_lengths[best_combination]
    combination_layer_xy = np.concatenate(
        [stem_layer_xy[batch_starts[layer] : batch_starts[layer] + batch_lengths[layer]] for layer in best_combination],
        axis=0,
    )

    if use_circles:
        completeness_indices = circumferential_completeness_index(
            np.ascontiguousarray(circles_or_ellipses[:, :3]),
            combination_layer_xy,
            num_regions=int(365 / 5),
            max_dist=bandwidth,
            batch_lengths_circles=np.ones(std_num_layers, dtype=np.int64),
            batch_lengths_xy=combination_batch_lengths,
        )
    else:
        completeness_indices = no_completeness_indices

    diameters, _, _, _ = estimate_stem_diameter(
        combination_layer_xy,
        combination_batch_lengths,
        centers,
        fallback_diameters,
        combination_heights,
        target_heights,
        gam_max_radius_diff,
        random_generator,
    )

    return diameters, completeness_indices, layer_diameter_std


def stem_direction(stem_xyz: npt.NDArray) -> npt.NDArray:
    """
    Computes the dominant 3D stem direction using principal component analysis (PCA).

    Args:
        stem_xyz: Coordinates of the points belonging to the tree stem.

    Returns:
        Unit vector describing the dominant stem direction, oriented so that its z-component is non-negative. An
        array of NaN values if :code:`stem_xyz` contains fewer than two points.
    """

    if len(stem_xyz) < 2:
        return np.array([np.nan, np.nan, np.nan])

    direction = PCA(n_components=1).fit(stem_xyz).components_[0]

    if direction[2] < 0:
        direction = -direction

    return direction


def under_branch_height(branch_xyz: npt.NDArray, ground_height: float) -> float:
    """
    Computes the height of the first (i.e., lowest) branching point above the ground, i.e., the length of the
    branch-free section of the stem. This is only computed if there is at least one branch point.

    Args:
        branch_xyz: Coordinates of the points belonging to the tree's branches.
        ground_height: Height of the ground surface underneath the tree.

    Returns:
        Height of the first branching point above the ground. :code:`NaN` if it cannot be determined.
    """

    if len(branch_xyz) == 0:
        return float("nan")

    return max(float(branch_xyz[:, 2].min()) - ground_height, 0.0)
