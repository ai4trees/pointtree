"""Region-growing-based tree instance segmentation algorithm."""  # pylint: disable=too-many-lines

__all__ = ["TreeXAlgorithm"]

import itertools
import multiprocessing
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from circle_detection import MEstimator, Ransac
import numpy as np
import numpy.typing as npt
from pointtorch import PointCloud
from pointtorch.operations.numpy import voxel_downsampling, make_labels_consecutive
from pygam import LinearGAM, s
import rasterio
from rasterio.transform import from_origin
from sklearn.cluster import DBSCAN

from pointtree.type_aliases import BoolArray, FloatArray, LongArray
from pointtree.evaluation import Profiler
from pointtree.operations import (
    create_digital_terrain_model,
    cloth_simulation_filtering,
    distance_to_dtm,
    fit_ellipse,
    estimate_with_linear_model,
    polygon_area,
)
from pointtree._tree_x_algorithm_cpp import (  # type: ignore[import-untyped] # pylint: disable=import-error, no-name-in-module
    segment_tree_crowns as segment_tree_crowns_cpp,
    collect_inputs_stem_layers_fitting as collect_inputs_stem_layers_fitting_cpp,
    collect_inputs_stem_layers_refined_fitting as collect_inputs_stem_layers_refined_fitting_cpp,
)
from pointtree.visualization import plot_fitted_shape

from ._instance_segmentation_algorithm import InstanceSegmentationAlgorithm
from .filters import (
    filter_instances_intensity,
    filter_instances_min_points,
    filter_instances_pca,
    filter_instances_vertical_extent,
)


class TreeXAlgorithm(InstanceSegmentationAlgorithm):  # pylint: disable=too-many-instance-attributes
    r"""
    Revised version of the tree instance segmentation algorithm originally introduced in the following papers:

    - `Tockner, Andreas et al. "Automatic Tree Crown Segmentation Using Dense Forest Point Clouds from Personal Laser \
      Scanning (PLS)." International Journal of Applied Earth Observation and Geoinformation 114 (2022): 103025. \
      <https://doi.org/10.1016/j.jag.2022.103025>`__
    - `Gollob, Christoph et al. "Forest Inventory with Long Range and High-Speed Personal \
      Laser Scanning (PLS) and Simultaneous Localization and Mapping (SLAM) technology." Remote Sensing 12.9 (2020): \
      1509. <https://doi.org/10.3390/rs12091509>`__.

    Args:
        invalid_tree_id: ID that is assigned to points that do not belong to any tree instance. Must either be zero or
            a negative number.
        num_workers: Number of workers to use for parallel processing. If set to :code:`-1`, all CPU threads are used.
        visualization_folder: Path of a directory in which to store visualizations of intermediate results of the
            algorithm. If set to :code:`None`, no visualizations are created.
        random_seed: Random seed to for reproducibility of random processes. It should be noted that even with the seed
            set, the algorithm is not completely deterministic, as some of the dependencies cannot be configured
            accordingly.

    The algorithm comprises the following steps:

    .. rubric:: 1. Terrain Classification Using the CSF Algorithm

    In the first step, the algorithm detects terrain points using the Cloth Simulation Filtering (CSF) algorithm \
    proposed in `Zhang, Wuming, et al. "An Easy-to-Use Airborne LiDAR Data Filtering Method Based on Cloth \
    Simulation." Remote Sensing 8.6 (2016): 501 <https://doi.org/10.3390/rs8060501>`__.

    Parameters:
        csf_terrain_classification_threshold: Maximum height above the cloth a point can have in order
            to be classified as terrain point (in meters). All points whose distance to the cloth is equal or below this
            threshold are classified as terrain points.
        csf_tree_classification_threshold: Minimum height above the cloth a point must have in order to be classified as
            tree point (in meters). All points whose distance to the cloth is equal or larger than this threshold are
            classified as tree points.
        csf_correct_steep_slope: Whether to include a post-processing step in the CSF algorithm that handles steep
            slopes.
        csf_iterations: Maximum number of iterations.
        csf_resolution: Resolution of the cloth grid (in meters).
        csf_rigidness: Rigidness of the cloth (the three levels :code:`1`, :code:`2`, and :code:`3` are available, where
            :code:`1` is the lowest and :code:`3` the highest rigidness).

    .. rubric:: 2. Construction of a Digital Terrain Model

    In the next step, a rasterized digital terrain model (DTM) is constructed from the terrain points identified in the
    previous step. For this purpose, a grid of regularly arranged DTM points is created and the height of the :math:`k`
    closest terrain points is interpolated to obtain the height of each DTM point on the grid. In the interpolation, the
    terrain height :math:`h(q)` at grid position :math:`q` is computed using the following formula:

    .. math::
        h(q) = \frac{1}{\sum_{p \in \mathcal{N}(q, k)} w(q, p)} \cdot \sum_{p \in \mathcal{N}(q, k)} p_z \cdot w(q, p)
    
    where :math:`\mathcal{N}(q, k)` is the set of the :math:`k` terrain points closest to grid position :math:`q`,
    :math:`p_z` is the z-coordinate of point :math:`p`, and :math:`w` is an inverse-distance weighting function with a
    hyperparameter :math:`c`:

    .. math::
        w(q, p) = \frac{1}{||p_{xy} - q_{xy}||^c}
    
    Before constructing the DTM, the terrain points are downsampled using voxel-based subsampling.

    Parameters:
        dtm_k: Number of terrain points between which interpolation is performed to obtain the terrain height of a DTM
            point.
        dtm_power: Power :math:`c` for inverse-distance weighting in the interpolation of terrain points.
        dtm_resolution: Resolution of the DTM grid (in meters).
        dtm_voxel_size: Voxel size with which the terrain points are downsampled before the DTM is created (in meters).

    .. rubric:: 3. Detection of Tree Stems

    The aim of this step is to identify clusters of points that represent individual tree stems, i.e., each stem should
    be represented by a single cluster. For this purpose, a horizontal layer is extracted from the point cloud that
    contains all points within a certain height range above the terrain (the height range is defined by
    :code:`stem_search_min_z` and :code:`stem_search_max_z`). This layer should be chosen so that it contains all tree
    stems and as few other objects as possible. The points within this slice are downsampled using voxel-based
    subsampling and then clustered in 2D using the DBSCAN algorithm proposed in
    `Ester, Martin, et al. "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise." \
    KDD. Vol. 96. No. 34, pp. 226-231. 1996. <https://dl.acm.org/doi/10.5555/3001460.3001507>`__ 
    The reasoning behind this is that tree stems are usually vertical structures that form dense clusters of points when
    projected onto the XY plane. However, when applying the DBSCAN algorithm in 2D, stems that are located close to each
    other may be assigned into a single cluster. To address such cases, an additional 3D DBSCAN clustering is applied to
    the points within each cluster to further split the clusters.

    After the clustering, the following filtering rules are applied to the clusters to filter out false positive
    clusters that do not represent tree stems:

    1. Clusters with less than :code:`stem_search_min_cluster_points` points are discarded.
    2. Clusters whose extent in the z-direction (i.e., the height difference between the highest and the lowest point in
       the cluster) is less than :code:`stem_search_min_cluster_height` are discarded.
    3. If reflection intensity values are provided in the input, the 80% quantile of the reflection intensities of the
       points in a cluster is calculated. Clusters that have an 80% quantile of intensities equal to or smaller than
       :code:`stem_search_min_cluster_intensity` are discarded.
    4. A principal component analysis is performed on the points in each cluster. Clusters are discarded if the first
       principal component explains less than :code:`stem_search_pc1_min_explained_variance` of the variance or the
       angle between the z-axis and the first principal component is greater than :code:`stem_search_max_inclination`.
    5. From the remaining clusters, :code:`stem_search_circle_fitting_num_layers` horizontal layers are extracted. Each
       layer has a height of :code:`stem_search_circle_fitting_layer_height` and overlaps with adjacent layers by
       :code:`stem_search_circle_fitting_layer_overlap`. Within each layer, a circle and an ellipse are fitted to the
       contained points. For all possible combinations of :code:`stem_search_circle_fitting_std_num_layers` layers, the
       standard deviation of the fitted circle diameters is computed. If any combination yields a standard deviation
       less than or equal to :code:`stem_search_circle_fitting_max_std_diameter`, the cluster is retained. If no such
       combination is found, the same procedure is repeated using the diameters of the fitted ellipses. If none of these
       combinations satisfy the diameter standard deviation threshold, the cluster is discarded. Additionally, if
       :code:`stem_search_circle_fitting_max_std_position` is not :code:`None`, the filtering also requires that the
       standard deviation of the x- and y-coordinates of the circle or ellipse centers of the layer combination does not
       exceed the given threshold in order for the cluster to be kept. If :code:`stem_search_ellipse_fitting` is set to
       :code:`False`, ellipse fitting is skipped, and only the fitted circles are used for filtering.
       If :code:`stem_search_refined_circle_fitting` is set to :code:`True`, a multi-stage fitting process is applied
       for circle and ellipse fitting: Initially, circles / ellipses are fitted to the layers extracted from the point
       clusters, which lack full point density due to the initial downsampling (this initial fitting step is also
       performed when :code:`stem_search_refined_circle_fitting` is :code:`False`). Subsequently, each circle or ellipse
       is re-fitted using the points from a clipping region around the outline of the initially fitted circle / ellipse.
       A buffer region is created around this outline, with the buffer width determined as follows: if the preliminary
       circle / ellipse diameter is less than or equal to :code:`stem_search_circle_fitting_switch_buffer_threshold`,
       the width is set to :code:`stem_search_circle_fitting_small_buffer_width`; otherwise, it is set to
       :code:`stem_search_circle_fitting_large_buffer_width`. All points from the full-resoultion point cloud that lie
       within the buffer area are collected, and the circle or ellipse is re-fitted using these points.

    Parameters:
        stem_search_min_z: Height above the terrain at which the horizontal layer begins that is considered for stem
            detection (in meters).
        stem_search_max_z: Height above the terrain at which the horizontal layer ends that is considered for stem
            detection (in meters).
        stem_search_voxel_size: Voxel size with which the points are downsampled before performing the stem detection
            (in meters).
        stem_search_dbscan_2d_eps: Parameter :math:`\epsilon` of the DBSCAN algorithm for the initial clustering in 2D.
            The parameter defines the radius of the circular neighborhood that is used to determine the number of
            neighbors for a given point (in meters).
        stem_search_dbscan_2d_min_points: Parameter :math:`MinPnts` of the DBSCAN algorithm for the initial clustering
            in 2D. The parameter defines the number of neighbors a given point must have in order to be considered as a
            core point. All neighbors of a core point are added to the clusters and then checked whether they are
            core points themselves.
        stem_search_dbscan_3d_eps: Parameter :math:`\epsilon` of the DBSCAN algorithm for the clustering in 3D (in
            meters).
        stem_search_dbscan_3d_min_points: Parameter :math:`MinPnts` of the DBSCAN algorithm for the clustering in 3D.
        stem_search_min_cluster_points: Minimum number of points a cluster must contain in order not to be discarded.
        stem_search_min_cluster_height: Minimum extent in the z-direction (i.e., the height difference between the
            highest and the lowest point in the cluster) a cluster must have in order not to be discarded (in meters).
        stem_search_min_cluster_intensity: Threshold for filtering of clusters based on reflection intensity values.
            Clusters are discarded if the 80 % percentile of the reflection intensities of the points in the cluster is
            below the given threshold. If no reflection intensity values are input to the algorithm, the intensity-based
            filtering is skipped.
        stem_search_pc1_min_explained_variance: Minimum percentage of variance that the first principal
            component of a cluster must explain in order to not be discarded (must be a value between zero and one).
        stem_search_max_inclination: Maximum inclination angle to the z-axis that the first
            principal component of a cluster can have in order to not be discarded (in degrees).
        stem_search_refined_circle_fitting: Whether the step for the refined circle / ellipse fitting should be
            executed.
        stem_search_ellipse_fitting: Whether the ellipse fitting should be executed.
        stem_search_circle_fitting_method: Circle fitting method to use: :code:`"m-estimator"` | :code:`"ransac"`.
        stem_search_circle_fitting_num_layers: Number of horizontal layers used for the circle / ellipse fitting.
            Depending on the settings for :code:`stem_search_circle_fitting_layer_height` and
            :code:`stem_search_circle_fitting_layer_overlap`, this parameter controls which height range of the stem
            clusters is considered for circle / ellipse fitting.
        stem_search_circle_fitting_layer_start: Height above the ground at which the lowest layer used for circle /
            ellipse fitting starts (in meters).
        stem_search_circle_fitting_layer_height: Height of the horizontal layers used for circle / ellipse fitting
            (in meters).
        stem_search_circle_fitting_layer_overlap: Overlap between adjacent horizontal layers used for circle / ellipse
            fitting (in meters).
        stem_search_circle_fitting_bandwidth: Bandwidth for circle fitting. It is used in the calculation of the
            goodness of fit and the circumferential completeness index and determines how far points may be from the
            outline of a circle to be counted as belonging to the outline. When calculating the goodness of fit, a
            Gaussian kernel is used to measure the contribution of a point to the outline of the circle, and the
            bandwidth of the kernel is set to the specified value (in meters).
        stem_search_circle_fitting_min_points: Minimum number of points that a horizontal layer must contain in order to
            perform circle / ellipse fitting on it.
        stem_search_circle_fitting_min_stem_diameter: Minimum circle / ellipse diameter to be considered a valid fit.
        stem_search_circle_fitting_max_stem_diameter: Maximum circle / ellipse diameter to be considered a valid fit.
        stem_search_circle_fitting_min_completeness_idx: Minimum circumferential completeness index that circles must
            achieve in the circle fitting procedure. If set to :code:`None`, circles are not filtered based on their
            circumferential completeness index.
        stem_search_circle_fitting_small_buffer_width: This parameter is only used when
            :code:`stem_search_refined_circle_fitting` is set to :code:`True`. It defines the width of the buffer area
            if the diameter of the initally fited circles or ellipses is less than or equal to
            :code:`stem_search_circle_fitting_switch_buffer_threshold` (in meters).
        stem_search_circle_fitting_large_buffer_width: This parameter is only used when
            :code:`stem_search_refined_circle_fitting` is set to :code:`True`. It defines the width of the buffer area
            if the diameter of the initally fited circles or ellipses is larger than
            :code:`stem_search_circle_fitting_switch_buffer_threshold` (in meters).
        stem_search_circle_fitting_switch_buffer_threshold: Threshold for the diameter of the preliminary circles or
            ellipses that controls when to switch between :code:`stem_search_circle_fitting_small_buffer_width` and
            :code:`stem_search_circle_fitting_large_buffer_width` (in meters). This parameter is only used when
            :code:`stem_search_refined_circle_fitting` is set to :code:`True`.
        stem_search_ellipse_filter_threshold: In the ellipse fitting, ellipses are only kept if the ratio of the radius
            along the semi-minor axis to the radius along the semi-major axis is greater than or equal to this
            threshold. This parameter is only used when :code:`stem_search_ellipse_fitting` is set to :code:`True`.
        stem_search_circle_fitting_max_std_diameter: Threshold for filtering the stem clusters based
            on the standard deviation of the diameters of the fitted circles / ellipses. If there is at
            least one combination of :code:`stem_search_circle_fitting_std_num_layers` layers for which the standard
            deviation of the diameters of the fitted circles / ellipses is below or equal to this threshold, the cluster
            is kept, otherwise it is discarded. If :code:`stem_search_circle_fitting_max_std_position` is not
            :code:`None`, it is additionally required that the standard deviation of the circle / ellipse center
            positions is smaller than or equal to the given threshold for the combination of layers.
        stem_search_circle_fitting_max_std_position: Threshold for filtering the stem clusters based
            on the standard deviation of the center positions of the fitted circles / ellipses. Requires that a
            combination of :code:`stem_search_circle_fitting_std_num_layers` layers must exist for which the standard
            deviation of the center positions of the fitted circles / ellipses is below or equal to this threshold in
            order to keep a cluster. If set to :code:`None`, this filtering criterion is deactivated.
        stem_search_circle_fitting_std_num_layers: Number of horizontal layers to consider in each
            sample for calculating the standard deviation of the diameters / center positions of the fitted circles /
            ellipses.

    .. rubric:: 4. Computation of Stem Positions and Diameters

    In this step, the stem clusters obtained in the previous step are used to compute the stem positions and stem
    diameters at breast height. For this purpose, the circles and ellipses fitted in the previous step are used. For
    each stem, the combination of those circles or ellipses (if no valid combination of circles was found) is selected
    whose diameters have the lowest standard deviation. To estimate the stem position at breast height, a linear model
    is fitted that predicts the center position of the selected circles or ellipses from the layer height. The
    prediction of the fitted model for a height of 1.3 m is used as an estimate of the stem position at breast height.
    
    To estimate the stem diameter at breast height, the stem radius for each of the selected layers is re-estimated by
    fitting a generalized additive model (GAM) to the points from the respective layer. If
    :code:`stem_search_refined_circle_fitting` is :code:`True`, only the points within the
    clipping area created for the refined circle / ellipse fitting are used for the GAM fitting. Otherwise, a clipping
    area with a width of :code:`stem_search_gam_buffer_width` is created around the circle / ellipse outlines to select
    the input points for the GAM fitting. Before fitting the GAM, the input points are centered around the circle /
    ellipse center and converted into polar coordinates. The GAM is then used to predict the radius of the polar
    coordinates from the angle. The fitted GAM is then used to predict the stem radii in one-degree intervals.
    From these predictions, the stem's boundary polygon is constructed and the stem diameter is estimated from the area
    of the boundary polygon. Finally, a linear model is fitted that predicts the stem diameter from the layer height.
    The prediction of the fitted model for a height of 1.3 m is used as an estimate of the stem diameter at breast
    height.

    Parameters:
        stem_search_gam_buffer_width: If the refined circle fitting is deactivated (i.e.,
            :code:`stem_search_refined_circle_fitting` is set to :code:`False`), all points in a buffer area around
            the outline of the initial circle / ellipse are cut out for the GAM fitting. This parameter defines the
            width of the buffer area.
        stem_search_gam_max_radius_diff: If the difference between the minimum and the maximum of predicted radii is
            greater than this parameter, the fitted GAM is considered invalid, and the diameter of the fitted circle /
            ellipse is used instead.

    .. rubric:: 5. Tree Segmentation Using Region Growing

    This stage aims to determine the complete sets of points that represent each tree. In particular, this involves
    segmenting the canopy points into individual tree crowns. To accomplish this, a region growing method is used.
    Before the region growing, the points are downsampled using voxel-based subsampling. In the first step, the region
    growing procedure selects an initial set of seed points for each tree. These should be points
    that are very likely to belong to the corresponding tree. In an iterative process, the sets of points assigned to
    each tree are then expanded. In each iteration, the neighboring points of each seed point within a certain search
    radius are determined. Neighboring points that are not yet assigned to any tree are added to the same tree as the
    seed point and become seed points in the next iteration. The region growing continues until there are no more seed
    points to be processed or the maximum number of iterations is reached.

    To select the initial seed points for a given tree, the following approach is used: (1) All points that were
    assigned to the respective stem during the stem detection stage are used as seed points. (2) Additionally, a
    cylinder with a height of :code:`tree_seg_seed_layer_height` and a diameter of
    :code:`tree_seg_seed_diameter_factor * d` is considered, where :code:`d` is the tree's
    stem diameter at breast height, which has been computed in the previous step. The cylinder's center is
    positioned at the stem center at breast height, which also has been computed in the previous stage. All points
    within the cylinder that have not yet been selected as seed points for other trees are selected as seed points.

    The search radius for the iterative region growing procedure is set as follows: First, the search radius is set
    to the voxel size used for voxel-based subsampling, which is done before starting the region growing procedure.
    The search radius is increased by the voxel size if one of the following conditions is fulfilled at the end of a
    region growing iteration:

    - The ratio between the number of points newly assigned to trees in the iteration and the number of remaining,
      unassigned points is below :code:`tree_seg_min_total_assignment_ratio`.
    - The ratio between the number of trees to which new points have been assigned in the iteration and the total
      number of trees is below :code:`tree_seg_min_tree_assignment_ratio`.

    The search radius is increased up to a maximum radius of :code:`tree_seg_max_search_radius`. If the
    search radius has not been increased for :code:`tree_seg_decrease_search_radius_after_num_iter`, it
    is reduced by the voxel size.

    To promote upward growth, the z-coordinates of the points are divided by :code:`tree_seg_z_scale` before
    the region growing.

    Since the terrain filtering in the first step of the algorithm may be inaccurate and some tree points may be
    falsely classified as terrain points, both terrain and non-terrain points are considered by the region growing
    procedure. However, to prevent large portions of terrain points from being included in tree instances, terrain
    points are only assigned to a tree if their cumulative search distance from the initial seed point is below the
    threshold defined by :code:`tree_seg_cum_search_dist_include_ground`. The cumulative search distance is defined as
    the total distance traveled between consecutive points until reaching a terrain point.

    Parameters:
        tree_seg_voxel_size: Voxel size with which the points are downsampled before the region growing (in meters).
        tree_seg_z_scale: Factor by which to divide the z-coordinates of the points before the region growing. To
            promote upward growth, this factor should be larger than 1.
        tree_seg_seed_layer_height: Height of the cylinders that are placed around the stem centers at breast height for
            seed point selection (in meters).
        tree_seg_seed_diameter_factor: Factor to multiply with the stem diameter at breast height to obtain the diameter
            of the cylinder used for seed point selection.
        tree_seg_seed_min_diameter: Minimum diameter of the cylinder used for seed point selection.
        tree_seg_min_total_assignment_ratio: Threshold controlling when to increase the search radius. If the ratio
            between the number of points newly assigned to trees in an iteration and the number of remaining, unassigned
            points is below this threshold, the search radius is increased by :code:`tree_seg_voxel_size` up to a
            maximum search radius of :code:`tree_seg_max_search_radius`.
        tree_seg_min_tree_assignment_ratio: Threshold controlling when to increase the search radius. If the ratio
            between the number of trees to which new points have been assigned in the iteration and the total number of
            trees is below this threshold, the search radius is increased by :code:`tree_seg_voxel_size` up to a maximum
            search radius of :code:`tree_seg_max_search_radius`.
        tree_seg_max_search_radius: Maximum search radius (in meters).
        tree_seg_decrease_search_radius_after_num_iter: Number of region growing iterations after which to decrease the
            search radius by :code:`tree_seg_voxel_size` if it has not been increased in these iterations.
        tree_seg_max_iterations: Maximum number of region growing iterations.
        tree_seg_cum_search_dist_include_terrain: Maximum cumulative search distance between the initial seed point and
            a terrain point to include that terrain point in a tree instance (in meters).

    Raises:
        ValueError: If :code:`stem_search_min_z` is set to a value smaller than
            :code:`csf_tree_classification_threshold`.
        ValueError: If :code:`stem_search_circle_fitting_layer_start` is set to a value smaller than
            :code:`stem_search_min_z`.
        ValueError: If :code:`invalid_tree_id` is set to a value greater than one.
        ValueError: If :code:`stem_search_circle_fitting_min_stem_diameter` is greater than or equal to
            :code:`stem_search_circle_fitting_max_stem_diameter`.
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals, too-many-statements
        self,
        *,
        invalid_tree_id: int = -1,
        num_workers: Optional[int] = -1,
        visualization_folder: Optional[Union[str, Path]] = None,
        random_seed: int = 0,
        # CSF parameters
        csf_terrain_classification_threshold: float = 0.5,
        csf_tree_classification_threshold: float = 0.5,
        csf_correct_steep_slope: bool = False,
        csf_iterations: int = 500,
        csf_resolution: float = 0.5,
        csf_rigidness: int = 2,
        # DTM construction parameters
        dtm_k: int = 400,
        dtm_power: float = 1,
        dtm_resolution: float = 0.25,
        dtm_voxel_size: float = 0.05,
        # parameters for the identification of stem clusters
        stem_search_min_z: float = 1.0,
        stem_search_max_z: float = 4.0,
        stem_search_voxel_size: float = 0.015,
        stem_search_dbscan_2d_eps: float = 0.025,
        stem_search_dbscan_2d_min_points: int = 100,
        stem_search_dbscan_3d_eps: float = 0.1,
        stem_search_dbscan_3d_min_points: int = 15,
        stem_search_min_cluster_points: Optional[int] = 300,
        stem_search_min_cluster_height: Optional[float] = 1.5,
        stem_search_min_cluster_intensity: Optional[float] = 6000,
        stem_search_pc1_min_explained_variance: Optional[float] = 0.5,
        stem_search_max_inclination: Optional[float] = 45,
        stem_search_refined_circle_fitting: bool = False,
        stem_search_ellipse_fitting: bool = False,
        stem_search_circle_fitting_method: Literal["m-estimator", "ransac"] = "ransac",
        stem_search_circle_fitting_layer_start: float = 1.0,
        stem_search_circle_fitting_num_layers: int = 15,
        stem_search_circle_fitting_layer_height: float = 0.225,
        stem_search_circle_fitting_layer_overlap: float = 0.025,
        stem_search_circle_fitting_bandwidth: float = 0.01,
        stem_search_circle_fitting_min_points: int = 15,
        stem_search_circle_fitting_min_stem_diameter: float = 0.02,
        stem_search_circle_fitting_max_stem_diameter: float = 1.0,
        stem_search_circle_fitting_min_completeness_idx: Optional[float] = 0.3,
        stem_search_circle_fitting_small_buffer_width: float = 0.06,
        stem_search_circle_fitting_large_buffer_width: float = 0.09,
        stem_search_circle_fitting_switch_buffer_threshold: float = 0.3,
        stem_search_ellipse_filter_threshold: float = 0.6,
        stem_search_circle_fitting_max_std_diameter: float = 0.02,
        stem_search_circle_fitting_max_std_position: Optional[float] = None,
        stem_search_circle_fitting_std_num_layers: int = 6,
        stem_search_gam_buffer_width: float = 0.03,
        stem_search_gam_max_radius_diff: Optional[float] = 0.3,
        # region growing parameters
        tree_seg_voxel_size: float = 0.05,
        tree_seg_z_scale: float = 2,
        tree_seg_seed_layer_height: float = 0.6,
        tree_seg_seed_diameter_factor: float = 1.05,
        tree_seg_seed_min_diameter: float = 0.05,
        tree_seg_min_total_assignment_ratio: float = 0.002,
        tree_seg_min_tree_assignment_ratio: float = 0.3,
        tree_seg_max_search_radius: float = 0.5,
        tree_seg_decrease_search_radius_after_num_iter: int = 10,
        tree_seg_max_iterations: int = 1000,
        tree_seg_cum_search_dist_include_terrain: float = 0.9,
    ):
        super().__init__()

        if invalid_tree_id > 0:
            raise ValueError("invalid_tree_id must either be zero or a negative number.")

        if stem_search_min_z < csf_tree_classification_threshold:
            raise ValueError("csf_tree_classification_threshold must be smaller than stem_search_min_z.")

        if stem_search_circle_fitting_layer_start < stem_search_min_z:
            raise ValueError("stem_search_min_z must be smaller than stem_search_circle_fitting_layer_start.")

        if stem_search_circle_fitting_min_stem_diameter >= stem_search_circle_fitting_max_stem_diameter:
            raise ValueError("Minimum stem diameter must be smaller than maximum stem diameter.")

        self._invalid_tree_id = invalid_tree_id
        self._num_workers = num_workers if num_workers is not None else 1

        if visualization_folder is None or isinstance(visualization_folder, Path):
            self._visualization_folder = visualization_folder
        else:
            self._visualization_folder = Path(visualization_folder)
            self._visualization_folder.mkdir(exist_ok=True, parents=True)
        self._random_seed = random_seed
        self._random_generator = np.random.default_rng(seed=random_seed)

        self._csf_terrain_classification_threshold = csf_terrain_classification_threshold
        self._csf_tree_classification_threshold = csf_tree_classification_threshold
        self._csf_correct_steep_slope = csf_correct_steep_slope
        self._csf_iterations = csf_iterations
        self._csf_resolution = csf_resolution
        self._csf_rigidness = csf_rigidness

        self._dtm_k = dtm_k
        self._dtm_power = dtm_power
        self._dtm_resolution = dtm_resolution
        self._dtm_voxel_size = dtm_voxel_size

        self._stem_search_min_z = stem_search_min_z
        self._stem_search_max_z = stem_search_max_z
        self._stem_search_voxel_size = stem_search_voxel_size
        self._stem_search_dbscan_2d_eps = stem_search_dbscan_2d_eps
        self._stem_search_dbscan_2d_min_points = stem_search_dbscan_2d_min_points
        self._stem_search_dbscan_3d_eps = stem_search_dbscan_3d_eps
        self._stem_search_dbscan_3d_min_points = stem_search_dbscan_3d_min_points

        self._stem_search_min_cluster_points = stem_search_min_cluster_points
        self._stem_search_min_cluster_height = stem_search_min_cluster_height
        self._stem_search_min_cluster_intensity = stem_search_min_cluster_intensity
        self._stem_search_pc1_min_explained_variance = stem_search_pc1_min_explained_variance
        self._stem_search_max_inclination = stem_search_max_inclination
        self._stem_search_refined_circle_fitting = stem_search_refined_circle_fitting
        self._stem_search_ellipse_fitting = stem_search_ellipse_fitting
        self._stem_search_circle_fitting_method = stem_search_circle_fitting_method
        self._stem_search_circle_fitting_num_layers = stem_search_circle_fitting_num_layers
        self._stem_search_circle_fitting_layer_start = stem_search_circle_fitting_layer_start
        self._stem_search_circle_fitting_layer_height = stem_search_circle_fitting_layer_height
        self._stem_search_circle_fitting_layer_overlap = stem_search_circle_fitting_layer_overlap
        self._stem_search_circle_fitting_bandwidth = stem_search_circle_fitting_bandwidth
        self._stem_search_circle_fitting_min_points = stem_search_circle_fitting_min_points
        self._stem_search_circle_fitting_min_stem_diameter = stem_search_circle_fitting_min_stem_diameter
        self._stem_search_circle_fitting_max_stem_diameter = stem_search_circle_fitting_max_stem_diameter
        self._stem_search_circle_fitting_min_completeness_idx = stem_search_circle_fitting_min_completeness_idx
        self._stem_search_circle_fitting_small_buffer_width = stem_search_circle_fitting_small_buffer_width
        self._stem_search_circle_fitting_large_buffer_width = stem_search_circle_fitting_large_buffer_width
        self._stem_search_circle_fitting_switch_buffer_threshold = stem_search_circle_fitting_switch_buffer_threshold
        self._stem_search_ellipse_filter_threshold = stem_search_ellipse_filter_threshold
        self._stem_search_circle_fitting_max_std_diameter = stem_search_circle_fitting_max_std_diameter
        if stem_search_circle_fitting_max_std_position:
            self._stem_search_circle_fitting_max_std_position = stem_search_circle_fitting_max_std_position
        else:
            self._stem_search_circle_fitting_max_std_position = np.inf
        self._stem_search_circle_fitting_std_num_layers = stem_search_circle_fitting_std_num_layers

        self._stem_search_gam_buffer_width = stem_search_gam_buffer_width
        self._stem_search_gam_max_radius_diff = stem_search_gam_max_radius_diff

        self._tree_seg_voxel_size = tree_seg_voxel_size
        self._tree_seg_z_scale = tree_seg_z_scale
        self._tree_seg_seed_layer_height = tree_seg_seed_layer_height
        self._tree_seg_seed_diameter_factor = tree_seg_seed_diameter_factor
        self._tree_seg_seed_min_diameter = tree_seg_seed_min_diameter
        self._tree_seg_min_total_assignment_ratio = tree_seg_min_total_assignment_ratio
        self._tree_seg_min_tree_assignment_ratio = tree_seg_min_tree_assignment_ratio
        self._tree_seg_max_search_radius = tree_seg_max_search_radius
        self._tree_seg_decrease_search_radius_after_num_iter = tree_seg_decrease_search_radius_after_num_iter
        self._tree_seg_max_iterations = tree_seg_max_iterations
        self._tree_seg_cum_search_dist_include_terrain = tree_seg_cum_search_dist_include_terrain

    def detect_stems(  # pylint: disable=too-many-locals, too-many-statements, too-many-branches
        self,
        stem_layer_xyz: FloatArray,
        dtm: FloatArray,
        dtm_offset: FloatArray,
        intensities: Union[FloatArray, None] = None,
        point_cloud_id: Optional[str] = None,
        crs: Optional[str] = None,
    ) -> Tuple[FloatArray, FloatArray, LongArray]:
        r"""
        Detects tree stems in a 3D point cloud.

        Args:
            stem_layer_xyz: Point coordinates of the points within the stem layer as defined by the constructor
                parameters :code:`stem_search_min_z` and :code:`stem_search_max_z`.
            dtm: Digital terrain model.
            dtm_offset: X- and y-coordinate of the top left corner of the DTM grid.
            intensities: Reflection intensities of all points in the point cloud. If set to :code:`None`, filtering
                steps that use intensity values are skipped.
            point_cloud_id: ID of the point cloud to be used in the file names of the visualization results. If set to
                :code:`None`, no visualizations are created.
            crs: EPSG code of the coordinate reference system of the input point cloud. The EPSG code is used to set the
                coordinate reference system when exporting intermediate data. If set to :code:`None`, no coordinate
                reference system is set for the exported data.

        Returns:
            :Tuple of three arrays:
                - X- and y-coordinates of the position of each detected stem at breast height (1.3 m).
                - Diameter of each detected stem at breast height.
                - Stem ID label of each point. Points that do not belong to any stem are assigned the label
                  :code:`-1`.

        Shape:
            - :code:`stem_layer_xyz`: :math:`(N, 3)`
            - :code:`dtm`: :math:`(H, W)`
            - :code:`dtm_offset`: :math:`(2)`
            - :code:`intensities`: :math:`(N)`
            - Output: :math:`(S, 2)`, :math:`(S)`, :math:`(N)`.

            | where
            |
            | :math:`N = \text{ number of points}`
            | :math:`S = \text{ number of detected stems}`
            | :math:`H = \text{ extent of the DTM grid in y-direction}`
            | :math:`W = \text{ extent of the DTM grid in x-direction}`
        """

        stem_layer_xyz_downsampled, selected_indices, inverse_indices = voxel_downsampling(
            stem_layer_xyz, voxel_size=self._stem_search_voxel_size
        )
        if intensities is not None:
            intensities = intensities[selected_indices]
            del selected_indices

        with Profiler("2D DBSCAN", self._performance_tracker):
            self._logger.info("Cluster stem points in 2D...")
            dbscan = DBSCAN(
                eps=self._stem_search_dbscan_2d_eps,
                min_samples=self._stem_search_dbscan_2d_min_points,
                n_jobs=self._num_workers,
            )
            dbscan.fit(stem_layer_xyz_downsampled[:, :2])
            cluster_labels = dbscan.labels_.astype(np.int64)
            unique_cluster_labels = np.unique(cluster_labels)
            unique_cluster_labels = unique_cluster_labels[unique_cluster_labels != -1]

            self._logger.info("Found %d stem candidates.", len(unique_cluster_labels))

            if self._visualization_folder is not None and point_cloud_id is not None:
                point_cloud_attributes = {"dbscan_2d": (cluster_labels + 1).astype(np.uint32)}

        # In the paper, it is not explicitely mentioned that a filtering step based on point count and vertical extent
        # is done between the 2D and the 3D DBSCAN clustering. However, including such filtering does not change the
        # final result because clusters can only get smaller during 3D clustering and saves computational resources.
        # Therefore, it is included to optimize the computational performance.
        with Profiler("Filtering of stem clusters based on point count", self._performance_tracker):
            cluster_labels, unique_cluster_labels = filter_instances_min_points(
                cluster_labels, unique_cluster_labels, min_points=self._stem_search_min_cluster_points, inplace=True
            )

            self._logger.info(
                "%d stem candidates remaining after discarding clusters with too few points.",
                len(unique_cluster_labels),
            )

            if self._visualization_folder is not None and point_cloud_id is not None:
                point_cloud_attributes["filtering_point_count_2d"] = (cluster_labels + 1).astype(np.uint32)

        with Profiler("Filtering of stem clusters based on vertical extent", self._performance_tracker):
            cluster_labels, unique_cluster_labels = filter_instances_vertical_extent(
                stem_layer_xyz_downsampled,
                cluster_labels,
                unique_cluster_labels,
                min_vertical_extent=self._stem_search_min_cluster_height,
                inplace=True,
            )

            self._logger.info(
                "%d stem candidates remaining after discarding clusters with too small " + "vertical extent.",
                len(unique_cluster_labels),
            )

            if self._visualization_folder is not None and point_cloud_id is not None:
                point_cloud_attributes["filtering_vertical_extent_2d"] = (cluster_labels + 1).astype(np.uint32)

        with Profiler("3D DBSCAN", self._performance_tracker):
            self._logger.info("Cluster stem points in 3D...")

            dbscan = DBSCAN(
                eps=self._stem_search_dbscan_3d_eps,
                min_samples=self._stem_search_dbscan_3d_min_points,
                n_jobs=self._num_workers,
            )

            next_label = unique_cluster_labels.max() + 1 if len(unique_cluster_labels) > 0 else 0
            for label in unique_cluster_labels:
                cluster_indices = np.flatnonzero(cluster_labels == label)

                dbscan.fit(stem_layer_xyz_downsampled[cluster_indices])
                new_cluster_labels = dbscan.labels_.astype(np.int64)
                new_cluster_labels[new_cluster_labels != -1] += next_label
                next_label = new_cluster_labels.max() + 1
                cluster_labels[cluster_indices] = new_cluster_labels

                del cluster_indices
                del new_cluster_labels

            cluster_labels, unique_cluster_labels = make_labels_consecutive(
                cluster_labels, ignore_id=-1, inplace=True, return_unique_labels=True
            )

            self._logger.info(
                "%d stem candidates after clustering in 3D.",
                len(unique_cluster_labels),
            )

        if self._visualization_folder is not None and point_cloud_id is not None:
            point_cloud_attributes["dbscan_3d"] = (cluster_labels + 1).astype(np.uint32)

        with Profiler("Filtering of stem clusters based on point count", self._performance_tracker):
            cluster_labels, unique_cluster_labels = filter_instances_min_points(
                cluster_labels, unique_cluster_labels, min_points=self._stem_search_min_cluster_points, inplace=True
            )

            self._logger.info(
                "%d stem candidates remaining after discarding clusters with too few points.",
                len(unique_cluster_labels),
            )

            if self._visualization_folder is not None and point_cloud_id is not None:
                point_cloud_attributes["filtered_point_count_3d"] = (cluster_labels + 1).astype(np.uint32)

        with Profiler("Filtering of stem clusters based on vertical extent", self._performance_tracker):
            cluster_labels, unique_cluster_labels = filter_instances_vertical_extent(
                stem_layer_xyz_downsampled,
                cluster_labels,
                unique_cluster_labels,
                min_vertical_extent=self._stem_search_min_cluster_height,
                inplace=True,
            )

            if self._visualization_folder is not None and point_cloud_id is not None:
                point_cloud_attributes["filtered_vertical_extent_3d"] = (cluster_labels + 1).astype(np.uint32)

            self._logger.info(
                "%d stem candidates remaining after discarding clusters with too small vertical extent.",
                len(unique_cluster_labels),
            )

        if intensities is not None:
            with Profiler("Filtering of stem clusters based on intensity values", self._performance_tracker):
                cluster_labels, unique_cluster_labels = filter_instances_intensity(
                    intensities,
                    cluster_labels,
                    unique_cluster_labels,
                    min_intensity=self._stem_search_min_cluster_intensity,
                    threshold_percentile=0.8,
                    inplace=True,
                )

                if self._visualization_folder is not None and point_cloud_id is not None:
                    point_cloud_attributes["filtered_intensity_3d"] = (cluster_labels + 1).astype(np.uint32)

                self._logger.info(
                    "%d stem candidates remaining after discarding clusters with small intensity.",
                    len(unique_cluster_labels),
                )

        with Profiler("Filtering of stem clusters based on PCA", self._performance_tracker):
            cluster_labels, unique_cluster_labels = filter_instances_pca(
                stem_layer_xyz_downsampled,
                cluster_labels,
                unique_cluster_labels,
                min_explained_variance=self._stem_search_pc1_min_explained_variance,
                max_inclination=self._stem_search_max_inclination,
                inplace=True,
            )

            if self._visualization_folder is not None and point_cloud_id is not None:
                point_cloud_attributes["filtered_pca_3d"] = (cluster_labels + 1).astype(np.uint32)

            self._logger.info(
                "%d stem candidates remaining after filtering based on pricipal component analysis.",
                len(unique_cluster_labels),
            )

        (
            layer_circles,
            layer_ellipses,
            terrain_heights_at_stem_positions,
            layer_heights,
            stem_layers_xy,
            batch_lengths_xy,
        ) = self.fit_circles_or_ellipses_to_stems(
            stem_layer_xyz_downsampled,
            cluster_labels,
            unique_cluster_labels,
            dtm,
            dtm_offset,
            point_cloud_id=point_cloud_id,
        )

        if self._stem_search_refined_circle_fitting:
            (
                layer_circles,
                layer_ellipses,
                stem_layers_xy,
                batch_lengths_xy,
            ) = self.fit_refined_circles_and_ellipses_to_stems(
                stem_layer_xyz,
                layer_circles,
                layer_ellipses,
                terrain_heights_at_stem_positions,
                point_cloud_id=point_cloud_id,
            )

        with Profiler(
            "Filtering of stem clusters based on standard deviation of circle / ellipse diameters",
            self._performance_tracker,
        ):
            filter_mask, best_circle_combination, best_ellipse_combination = self.filter_instances_stem_layers_std(
                layer_circles, layer_ellipses
            )
            layer_circles = layer_circles[filter_mask]
            layer_circles[best_circle_combination[:, 0] == -1] = -1
            layer_ellipses = layer_ellipses[filter_mask]

            cluster_labels[~np.isin(cluster_labels, unique_cluster_labels[filter_mask], assume_unique=True)] = -1
            cluster_labels, unique_cluster_labels = make_labels_consecutive(
                cluster_labels, ignore_id=-1, inplace=True, return_unique_labels=True
            )

            self.rename_visualizations_after_filtering(filter_mask, point_cloud_id=point_cloud_id)

            filter_mask = np.repeat(filter_mask, self._stem_search_circle_fitting_num_layers)
            stem_layers_xy = stem_layers_xy[np.repeat(filter_mask, batch_lengths_xy)]
            batch_lengths_xy = batch_lengths_xy[filter_mask]

            self._logger.info(
                "%d stems remaining after discarding clusters with too high standard deviation.", len(layer_circles)
            )

        if self._visualization_folder is not None and point_cloud_id is not None:
            point_cloud_attributes["filtered_circle_fitting_3d"] = (cluster_labels + 1).astype(np.uint32)
            self.export_point_cloud(
                stem_layer_xyz_downsampled,
                point_cloud_attributes,
                "stem_clusters",
                point_cloud_id,
                crs=crs,
            )
            del point_cloud_attributes
        del unique_cluster_labels

        if not self._stem_search_refined_circle_fitting:
            if not stem_layer_xyz_downsampled.flags.f_contiguous:
                stem_layer_xyz_downsampled = stem_layer_xyz_downsampled.copy(order="F")

            stem_layers_xy, batch_lengths_xy = collect_inputs_stem_layers_refined_fitting_cpp(
                stem_layer_xyz_downsampled,
                layer_circles.reshape((-1, 3)).astype(stem_layer_xyz_downsampled.dtype, order="F"),
                layer_ellipses.reshape((-1, 5)).astype(stem_layer_xyz_downsampled.dtype, order="F"),
                terrain_heights_at_stem_positions,
                float(self._stem_search_circle_fitting_layer_start),
                int(self._stem_search_circle_fitting_num_layers),
                float(self._stem_search_circle_fitting_layer_height),
                float(self._stem_search_circle_fitting_layer_overlap),
                float(self._stem_search_circle_fitting_switch_buffer_threshold),
                float(self._stem_search_gam_buffer_width),
                float(self._stem_search_gam_buffer_width),
                0,
                int(self._num_workers),
            )
        del stem_layer_xyz_downsampled

        with Profiler("Computation of stem positions", self._performance_tracker):
            self._logger.info("Compute stem positions...")
            stem_positions = self.compute_stem_positions(
                layer_circles, layer_ellipses, layer_heights, best_circle_combination, best_ellipse_combination
            )

        with Profiler("Computation of stem diameters", self._performance_tracker):
            self._logger.info("Compute stem diameters...")
            stem_diameters = self.compute_stem_diameters(
                layer_circles,
                layer_ellipses,
                layer_heights,
                stem_layers_xy,
                batch_lengths_xy,
                best_circle_combination,
                best_ellipse_combination,
                point_cloud_id=point_cloud_id,
            )

        return stem_positions, stem_diameters, cluster_labels[inverse_indices]

    def fit_circles_or_ellipses_to_stems(  # pylint: disable=too-many-locals, too-many-statements, too-many-branches
        self,
        stem_layer_xyz: FloatArray,
        cluster_labels: LongArray,
        unique_cluster_labels: LongArray,
        dtm: FloatArray,
        dtm_offset: FloatArray,
        point_cloud_id: Optional[str] = None,
    ) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, LongArray]:
        r"""
        Given a set of point clusters that may represent individual tree stems, circles and ellipses are fitted to
        multiple horinzontal layers of each cluster. If a horizontal layer contains less than
        :code:`stem_search_circle_fitting_min_points` points, neither a circle nor an ellipse is
        fitted. To obtain the horizontal layers, :code:`stem_search_circle_fitting_num_layers` horizontal layers with a
        height of :code:`stem_search_circle_fitting_layer_height` are created starting at a height of
        :code:`stem_search_circle_fitting_layer_start`. The layers have an overlap of
        :code:`stem_search_circle_fitting_layer_overlap` to the previous layer (the variables are constructor
        parameters).

        The ellipse fitting is only done if :code:`stem_search_ellipse_fitting` (constructor parameter) is set to
        :code:`True`.

        Args:
            stem_layer_xyz: Point coordinates of the points within the stem layer as defined by the constructor
                parameters :code:`stem_search_min_z` and :code:`stem_search_max_z`.
            cluster_labels: Indices indicating to which cluster each point belongs. Points not belonging to any cluster
                should be assigned the ID :code:`-1`.
            unique_cluster_labels: Unique cluster labels, i.e., an array that should contain each cluster ID once. The
                cluster IDs are expected to start with zero and to be in a continuous range.
            dtm: Digital terrain model.
            dtm_offset: X- and y-coordinate of the top left corner of the DTM grid.
            point_cloud_id: ID of the point cloud to be used in the file names of the visualization results. If set to
                :code:`None`, no visualizations are created.

        Returns:
            :Tuple of six arrays:
                - Parameters of the circles that were fitted to the layers of each cluster. Each circle is represented
                  by three values, namely the x- and y-coordinates of its center and its radius. If the circle fitting
                  is not successfull for a layer, all parameters are set to :code:`-1`.
                - Parameters of the ellipses that were fitted to the layers of each cluster. Each ellipse is represented
                  by five values, namely the x- and y-coordinates of its center, its radius along the semi-major and
                  along the semi-minor axis, and the counterclockwise angle of rotation from the x-axis to the
                  semi-major axis of the ellipse. If the ellipse fitting is not successfull for a layer or results in an
                  ellipse whose axis ratio is smaller than :code:`stem_search_ellipse_filter_threshold` (constructor
                  parameter), all parameters are set to :code:`-1`.
                - Terrain height at the centroid position of each stem cluster.
                - Height above the terrain of the midpoint of each horizontal layer.
                - X- and y-coordinates of the points in each horizontal layer of each cluster. Points belonging to the
                  same layer of the same cluster are stored consecutively.
                - Number of points belonging to each horizontal layer of each cluster.

        Shape:
            - :code:`stem_layer_xyz`: :math:`(N, 3)`
            - :code:`cluster_labels`: :math:`(N)`
            - :code:`unique_cluster_labels`: :math:`(S)`
            - Output: :math:`(S, L, 3)`, :math:`(S, L, 5)`, :math:`(S)`, :math:`(L)`,
              :math:`(N_{0,0} + ... + N_{S, L}, 2)`, :math:`(S \cdot L)`

            | where
            |
            | :math:`N = \text{ number of points in the stem layer}`
            | :math:`S = \text{ number of stem clusters}`
            | :math:`L = \text{ number of horinzontal layers to which circles / ellipses are fitted}`
            | :math:`N_{t,s} = \text{ number of points selected in the l-th layer of cluster } s`
        """

        with Profiler("Fitting of circles / ellipses to stem candidates", self._performance_tracker):
            self._logger.info("Fitting circles / ellipses to stem candidates...")

            num_layers = self._stem_search_circle_fitting_num_layers

            layer_circles = np.full(
                (len(unique_cluster_labels), num_layers, 3),
                fill_value=-1,
                dtype=stem_layer_xyz.dtype,
            )

            layer_ellipses = np.full(
                (len(unique_cluster_labels), num_layers, 5),
                fill_value=-1,
                dtype=stem_layer_xyz.dtype,
            )

            if not stem_layer_xyz.flags.f_contiguous:
                stem_layer_xyz = stem_layer_xyz.copy(order="F")

            if not dtm.flags.f_contiguous:
                dtm = dtm.copy(order="F")

            stem_layer_xy, batch_lengths_xy, terrain_heights_at_cluster_positions, layer_heights = (
                collect_inputs_stem_layers_fitting_cpp(
                    stem_layer_xyz,
                    cluster_labels,
                    unique_cluster_labels,
                    dtm,
                    dtm_offset,
                    float(self._dtm_resolution),
                    float(self._stem_search_circle_fitting_layer_start),
                    int(num_layers),
                    float(self._stem_search_circle_fitting_layer_height),
                    float(self._stem_search_circle_fitting_layer_overlap),
                    int(self._stem_search_circle_fitting_min_points),
                    int(self._num_workers),
                )
            )

            if len(unique_cluster_labels) == 0:
                return (
                    layer_circles,
                    layer_ellipses,
                    terrain_heights_at_cluster_positions,
                    layer_heights,
                    stem_layer_xy,
                    batch_lengths_xy,
                )

            min_radius = self._stem_search_circle_fitting_min_stem_diameter / 2
            max_radius = self._stem_search_circle_fitting_max_stem_diameter / 2
            min_start_radius = min_radius + min(
                2 * self._stem_search_circle_fitting_bandwidth, (max_radius - min_radius) / 4
            )
            max_start_radius = max_radius - min(
                2 * self._stem_search_circle_fitting_bandwidth, (max_radius - min_radius) / 4
            )

            circle_detector: Union[MEstimator, Ransac]
            if self._stem_search_circle_fitting_method == "m-estimator":
                circle_detector = MEstimator(
                    bandwidth=self._stem_search_circle_fitting_bandwidth,
                    break_min_change=1e-6,
                    min_step_size=1e-10,
                    max_iterations=300,
                    armijo_min_decrease_percentage=0.5,
                    armijo_attenuation_factor=0.25,
                )
                circle_detector.detect(
                    stem_layer_xy,
                    batch_lengths=batch_lengths_xy,
                    n_start_x=3,
                    n_start_y=3,
                    min_start_radius=min_start_radius,
                    max_start_radius=max_start_radius,
                    break_min_radius=min_radius,
                    break_max_radius=max_radius,
                    n_start_radius=3,
                    num_workers=self._num_workers,
                )
            else:
                circle_detector = Ransac(bandwidth=self._stem_search_circle_fitting_bandwidth)
                circle_detector.detect(
                    stem_layer_xy,
                    batch_lengths=batch_lengths_xy,
                    break_min_radius=min_radius,
                    break_max_radius=max_radius,
                    num_workers=self._num_workers,
                    seed=self._random_seed,
                )
            circle_detector.filter(
                max_circles=1,
                deduplication_precision=4,
                min_circumferential_completeness_idx=self._stem_search_circle_fitting_min_completeness_idx,
                circumferential_completeness_idx_max_dist=self._stem_search_circle_fitting_bandwidth,
                circumferential_completeness_idx_num_regions=int(365 / 5),
                non_maximum_suppression=True,
                num_workers=self._num_workers,
            )

            ellipses = None
            if self._stem_search_ellipse_fitting:
                ellipses = fit_ellipse(stem_layer_xy, batch_lengths_xy)

            visualization_tasks: List[Tuple[Any, ...]] = []

            batch_starts_xy = np.cumsum(np.concatenate((np.array([0], dtype=np.int64), batch_lengths_xy)))[:-1]
            batch_starts_circles = np.cumsum(
                np.concatenate((np.array([0], dtype=np.int64), circle_detector.batch_lengths_circles))
            )[:-1]
            for cluster_idx, label in enumerate(unique_cluster_labels):
                for layer in range(self._stem_search_circle_fitting_num_layers):
                    flat_idx = cluster_idx * num_layers + layer
                    batch_start_idx_xy = batch_starts_xy[flat_idx]
                    batch_end_idx_xy = batch_start_idx_xy + batch_lengths_xy[flat_idx]
                    circle_idx = batch_starts_circles[flat_idx]
                    if batch_lengths_xy[flat_idx] < self._stem_search_circle_fitting_min_points:
                        self._logger.info(
                            "Layer %d of stem cluster %d contains too few points to fit a circle or an ellipse.",
                            layer,
                            label,
                        )
                        continue

                    has_circle = circle_detector.batch_lengths_circles[flat_idx] > 0
                    has_ellipse = ellipses is not None and ellipses[flat_idx, 2] != -1

                    if has_ellipse:
                        # filter out ellipses if radius is outside the accepted range
                        radius_major, radius_minor = ellipses[flat_idx, 2:4]  # type: ignore[index]

                        if radius_minor / radius_major < self._stem_search_ellipse_filter_threshold:
                            has_ellipse = False

                    if not has_circle and not has_ellipse:
                        self._logger.info(
                            "Neither a circle nor an ellipse was found for layer %d of stem cluster %d.",
                            layer,
                            label,
                        )
                        continue

                    if has_circle:
                        layer_circles[cluster_idx, layer, :3] = circle_detector.circles[circle_idx]
                        if self._visualization_folder is not None and point_cloud_id is not None:
                            visualization_path = (
                                self._visualization_folder / point_cloud_id / f"circle_stem_{label}_layer_{layer}.png"
                            )
                            visualization_tasks.append(
                                (
                                    stem_layer_xy[batch_start_idx_xy:batch_end_idx_xy],
                                    visualization_path,
                                    circle_detector.circles[circle_idx],
                                )
                            )
                    if has_ellipse:
                        ellipse = ellipses[flat_idx]  # type: ignore[index]

                        layer_ellipses[cluster_idx, layer] = ellipse
                        if self._visualization_folder is not None and point_cloud_id is not None:
                            visualization_path = (
                                self._visualization_folder / point_cloud_id / f"ellipse_stem_{label}_layer_{layer}.png"
                            )
                            visualization_tasks.append(
                                (stem_layer_xy[batch_start_idx_xy:batch_end_idx_xy], visualization_path, None, ellipse)
                            )

        if len(visualization_tasks) > 0:
            with Profiler(
                "Visualization of circles and ellipses fitted to stem candidates",
                self._performance_tracker,
            ):
                self._logger.info("Visualize circles / ellipses fitted to stem candidates...")
                num_workers = self._num_workers if self._num_workers > 0 else multiprocessing.cpu_count()
                with multiprocessing.Pool(processes=num_workers) as pool:
                    pool.starmap(plot_fitted_shape, visualization_tasks)

        return (
            layer_circles,
            layer_ellipses,
            terrain_heights_at_cluster_positions,
            layer_heights,
            stem_layer_xy,
            batch_lengths_xy,
        )

    def fit_refined_circles_and_ellipses_to_stems(  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        self,
        stem_layer_xyz: FloatArray,
        preliminary_layer_circles: FloatArray,
        preliminary_layer_ellipses: FloatArray,
        terrain_heights_at_cluster_positions: FloatArray,
        point_cloud_id: Optional[str] = None,
    ) -> Tuple[
        FloatArray,
        FloatArray,
        FloatArray,
        LongArray,
    ]:
        r"""
        Given a set of point clusters that may represent individual tree stems, circles and ellipses are fitted to
        multiple horinzontal layers of each cluster. To obtain the horizontal layers,
        :code:`stem_search_circle_fitting_num_layers` horizontal layers with a height of
        :code:`stem_search_circle_fitting_layer_height` are created starting at a height of
        :code:`stem_search_circle_fitting_layer_start`. The layers have an overlap of
        :code:`stem_search_circle_fitting_layer_overlap` to the previous layer (the variables are constructor
        parameters).

        The points used for the circle and ellipse fitting in each layer are selected based on the results of a
        preliminary circle / ellipse fitting step. A buffer area is created around the outline of the respective circle
        or ellipse from the previous step. This buffer has a width of
        :code:`stem_search_circle_fitting_small_buffer_width` if the preliminary circle or ellipse diameter is less than
        or equal to :code:`stem_search_circle_fitting_switch_buffer_threshold` and otherwise
        :code:`stem_search_circle_fitting_large_buffer_width` (the variables are constructor parameters). All points
        from the respective layer that lie within the buffer area are selected and the fitting of the circles / ellipses
        is repeated using only these points.

        The ellipse fitting is only done if :code:`stem_search_ellipse_fitting` (constructor parameter) is set to
        :code:`True`. In the ellipse fitting, ellipses are only kept if the ratio of the radius along the semi-minor
        axis to the radius along the semi-major axis is greater than or equal to `stem_search_ellipse_filter_threshold`
        (constructor parameter).

        Args:
            stem_layer_xyz: Point coordinates of the points within the stem layer as defined by the constructor
                parameters :code:`stem_search_min_z` and :code:`stem_search_max_z`.
            preliminary_layer_circles: Parameters of the preliminary circles. Each circle must be represented by three
                values, namely the x- and y-coordinates of its center and its radius. If the preliminary circle fitting
                was unsucessfull for the respective layer, all values must be set to :code:`-1`.
            preliminary_layer_ellipses: Parameters of the preliminary ellipses.  Each ellipse must represented by five
                values, namely the x- and y-coordinates of its center, its radius along the semi-major and along the
                semi-minor axis, and the counterclockwise angle of rotation from the x-axis to the semi-major axis of
                the ellipse. If the preliminary circle fitting was unsucessfull for the respective layer, all values
                must be set to :code:`-1`.
            terrain_heights_at_cluster_positions: Terrain height at the centroid position of each cluster.
            point_cloud_id: ID of the point cloud to be used in the file names of the visualization results. If set to
                :code:`None`, no visualizations are created.

        Returns:
            :Tuple of four arrays:
                - Parameters of the circles that were fitted to the layers of each cluster. Each circle is represented
                  by three values, namely the x- and y-coordinates of its center and its radius. If the circle fitting
                  is not successful for a layer, all parameters are set to :code:`-1`.
                - Parameters of the ellipses that were fitted to the layers of each cluster. Each ellipse is represented
                  by five values, namely the x- and y-coordinates of its center, its radius along the semi-major and
                  along the semi-minor axis, and the counterclockwise angle of rotation from the x-axis to the
                  semi-major axis of the ellipse. If the ellipse fitting is not successful for a layer or results in an
                  ellipse whose axis ratio is smaller than :code:`stem_search_ellipse_filter_threshold` (constructor
                  parameter), all parameters are set to :code:`-1`.
                - X- and y-coordinates of the points in each horizontal layer of each cluster that were selected for the
                  circle and ellipse fitting in that layer based on the preliminary circles or ellipses. Points
                  belonging to the same layer of the same cluster are stored consecutively.
                - Number of points belonging to each horizontal layer of each cluster.

        Shape:
            - :code:`stem_layer_xyz`: :math:`(N, 3)`
            - :code:`preliminary_layer_circles`: :math:`(S, L, 3)`
            - :code:`preliminary_layer_ellipses`: :math:`(S, L, 5)`
            - :code:`terrain_heights`: :math:`(S)`
            - Output: :math:`(S, L, 3)`, :math:`(S, L, 5)`, :math:`(L)`, :math:`(N_{0,0} + ... + N_{S,L}, 2)`.

            | where
            |
            | :math:`N = \text{ number of points in the stem layer}`
            | :math:`S = \text{ number of stem clusters}`
            | :math:`L = \text{ number of horinzontal layers to which circles and ellipses are fitted}`
            | :math:`N_{s,l} = \text{ number of points selected from the l-th layer of cluster } s`
        """

        with Profiler("Fitting of refined circles and ellipses to stem candidates", self._performance_tracker):
            self._logger.info("Fitting of refined circles / ellipses to stem candidates...")

            num_clusters = len(preliminary_layer_circles)
            num_layers = self._stem_search_circle_fitting_num_layers

            layer_circles = np.full(
                (num_clusters, self._stem_search_circle_fitting_num_layers, 3),
                fill_value=-1,
                dtype=stem_layer_xyz.dtype,
            )
            layer_ellipses = np.full(
                (num_clusters, self._stem_search_circle_fitting_num_layers, 5),
                fill_value=-1,
                dtype=stem_layer_xyz.dtype,
            )

            if not stem_layer_xyz.flags.f_contiguous:
                stem_layer_xyz = stem_layer_xyz.copy(order="F")
            preliminary_layer_circles = preliminary_layer_circles.reshape((-1, 3)).astype(
                stem_layer_xyz.dtype, order="F"
            )
            preliminary_layer_ellipses = preliminary_layer_ellipses.reshape((-1, 5)).astype(
                stem_layer_xyz.dtype, order="F"
            )

            stem_layers_xy, batch_lengths_xy = collect_inputs_stem_layers_refined_fitting_cpp(
                stem_layer_xyz,
                preliminary_layer_circles,
                preliminary_layer_ellipses,
                terrain_heights_at_cluster_positions,
                float(self._stem_search_circle_fitting_layer_start),
                int(num_layers),
                float(self._stem_search_circle_fitting_layer_height),
                float(self._stem_search_circle_fitting_layer_overlap),
                float(self._stem_search_circle_fitting_switch_buffer_threshold),
                float(self._stem_search_circle_fitting_small_buffer_width),
                float(self._stem_search_circle_fitting_large_buffer_width),
                0,
                int(self._num_workers),
            )

            if num_clusters == 0:
                return (
                    layer_circles,
                    layer_ellipses,
                    np.empty((0, 2), dtype=stem_layer_xyz.dtype),
                    np.empty(0, dtype=np.int64),
                )

            min_radius = self._stem_search_circle_fitting_min_stem_diameter / 2
            max_radius = self._stem_search_circle_fitting_max_stem_diameter / 2
            min_start_radius = min_radius + min(
                2 * self._stem_search_circle_fitting_bandwidth, (max_radius - min_radius) / 4
            )
            max_start_radius = max_radius - min(
                2 * self._stem_search_circle_fitting_bandwidth, (max_radius - min_radius) / 4
            )

            with Profiler("Fitting of refined circles to stem candidates", self._performance_tracker):
                self._logger.info("Fit refined circles...")

                circle_detector: Union[MEstimator, Ransac]
                if self._stem_search_circle_fitting_method == "m-estimator":
                    circle_detector = MEstimator(
                        bandwidth=self._stem_search_circle_fitting_bandwidth,
                        break_min_change=1e-6,
                        min_step_size=1e-10,
                        max_iterations=300,
                        armijo_min_decrease_percentage=0.5,
                        armijo_attenuation_factor=0.25,
                    )
                    circle_detector.detect(
                        stem_layers_xy,
                        batch_lengths=batch_lengths_xy,
                        n_start_x=3,
                        n_start_y=3,
                        min_start_radius=min_start_radius,
                        max_start_radius=max_start_radius,
                        break_min_radius=min_radius,
                        break_max_radius=max_radius,
                        n_start_radius=3,
                        num_workers=self._num_workers,
                    )
                else:
                    circle_detector = Ransac(bandwidth=self._stem_search_circle_fitting_bandwidth)
                    circle_detector.detect(
                        stem_layers_xy,
                        batch_lengths=batch_lengths_xy,
                        break_min_radius=min_radius,
                        break_max_radius=max_radius,
                        num_workers=self._num_workers,
                        seed=self._random_seed,
                    )

                circle_detector.filter(
                    max_circles=1,
                    deduplication_precision=4,
                    min_circumferential_completeness_idx=self._stem_search_circle_fitting_min_completeness_idx,
                    circumferential_completeness_idx_max_dist=self._stem_search_circle_fitting_bandwidth,
                    circumferential_completeness_idx_num_regions=int(365 / 5),
                    non_maximum_suppression=True,
                    num_workers=self._num_workers,
                )

            with Profiler("Fitting of refined ellipses to stem candidates", self._performance_tracker):
                self._logger.info("Fit refined ellipses...")

            ellipses = None
            if self._stem_search_ellipse_fitting:
                ellipses = fit_ellipse(stem_layers_xy, batch_lengths_xy, num_workers=self._num_workers)

            visualization_tasks: List[Tuple[Any, ...]] = []

            circle_idx = 0
            batch_start_idx = 0
            for cluster_idx in range(num_clusters):
                for layer in range(self._stem_search_circle_fitting_num_layers):
                    flat_idx = cluster_idx * num_layers + layer
                    batch_end_idx = batch_start_idx + batch_lengths_xy[flat_idx]

                    if circle_detector.batch_lengths_circles[flat_idx] > 0:
                        layer_circles[cluster_idx, layer] = circle_detector.circles[circle_idx]

                        if self._visualization_folder is not None and point_cloud_id is not None:
                            visualization_path = (
                                self._visualization_folder
                                / point_cloud_id
                                / f"refined_circle_stem_{cluster_idx}_layer_{layer}.png"
                            )
                            visualization_tasks.append(
                                (
                                    stem_layers_xy[batch_start_idx:batch_end_idx],
                                    visualization_path,
                                    circle_detector.circles[circle_idx],
                                )
                            )

                    has_ellipse = ellipses is not None and ellipses[flat_idx, 2] != -1
                    if has_ellipse:
                        radius_major, radius_minor = ellipses[flat_idx, 2:4]

                        if radius_minor / radius_major >= self._stem_search_ellipse_filter_threshold:
                            layer_ellipses[cluster_idx, layer] = ellipses[flat_idx]  # type: ignore[index]
                        else:
                            has_ellipse = False

                        if self._visualization_folder is not None and point_cloud_id is not None and has_ellipse:
                            visualization_path = (
                                self._visualization_folder
                                / point_cloud_id
                                / f"refined_ellipse_stem_{cluster_idx}_layer_{layer}.png"
                            )
                            visualization_tasks.append(
                                (
                                    stem_layers_xy[batch_start_idx:batch_end_idx],
                                    visualization_path,
                                    None,
                                    ellipses[flat_idx],  # type: ignore[index]
                                )
                            )

                    circle_idx += circle_detector.batch_lengths_circles[flat_idx]
                    batch_start_idx = batch_end_idx

        if len(visualization_tasks) > 0:
            with Profiler(
                "Visualization of refined circles and ellipses fitted to stem candidates",
                self._performance_tracker,
            ):
                self._logger.info("Visualize refined circles / ellipses fitted to stem candidates...")
                num_workers = self._num_workers if self._num_workers > 0 else multiprocessing.cpu_count()
                with multiprocessing.Pool(processes=num_workers) as pool:
                    pool.starmap(plot_fitted_shape, visualization_tasks)

        return layer_circles, layer_ellipses, stem_layers_xy, batch_lengths_xy

    def filter_instances_stem_layers_std(  # pylint: disable=too-many-locals
        self, layer_circles: FloatArray, layer_ellipses: FloatArray
    ) -> Tuple[BoolArray, LongArray, LongArray]:
        r"""
        Filters the point clusters that may represent individual tree stems based on the circles and ellipses fitted to
        different horizontal layers of the clusters. For each cluster, the standard deviation of the fitted circle or
        ellipse diameters is computed for all possible combinations of
        :code:`stem_search_circle_fitting_std_num_layers` layers. If for any of the combinations the standard
        deviation of the circle diameters is smaller than or equal to
        :code:`stem_search_circle_fitting_max_std_diameter`, the cluster is kept. Otherwise, it is checked if for any of
        the combinations the standard deviation of the ellipse diameters is smaller than or equal to
        :code:`stem_search_circle_fitting_max_std_diameter`. If that is also not the case, the cluster is discarded.

        If :code:`stem_search_circle_fitting_max_std_position` is not :code:`None`, the standard deviation of the x- and
        y-coordinates of the circle / ellipse centers of the selected combination of layers must be smaller than or
        equal to the given threshold to keep the stem cluster (the variables are constructor parameters).

        Args:
            layer_circles: Parameters of the circles that were fitted to the horizontal layers of each cluster. Each
                circle must be represented by three values, namely the x- and y-coordinates of its center and its
                radius. If no circle was found for a certain layer, the circle parameters for that layer must be set to
                :code:`-1`.
            layer_ellipses: Parameters of the ellipses that were fitted to the horizontal layers of each cluster. Each
                ellipse must represented by five values, namely the x- and y-coordinates of its center, its
                radius along the semi-major and along the semi-minor axis, and the counterclockwise angle of rotation
                from the x-axis to the semi-major axis of the ellipse. If no ellipse was found for a certain layer, the
                ellipse parameters for that layer must be set to :code:`-1`.

        Returns:
            :Tuple of three arrays:
                - Boolean mask indicating which stem clusters are retained after the filtering.
                - Indices of the combination of layers with the lowest standard deviation of the circle diameters for
                  each cluster. If for a stem cluster, no combination of layers was found for which the fitted circles
                  fulfill the filtering criteria, the indices for that cluster are set to :code:`-1`.
                - Indices of the combination of layers with the lowest standard deviation of the ellipse diameters for
                  each cluster. If for a stem cluster, no combination of layers was found for which the fitted ellipses
                  fulfill the filtering criteria, the indices for that cluster are set to :code:`-1`.

        Shape:
            - :code:`layer_circles`: :math:`(S, L, 3)`
            - :code:`layer_ellipses`: :math:`(S, L, 5)`
            - Output: :math:`(S)`, :math:`(S', L')`, :math:`(S', L')`.

            | where
            |
            | :math:`S = \text{ number of stem clusters before the filtering}`
            | :math:`S' = \text{ number of stem clusters after the filtering}`
            | :math:`L = \text{ number of horinzontal layers to which circles and ellipses are fitted}`
            | :math:`L' = \text{ number of horinzontal layers considered for filtering}`
        """

        num_instances = len(layer_circles)
        num_layers = layer_circles.shape[1]

        filter_mask = np.zeros(num_instances, dtype=bool)
        best_circle_combination = np.full(
            (num_instances, self._stem_search_circle_fitting_std_num_layers), dtype=np.int64, fill_value=-1
        )
        best_ellipse_combination = np.full(
            (num_instances, self._stem_search_circle_fitting_std_num_layers), dtype=np.int64, fill_value=-1
        )

        for label in range(num_instances):
            existing_circle_layers = np.arange(num_layers, dtype=np.int64)[layer_circles[label, :, 2] != -1]
            existing_ellipse_layers = np.arange(num_layers, dtype=np.int64)[layer_ellipses[label, :, 2] != -1]

            if (
                len(existing_circle_layers) < self._stem_search_circle_fitting_std_num_layers
                and len(existing_ellipse_layers) < self._stem_search_circle_fitting_std_num_layers
            ):
                continue

            if len(existing_circle_layers) >= self._stem_search_circle_fitting_std_num_layers:
                circle_diameters = layer_circles[label, :, 2] * 2
                combinations = np.array(
                    list(
                        itertools.combinations(existing_circle_layers, self._stem_search_circle_fitting_std_num_layers)
                    )
                )
                minimum_std = np.inf
                for combination in combinations:
                    diameter_std = np.std(circle_diameters[combination])
                    position_std = np.zeros(2, dtype=diameter_std.dtype)
                    if self._stem_search_circle_fitting_max_std_position is not None:
                        position_std = np.std(layer_circles[label, combination, :2], axis=0)
                    if (
                        diameter_std <= self._stem_search_circle_fitting_max_std_diameter
                        and (position_std <= self._stem_search_circle_fitting_max_std_position).all()
                    ):
                        filter_mask[label] = True
                        if diameter_std < minimum_std:
                            minimum_std = diameter_std
                            best_circle_combination[label] = combination
            if not filter_mask[label]:
                ellipse_diameters = (layer_ellipses[label, :, 2:4]).sum(axis=-1)
                combinations = np.array(
                    list(
                        itertools.combinations(existing_ellipse_layers, self._stem_search_circle_fitting_std_num_layers)
                    )
                )
                minimum_std = np.inf
                for combination in combinations:
                    diameter_std = np.std(ellipse_diameters[combination])

                    position_std = np.zeros(2, dtype=diameter_std.dtype)
                    if self._stem_search_circle_fitting_max_std_position is not None:
                        position_std = np.std(layer_ellipses[label, combination, :2], axis=0)

                    if (
                        diameter_std <= self._stem_search_circle_fitting_max_std_diameter
                        and (position_std <= self._stem_search_circle_fitting_max_std_position).all()
                    ):
                        filter_mask[label] = True
                        if diameter_std < minimum_std:
                            minimum_std = diameter_std
                            best_ellipse_combination[label] = combination

        return filter_mask, best_circle_combination[filter_mask], best_ellipse_combination[filter_mask]

    def rename_visualizations_after_filtering(self, filter_mask: BoolArray, point_cloud_id: Optional[str]) -> None:
        r"""
        Renames visualization files that plot the circles / ellipses fitted to stem layers after the filtering of the
        stems. In the course of this, the stem IDs in the file names are updated and the postfix :code:`_valid` is
        added to the file names for stem clusters that were kept during filtering, while the postfix :code:`_invalid` is
        added to the file names for stem clusters that were filtered out.

        Args:
            filter_mask: Boolean mask indicating which stems were kept during filtering.
            point_cloud_id: ID of the point cloud used in the file names of the visualizations. If set to
                :code:`None`, it is assumed that no visualizations were created.

        Shape:
            - :code:`filter_mask`: :math:`(S)`

            | where
            |
            | :math:`S = \text{ number of stem clusters}`
        """

        if self._visualization_folder is None or point_cloud_id is None:
            return

        next_valid_label = 0
        next_invalid_label = filter_mask.sum()
        for label, is_valid in enumerate(filter_mask):
            for layer in range(self._stem_search_circle_fitting_num_layers):
                visualization_paths = [
                    self._visualization_folder / point_cloud_id / f"circle_stem_{label}_layer_{layer}.png",
                    self._visualization_folder / point_cloud_id / f"ellipse_stem_{label}_layer_{layer}.png",
                    self._visualization_folder / point_cloud_id / f"refined_circle_stem_{label}_layer_{layer}.png",
                    self._visualization_folder / point_cloud_id / f"refined_ellipse_stem_{label}_layer_{layer}.png",
                ]
                for visualization_path in visualization_paths:
                    if not visualization_path.exists():
                        continue

                    if is_valid:
                        new_visualization_path_str = str(visualization_path)
                        new_visualization_path_str = new_visualization_path_str.replace(
                            f"stem_{label}_", f"stem_{next_valid_label}_"
                        )
                        new_visualization_path_str = new_visualization_path_str.replace(
                            f"layer_{layer}.png", f"layer_{layer}_valid.png"
                        )
                    else:
                        new_visualization_path_str = str(visualization_path)
                        new_visualization_path_str = new_visualization_path_str.replace(
                            f"stem_{label}_", f"stem_{next_invalid_label}_"
                        )
                        new_visualization_path_str = new_visualization_path_str.replace(
                            f"layer_{layer}.png", f"layer_{layer}_invalid.png"
                        )
                    new_visualization_path = Path(new_visualization_path_str)

                    if new_visualization_path.exists():
                        new_visualization_path.unlink()
                    visualization_path.rename(new_visualization_path)

            if is_valid:
                next_valid_label += 1
            else:
                next_invalid_label += 1

    def compute_stem_positions(  # pylint: disable=too-many-locals
        self,
        layer_circles: FloatArray,
        layer_ellipses: FloatArray,
        layer_heights: FloatArray,
        best_circle_combination: LongArray,
        best_ellipse_combination: LongArray,
    ) -> FloatArray:
        r"""
        Calculates the stem positions using the circles or ellipses fitted to multiple horizontal layers of the stems.
        For this purpose, the combination of :code:`stem_search_circle_fitting_std_num_layers` (constructor parameter)
        circles or ellipses with the smallest standard deviation of the diameters is selected. A linear model is fitted
        to these circles or ellipses to predict the centers of the circles or ellipses as a function of the height above
        the ground. The prediction of the linear model for a height of 1.3 m above the ground is returned as am estimate
        of the stem position at breast height.

        Args:
            layer_circles: Parameters of the circles that were fitted to the horizontal layers of each cluster. Each
                circle must be represented by three values, namely the x- and y-coordinates of its center and its
                radius. If no circle was found for a certain layer, the circle parameters for that layer must be set to
                :code:`-1`.
            layer_ellipses: Parameters of the ellipses that were fitted to the horizontal layers of each cluster. Each
                ellipse must represented by five values, namely the x- and y-coordinates of its center, its
                radius along the semi-major and along the semi-minor axis, and the counterclockwise angle of rotation
                from the x-axis to the semi-major axis of the ellipse. If no ellipse was found for a certain layer, the
                ellipse parameters for that layer must be set to :code:`-1`.
            layer_heights: Heights above the ground of the midpoints of the horizontal layers to which the circles or
                ellipses were fitted.
            best_circle_combination: Indices of the combination of layers with the lowest standard deviation of the
                circle diameters for each stem cluster. If less than
                :code:`stem_search_circle_fitting_std_num_layers` (constructor parameter) circles were found for a stem
                cluster, the indices for that cluster must be set to :code:`-1`.
            best_ellipse_combination: Indices of the combination of layers with the lowest standard deviation of the
                ellipse diameters for each cluster. If more than
                :code:`stem_search_circle_fitting_std_num_layers` (constructor parameter) circles were found for a stem
                cluster, the ellipses are not considered for calculating the stem position.

        Returns:
            X- and y-coordinates of the position of each stem at breast height (1.3 m).

        Shape:
            - :code:`layer_circles`: :math:`(S, L, 3)`
            - :code:`layer_ellipses`: :math:`(S, L, 5)`
            - :code:`layer_heights`: :math:`(L)`
            - :code:`best_circle_combination`: :math:`(S, L')`
            - :code:`best_ellipse_combination`: :math:`(S, L')`
            - Output: :math:`(S, 2)`.

            | where
            |
            | :math:`S = \text{ number of detected stems}`
            | :math:`L = \text{ number of horinzontal layers to which circles and ellipses are fitted}`
            | :math:`L' = \text{ number of horinzontal layers considered for filtering}`
        """

        num_instances = len(layer_circles)

        stem_positions = np.empty((num_instances, 2), dtype=layer_circles.dtype, order="F")

        for label in range(num_instances):
            has_circle_combination = best_circle_combination[label, 0] != -1
            if has_circle_combination:
                best_combination = best_circle_combination[label]
                circles_or_ellipses = layer_circles[label, best_combination]
            else:
                best_combination = best_ellipse_combination[label]
                circles_or_ellipses = layer_ellipses[label, best_combination]

            layer_heights_combination = layer_heights[best_combination]

            centers = circles_or_ellipses[:, :2]

            prediction_x, _ = estimate_with_linear_model(
                layer_heights_combination, centers[:, 0], np.array([1.3], dtype=layer_circles.dtype)
            )
            prediction_y, _ = estimate_with_linear_model(
                layer_heights_combination, centers[:, 1], np.array([1.3], dtype=layer_circles.dtype)
            )
            stem_positions[label, 0] = prediction_x[0]
            stem_positions[label, 1] = prediction_y[0]

        return stem_positions

    def compute_stem_diameters(  # pylint: disable=too-many-locals,
        self,
        layer_circles: FloatArray,
        layer_ellipses: FloatArray,
        layer_heights: FloatArray,
        stem_layer_xy: FloatArray,
        batch_lengths_xy: LongArray,
        best_circle_combination: LongArray,
        best_ellipse_combination: LongArray,
        *,
        point_cloud_id: Optional[str] = None,
    ) -> FloatArray:
        r"""
        Calculates the stem diameters using the circles or ellipses fitted to multiple horizontal layers of the stems.
        For this purpose, the combination of :code:`stem_search_circle_fitting_std_num_layers` (constructor parameter)
        circles or ellipses with the smallest standard deviation of the diameters is selected. The stem diameter for
        each selected layer is computed by fitting a GAM to the points of that layer, using the centers of the
        previously fitted circles or ellipses to normalize the points. A linear model is then fitted to the stem
        diameters obtained from the GAM to predict the stem diameter as a function of the height above the ground. The
        prediction of the linear model for a height of 1.3 m above the ground is returned as an estimate of the stem
        diameter at breast height.

        Args:
            layer_circles: Parameters of the circles that were fitted to the horizontal layers of each cluster. Each
                circle must be represented by three values, namely the x- and y-coordinates of its center and its
                radius. If no circle was found for a certain layer, the circle parameters for that layer must be set to
                :code:`-1`.
            layer_ellipses: Parameters of the ellipses that were fitted to the horizontal layers of each cluster. Each
                ellipse must represented by five values, namely the x- and y-coordinates of its center, its
                radius along the semi-major and along the semi-minor axis, and the counterclockwise angle of rotation
                from the x-axis to the semi-major axis of the ellipse. If no ellipse was found for a certain layer, the
                ellipse parameters for that layer must be set to :code:`-1`.
            layer_heights: Heights above the ground of the midpoints of the horizontal layers to which the circles or
                ellipses were fitted.
            stem_layer_xy: X- and y-coordinates of the points belonging to the different horizontal layers of the
                stems. Points that belong to the same layer of the same stem must be stored consecutively and the
                number of points belonging to each layer must be specified using :code:`batch_lengths_xy`.
            batch_lengths_xy: Number of points belonging to each horizontal layer of each stem.
            best_circle_combination: Indices of the combination of layers with the lowest standard deviation of the
                circle diameters for each stem cluster. If no valid combination of circles was found for a stem
                cluster, the indices for that cluster must be set to :code:`-1`.
            best_ellipse_combination: Indices of the combination of layers with the lowest standard deviation of the
                ellipse diameters for each cluster. If a valid combination of circles was found for a stem cluster, the
                ellipses are not considered for calculating the stem position. If no valid combination of ellipse was
                found for a stem cluster, the indices for that cluster must be set to :code:`-1`.
            point_cloud_id: ID of the point cloud to be used in the file names of the visualization results. If set to
                :code:`None`, no visualizations are created.

        Returns:
            Diameters at breast height of each stem (1.3 m).

        Shape:
            - :code:`layer_circles`: :math:`(S, L, 3)`
            - :code:`layer_ellipses`: :math:`(S, L, 5)`
            - :code:`layer_heights`: :math:`(L)`
            - :code:`stem_layer_xy`: :math:`(N, 2)`
            - :code:`batch_lengths_xy`: :math:`(S \cdot L)`
            - :code:`best_circle_combination`: :math:`(S, L')`
            - :code:`best_ellipse_combination`: :math:`(S, L')`
            - Output: :math:`(S)`.

            | where
            |
            | :math:`N = \text{ number of points in the stem layer}`
            | :math:`S = \text{ number of detected stems}`
            | :math:`L = \text{ number of horinzontal layers to which circles and ellipses are fitted}`
            | :math:`L' = \text{ number of horinzontal layers considered for filtering}`
        """

        num_instances = len(layer_circles)
        len_layer_combination = best_circle_combination.shape[1]
        num_layers = self._stem_search_circle_fitting_num_layers

        stem_diameters = np.empty(num_instances, dtype=stem_layer_xy.dtype)

        visualization_tasks = []

        batch_starts = np.cumsum(np.concatenate((np.array([0], dtype=np.int64), batch_lengths_xy)))[:-1]
        for label in range(num_instances):
            has_circle_combination = best_circle_combination[label, 0] != -1
            if has_circle_combination:
                best_combination = best_circle_combination[label]
                circles_or_ellipses = layer_circles[label, best_combination]
            else:
                best_combination = best_ellipse_combination[label]
                circles_or_ellipses = layer_ellipses[label, best_combination]

            layer_heights_combination = layer_heights[best_combination]
            centers = circles_or_ellipses[:, :2]

            layer_diameters = np.empty(len_layer_combination, dtype=stem_layer_xy.dtype)

            for layer_idx, layer in enumerate(best_combination):
                flat_idx = label * num_layers + layer
                batch_start_idx = batch_starts[flat_idx]
                batch_end_idx = batch_start_idx + batch_lengths_xy[flat_idx]
                diameter_gam = None
                polygon_vertices = None
                if batch_start_idx < batch_end_idx:
                    diameter_gam, polygon_vertices = self.stem_diameter_estimation_gam(
                        stem_layer_xy[batch_start_idx:batch_end_idx], centers[layer_idx]
                    )
                if diameter_gam is not None:
                    layer_diameters[layer_idx] = diameter_gam
                else:
                    if has_circle_combination:
                        layer_diameters[layer_idx] = circles_or_ellipses[layer_idx, 2] * 2
                    else:
                        layer_diameters[layer_idx] = circles_or_ellipses[layer_idx, 2:4].sum()

                if (
                    polygon_vertices is not None
                    and self._visualization_folder is not None
                    and point_cloud_id is not None
                ):
                    if diameter_gam is not None:
                        file_name = f"gam_stem_{label}_layer_{layer}.png"
                    else:
                        file_name = f"gam_stem_{label}_layer_{layer}_invalid.png"

                    visualization_path = self._visualization_folder / point_cloud_id / file_name
                    visualization_tasks.append(
                        (
                            stem_layer_xy[batch_start_idx:batch_end_idx],
                            visualization_path,
                            None,
                            None,
                            polygon_vertices,
                        )
                    )

            prediction, _ = estimate_with_linear_model(
                layer_heights_combination, layer_diameters, np.array([1.3], dtype=stem_layer_xy.dtype)
            )
            stem_diameters[label] = prediction[0]

        if len(visualization_tasks) > 0:
            num_workers = self._num_workers if self._num_workers > 0 else multiprocessing.cpu_count()
            with multiprocessing.Pool(processes=num_workers) as pool:
                pool.starmap(plot_fitted_shape, visualization_tasks)

        return stem_diameters

    def stem_diameter_estimation_gam(  # pylint: disable=too-many-locals
        self,
        points: FloatArray,
        center: FloatArray,
    ) -> Tuple[Optional[float], FloatArray]:
        r"""
        Estimates the diameter of a tree stem at a certain height using a generalized additive model (GAM). It is
        assumed that a circle or an ellipse has already been fitted to the points of the tree stem in a layer around the
        respective height. To create the GAM, the points are converted into polar coordinates, using the center of the
        previously fitted circle or ellipse as the coordinate origin. The GAM is then fitted to predict the radius of
        the points based on the angles. The fitted GAM is then used to predict the stem radii in one-degree intervals.
        From these predictions, the stem's boundary polygon is constructed and the stem diameter is computed from the
        area of the boundary polygon. Assuming the boundary polygon is approximately circular, the stem diameter is
        calculated using the formula for a circle's diameter.

        .. math::

            d = 2 \cdot \sqrt{\frac{A_{polygon}}{\pi}}

        If any of the predicted radii is negative or the difference between the minimum
        and maximum of the predicted radii is greater than the value of :code:`stem_search_gam_max_radius_diff`
        (constructor parameter), the fitted GAM is considered invalid, and :code:`None` is returned for the stem
        diameter. In this case the diameter of the previously fitted circle or ellipse can be used as a more robust
        estimate of the stem diameter.

        Args:
            points: Points belonging to the stem layer for which to estimate the diameter.
            center: Center of the circle or ellipse that has been fitted to the stem layer.

        Returns:
            : Tuple with two elements:
                - Estimated stem diameter. The estimated stem diameter may be :code:`None` if the fitted GAM is invalid.
                - Array containing the sorted vertices of the stem's boundary polygon predicted by the GAM as cartesian
                  coordinates.

        Shape:
            - :code:`points`: :math:`(N, 2)` or :math:`(N, 3)`
            - :code:`center`: :math:`(2)`

            | where
            |
            | :math:`N = \text{ number of points}`
        """

        points_centered = points[:, :2] - center.reshape((-1, 2))

        # calculate polar coordinates
        polar_radius = np.linalg.norm(points_centered[:, :2], axis=-1)

        # add small random offset to avoid perfect separation
        polar_radius += self._random_generator.normal(0, 1e-8, len(points))

        polar_angle = np.arctan2(points_centered[:, 1], points_centered[:, 0])

        # fit GAM
        polar_xy = np.column_stack((polar_angle, polar_radius))
        gam = LinearGAM(s(0, basis="cp", edge_knots=[-np.pi, np.pi])).fit(polar_xy[:, 0], polar_xy[:, 1])
        del polar_xy

        # predict stem outline using fitted GAM
        polar_angles = np.asarray([-np.pi + 2 * np.pi * k / 360 for k in range(360)])
        polar_radii = gam.predict(polar_angles)

        cartesian_coords_x = polar_radii * np.cos(polar_angles)
        cartesian_coords_y = polar_radii * np.sin(polar_angles)

        cartesian_coords = np.column_stack((cartesian_coords_x, cartesian_coords_y))
        cartesian_coords = cartesian_coords + center.reshape((-1, 2))

        radius_diff = polar_radii.max() - polar_radii.min()

        if (
            self._stem_search_gam_max_radius_diff is not None and radius_diff > self._stem_search_gam_max_radius_diff
        ) or (polar_radii < 0).any():
            return None, cartesian_coords

        stem_area = polygon_area(cartesian_coords_x, cartesian_coords_y)
        diameter_gam = 2 * np.sqrt(stem_area / np.pi)

        return diameter_gam, cartesian_coords

    def segment_crowns(
        self,
        xyz: FloatArray,
        dists_to_dtm: FloatArray,
        is_tree: BoolArray,
        stem_positions: FloatArray,
        stem_diameters: FloatArray,
        stem_labels: LongArray,
    ) -> LongArray:
        r"""
        Computes a point-wise segmentation of the individual trees using a region growing procedure. In the first step,
        the region growing procedure selects an initial set of seed points for each tree. These should be points
        that are very likely to belong to the corresponding tree. In an iterative process, the sets of points assigned
        to each tree are then expanded. In each iteration, the neighboring points of each seed point within a certain
        search radius are determined. Neighboring points that are not yet assigned to any tree are added to the same
        tree as the seed point and become seed points in the next iteration. The region growing continues until there
        are no more seed points to be processed or the maximum number of iterations is reached.

        To select the initial seed points for a given tree, the following approach is used: (1) All points that were
        assigned to the respective stem during the stem detection stage are used as seed points. (2) Additionally, a
        cylinder with a height of :code:`tree_seg_seed_layer_height` (constructor parameter) and a diameter of
        :code:`tree_seg_seed_diameter_factor * d` (constructor parameter) is considered, where :code:`d` is the tree's
        stem diameter at breast height, which has been computed in the previous step. The cylinder's center is
        positioned at the stem center at breast height, which also has been computed in the previous stage. All points
        within the cylinder that have not yet been selected as seed points for other trees are selected as seed points.

        The search radius for the iterative region growing procedure is set as follows: First, the search radius is set
        to the voxel size used for voxel-based subsampling, which is done before starting the region growing procedure.
        The search radius is increased by the voxel size if one of the following conditions is fulfilled at the end of a
        region growing iteration:

        - The ratio between the number of points newly assigned to trees in the iteration and the number of remaining,
          unassigned points is below :code:`tree_seg_min_total_assignment_ratio`.
        - The ratio between the number of trees to which new points have been assigned in the iteration and the total
          number of trees is below :code:`tree_seg_min_tree_assignment_ratio`.

        The search radius is increased up to a maximum radius of :code:`tree_seg_max_search_radius`. If the
        search radius has not been increased for :code:`tree_seg_decrease_search_radius_after_num_iter`, it
        is reduced by the voxel size (the variables are constructor parameters).

        To promote upward growth, the z-coordinates of the points are divided by :code:`tree_seg_z_scale` before
        the region growing.

        Since the terrain filtering in the first step of the algorithm may be inaccurate and some tree points may be
        falsely classified as terrain points, both terrain and non-terrain points are considered by the region growing
        procedure. However, to prevent large portions of terrain points from being included in tree instances, terrain
        points are only assigned to a tree if their cumulative search distance from the initial seed point is below the
        threshold defined by :code:`tree_seg_cum_search_dist_include_ground` (constructor parameter). The cumulative
        search distance is defined as the total distance traveled between consecutive points until reaching a terrain
        point.

        Args:
            xyz: Coordinates of the points which to consider in the region growing procedure. This can include both
                terrain and non-terrain points.
            dists_to_dtm: Height of each point above the ground.
            is_tree: Boolean array indicating which points have been identified as potential tree points in the terrain
                classification step. The points for which the corresponding entry is :code:`True` are considered in
                all region growing iterations, while terrain points are only considered if the cumulative search
                distance is below the threshold defined by :code:`tree_seg_cum_search_dist_include_ground` (constructor
                parameter).
            stem_positions: X- and y-coordinates of the positions of the stems to be used for seed point selection.
            stem_diameters: Diameters of of the stems to be used for seed point selection.
            stem_labels: Cluster labels for each point that represent the detected stems. Points that do not belong to
                any stem must have the label :code:`-1`.

        Returns:
            Tree instance labels for all points. For points not belonging to any tree, the label is set to
            :code:`invalid_tree_id` (constructor parameter).

        Raises:
            ValueError: If :code:`xyz`, :code:`dists_to_dtm`, and :code:`is_tree` have different lengths.
            ValueError: If :code:`tree_positions` and :code:`stem_diameters` have different lengths.

        Shape:
            - :code:`xyz`: :math:`(N, 3)`
            - :code:`dists_to_dtm`: :math:`(N)`
            - :code:`is_tree`: :math:`(N)`
            - :code:`tree_positions`: :math:`(S, 2)`
            - :code:`stem_diameters`: :math:`(S)`
            - :code:`cluster_labels`: :math:`(N)`
            - Output: :math:`(N)`

            | where
            |
            | :math:`N = \text{ number of points}`
            | :math:`S = \text{ number of detected stems}`
        """

        if len(xyz) != len(dists_to_dtm):
            raise ValueError("xyz and dists_to_dtm must have the same length.")
        if len(xyz) != len(is_tree):
            raise ValueError("xyz and is_tree must have the same length.")
        if len(xyz) != len(stem_labels):
            raise ValueError("xyz and stem_labels must have the same length.")

        downsampled_xyz, downsampled_indices, inverse_indices = voxel_downsampling(
            xyz, voxel_size=self._tree_seg_voxel_size
        )
        dists_to_dtm = dists_to_dtm[downsampled_indices]
        is_tree = is_tree[downsampled_indices]
        stem_labels = stem_labels[downsampled_indices]
        del downsampled_indices

        if not downsampled_xyz.flags.f_contiguous:
            downsampled_xyz = downsampled_xyz.copy(order="F")
        if not stem_positions.flags.f_contiguous:
            stem_positions = stem_positions.copy(order="F")
        dists_to_dtm = dists_to_dtm.astype(downsampled_xyz.dtype)
        stem_positions = stem_positions.astype(downsampled_xyz.dtype)
        stem_diameters = stem_diameters.astype(downsampled_xyz.dtype)

        instance_ids = segment_tree_crowns_cpp(
            downsampled_xyz,
            dists_to_dtm,
            is_tree,
            stem_positions,
            stem_diameters,
            stem_labels,
            float(self._tree_seg_voxel_size),
            float(self._tree_seg_z_scale),
            float(self._tree_seg_seed_layer_height),
            float(self._tree_seg_seed_diameter_factor),
            float(self._tree_seg_seed_min_diameter),
            float(self._tree_seg_min_total_assignment_ratio),
            float(self._tree_seg_min_tree_assignment_ratio),
            float(self._tree_seg_max_search_radius),
            int(self._tree_seg_decrease_search_radius_after_num_iter),
            int(self._tree_seg_max_iterations),
            float(self._tree_seg_cum_search_dist_include_terrain),
            int(self._num_workers),
        )

        instance_ids = make_labels_consecutive(instance_ids, ignore_id=-1, inplace=True)

        if self._invalid_tree_id != -1:
            if self._invalid_tree_id == 0:
                instance_ids[instance_ids != -1] += 1
            instance_ids[instance_ids == -1] = self._invalid_tree_id

        full_instance_ids = np.full(len(xyz), fill_value=self._invalid_tree_id, dtype=np.int64)
        full_instance_ids = instance_ids[inverse_indices]

        return full_instance_ids

    def __call__(  # pylint: disable=too-many-locals
        self,
        xyz: FloatArray,
        intensities: Optional[FloatArray] = None,
        point_cloud_id: Optional[str] = None,
        crs: Optional[str] = None,
    ) -> Tuple[LongArray, FloatArray, FloatArray]:
        r"""
        Runs the tree instance segmentation for the given point cloud.

        Args:
            xyz: 3D coordinates of all points in the point cloud.
            intensities: Reflection intensities of all points in the point cloud. If set to :code:`None`, filtering
                steps that use intensity values are skipped.
            point_cloud_id: ID of the point cloud to be used in the file names of the visualization results. If set to
                :code:`None`, no visualizations are created.
            crs: EPSG code of the coordinate reference system of the input point cloud. The EPSG code is used to set the
                coordinate reference system when exporting intermediate data, such as a digital terrain model file.
                If set to :code:`None`, no coordinate reference system is set for the exported data.

        Returns:
            :Tuple of three arrays:
                - Tree instance labels for all points. For points not belonging to any tree, the label is set to
                  :code:`invalid_instance_id` (constructor parameter).
                - Stem positions of the detected trees (xy-coordinates of the stem center at breast height).
                - Stem diameters at breast height of the detected trees.

        Raises:
            ValueError: If :code:`intensities` is not :code:`None`.
            ValueError: If :code:`xyz` and :code:`intensities` have different lengths.

        Shape:
            - :code:`xyz`: :math:`(N, 3)`
            - :code:`intensities`: :math:`(N)`
            - Output: :math:`(N)`, :math:`(T)`, :math:`(T)`

            | where
            |
            | :math:`N = \text{ number of points}`
            | :math:`T = \text{ number of detected trees}`

        **Example**::

                from pointtree.instance_segmentation import TreeXAlgorithm
                from pointtorch import read

                point_cloud = read("path to point cloud file")
                algorithm = TreeXAlgorithm()
                xyz = point_cloud.xyz()
                intensities = point_cloud["intensity"].to_numpy()

                instance_ids, stem_positions, stem_diameters = algorithm(xyz, intensities)
        """

        self._random_generator = np.random.default_rng(seed=self._random_seed)

        if intensities is not None and len(xyz) != len(intensities):
            raise ValueError("xyz and intensities must have the same length.")

        with Profiler("Construction of digital terrain model", self._performance_tracker):
            with Profiler("Terrain classification", self._performance_tracker):
                self._logger.info("Detect terrain points...")
                terrain_classification = cloth_simulation_filtering(
                    xyz,
                    classification_threshold=self._csf_terrain_classification_threshold,
                    resolution=self._csf_resolution,
                    rigidness=self._csf_rigidness,
                    correct_steep_slope=self._csf_correct_steep_slope,
                    iterations=self._csf_iterations,
                )

            with Profiler("DTM rasterization", self._performance_tracker):
                self._logger.info("Construct rasterized digital terrain model...")
                dtm, dtm_offset = create_digital_terrain_model(
                    xyz[terrain_classification == 0],
                    grid_resolution=self._dtm_resolution,
                    k=self._dtm_k,
                    p=self._dtm_power,
                    voxel_size=self._dtm_voxel_size,
                    num_workers=self._num_workers,
                )
                del terrain_classification

                if self._visualization_folder is not None and point_cloud_id is not None:
                    self.export_dtm(dtm, dtm_offset, point_cloud_id, crs=crs)

        with Profiler("Computation of point heights above terrain", self._performance_tracker):
            self._logger.info("Compute point distances to terrain...")
            dists_to_dtm = distance_to_dtm(xyz, dtm, dtm_offset, self._dtm_resolution)

        with Profiler("Detection of tree stems", self._performance_tracker):
            self._logger.info("Detect stems...")
            stem_layer_filter = np.flatnonzero(
                np.logical_and(
                    dists_to_dtm >= self._stem_search_min_z,
                    dists_to_dtm < self._stem_search_max_z,
                )
            )
            stem_layer_xyz = xyz[stem_layer_filter]

            if self._visualization_folder is not None and point_cloud_id is not None:
                self.export_point_cloud(
                    stem_layer_xyz,
                    {"dist_to_dtm": dists_to_dtm[stem_layer_filter]},
                    "stem_layer",
                    point_cloud_id,
                    crs=crs,
                )

            stem_positions, stem_diameters, cluster_labels = self.detect_stems(
                stem_layer_xyz,
                dtm,
                dtm_offset,
                intensities=intensities[stem_layer_filter] if intensities is not None else None,
                point_cloud_id=point_cloud_id,
                crs=crs,
            )
            cluster_labels_full = np.full(len(xyz), fill_value=-1, dtype=np.int64)
            cluster_labels_full[stem_layer_filter] = cluster_labels
            del dtm
            del dtm_offset
            del stem_layer_filter
            del stem_layer_xyz
            del cluster_labels

        if len(stem_positions) == 0:
            return (
                np.full(len(xyz), fill_value=self._invalid_tree_id, dtype=xyz.dtype),
                stem_positions,
                stem_diameters,
            )

        with Profiler("Segmentation of entire trees", self._performance_tracker):
            self._logger.info("Segment tree crowns...")
            instance_ids = self.segment_crowns(
                xyz,
                dists_to_dtm,
                is_tree=dists_to_dtm >= self._csf_tree_classification_threshold,
                stem_positions=stem_positions,
                stem_diameters=stem_diameters,
                stem_labels=cluster_labels_full,
            )

        self._logger.info("Finished segmentation.")

        print(self.performance_metrics())

        return instance_ids, stem_positions, stem_diameters

    def export_dtm(
        self, dtm: FloatArray, dtm_offset: FloatArray, point_cloud_id: str, crs: Optional[str] = None
    ) -> None:
        r"""
        Exports the given digital terrain model as a GeoTIF file to :code:`visualization_folder` (constructor
        parameter).

        Args:
            dtm: Digital terrain model.
            dtm_offset: X- and y-coordinate of the top left corner of the DTM grid.
            point_cloud_id: ID of the point cloud to be used in the file name.
            crs: Coordinate reference system to be used. If set to :code:`None`, the output file is not georeferenced.

        Raises:
            ValueError: If :code:`visualization_folder` is :code:`None`.

        Shape:
            - :code:`dtm`: :math:`(H, W)`
            - :code:`dtm_offset`: :math:`(2)`

            | where
            |
            | :math:`H = \text{ extent of the DTM grid in y-direction}`
            | :math:`W = \text{ extent of the DTM grid in x-direction}`
        """

        if self._visualization_folder is None:
            raise ValueError("To create a DTM file, the visualization folder must not be None.")

        dtm_origin = dtm_offset - self._dtm_resolution / 2

        transform = from_origin(dtm_origin[0], dtm_origin[1], self._dtm_resolution, self._dtm_resolution)

        with rasterio.open(
            self._visualization_folder / f"{point_cloud_id}_dtm.tif",
            "w",
            driver="GTiff",
            height=dtm.shape[0],
            width=dtm.shape[1],
            count=1,  # single-band raster
            dtype=dtm.dtype,
            crs=crs,
            transform=transform,
        ) as tif_file:
            tif_file.write(dtm, 1)

    def export_point_cloud(
        self,
        xyz: FloatArray,
        attributes: Dict[str, npt.NDArray],
        step_name: str,
        point_cloud_id: str,
        crs: Optional[str] = None,
    ) -> None:
        r"""
        Exports a point cloud representing intermediate results as LAZ file to :code:`visualization_folder` (constructor
        parameter).

        Args:
            xyz: Point coordinates of the points.
            attributes: Dictionary with additional per-point attributes. The keys of the dictionary are the attribute
                names and the corresponding values must be numpy arrays of the same length as :code:`xyz`.
            step_name: Name of the processing step to be included in the file name.
            point_cloud_id: ID of the point cloud to be used in the file name.
            crs: Coordinate reference system to be used. If set to :code:`None`, the output file is not georeferenced.

        Raises:
            ValueError: If :code:`visualization_folder` is :code:`None`.
            ValueError: If the values in :code:`attributes` have a different length than :code:`xyz`.

        Shape:
            - :code:`xyz`: :math:`(N, 3)`
            - :code:`attributes`: each value must have shape :math:`(N)`

            | where
            |
            | :math:`N = \text{ number of points}`
        """

        if self._visualization_folder is None:
            raise ValueError("To export intermediate results, the visualization folder must not be None.")

        point_cloud = PointCloud(xyz, columns=["x", "y", "z"])
        for key, value in attributes.items():
            if len(xyz) != len(value):
                raise ValueError("xyz and all attributes must have the same length.")
            point_cloud[key] = value

        point_cloud.crs = crs
        point_cloud.to(self._visualization_folder / f"{point_cloud_id}_{step_name}.laz")
