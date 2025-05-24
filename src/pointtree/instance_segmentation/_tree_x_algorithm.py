"""Region-growing-based tree instance segmentation algorithm."""  # pylint: disable=too-many-lines

__all__ = ["TreeXAlgorithm"]

import itertools
import multiprocessing
from pathlib import Path
import sys
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
    collect_inputs_trunk_layers_fitting as collect_inputs_trunk_layers_fitting_cpp,
    collect_inputs_trunk_layers_refined_fitting as collect_inputs_trunk_layers_refined_fitting_cpp,
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
    Revised version of the tree instance segmentation algorithm originally introduced in `Tockner, Andreas, et al. \
    "Automatic Tree Crown Segmentation Using Dense Forest Point Clouds from Personal Laser Scanning (PLS)." \
    International Journal of Applied Earth Observation and Geoinformation 114 (2022): 103025. \
    <https://doi.org/10.1016/j.jag.2022.103025>`__ and `Gollob, Christoph, Tim Ritter, and Arne Nothdurft. "Forest
    Inventory with Long Range and High-Speed Personal Laser Scanning (PLS) and Simultaneous Localization and Mapping
    (SLAM) technology." Remote Sensing 12.9 (2020): 1509. <https://doi.org/10.3390/rs12091509>`__.

    Args:
        invalid_tree_id: ID that is assigned to points that do not belong to any tree instance. Must either be zero or
            a negative number. Defaults to -1.
        num_workers (int, optional): Number of workers to use for parallel processing. If :code:`workers` is set to -1,
            all CPU threads are used. Defaults to :code:`-1`.
        visualization_folder (str | pathlib.Path, optional): Path of a directory in which to store visualizations of
            intermediate results of the algorithm. Defaults to :code:`None`, which means that no visualizations are
            created.
        random_seed (int, optional): Random seed to for reproducibility of random processes. Defaults to 0.

    The  individual steps of the algorithm and their parameters are described in the following:

    .. rubric:: 1. Terrain Filtering Using the CSF Algorithm
    In the first step, the algorithm detects terrain points using the Cloth Simulation Filtering (CSF) algorithm \
    proposed in `Zhang, Wuming, et al. "An Easy-to-Use Airborne LiDAR Data Filtering Method Based on Cloth \
    Simulation." Remote Sensing 8.6 (2016): 501 <https://doi.org/10.3390/rs8060501>`__.

    Parameters:
        csf_terrain_classification_threshold (float, optional): Maximum height above the cloth a point can have in order
            to be classified as terrain point (in meters). All points whose distance to the cloth is equal or below this
            threshold are classified as terrain points. Defaults to 0.5 m.
        csf_tree_classification_threshold (float, optional): Minimum height above the cloth a point must have in order
            to be classified as tree point (in meters). All points whose distance to the cloth is equal or larger than
            this threshold are classified as tree points. Defaults to 0.5 m.
        csf_correct_steep_slope (bool, optional): Whether the cloth should be corrected for steep slopes in a
            post-pressing step. Defaults to `False`.
        csf_iterations (int, optional): Maximum number of iterations. Defaults to 500.
        csf_resolution (float, optional): Resolution of the cloth grid (in meters). Defaults to 0.5 m.
        csf_rigidness (int, optional): Rigidness of the cloth (the three levels 1, 2, and 3 are available, where 1 is
            the lowest and 3 the highest rigidness). Defaults to 2.

    .. rubric:: 2. Construction of a Digital Terrain Model
    In the next step, a rasterized digital terrain model (DTM) is constructed from the terrain points identified in the
    previous step. For this purpose, a grid of regularly arranged DTM points is created and the height of the :math:`k`
    closest terrain points is interpolated to obtain the height of each DTM point on the grid. In the interpolation,
    terrain points :math:`p_t` are weighted with a factor proportional to a power of :math:`p` of their inverse distance
    to the corresponding DTM point :math:`p_{dtm}`, i.e., :math:`\frac{1}{||(p_{dtm} - p)||^p}`. Before constructing the
    DTM, the terrain points are downsampled using voxel-based subsampling.

    Parameters:
        dtm_k (int, optional): Number of terrain points between which interpolation is performed to obtain the terrain
            height of a DTM point. Defaults to 400.
        dtm_p (float, optional): Power :math:`p` for inverse-distance weighting in the interpolation of terrain points.
            Defaults to 1.
        dtm_resolution (float, optional): Resolution of the DTM grid (in meters). Defaults to 0.25 m.
        dtm_voxel_size (float, optional): Voxel size with which the terrain points are downsampled before the DTM is
            created (in meters). Defaults to 0.05 m.

    .. rubric:: 3. Identification of Trunk Clusters
    The aim of this step is to identify clusters of points that may represent individual tree trunks, i.e., each
    trunk should be represented by a separate cluster. For this purpose, the point cloud is normalized by subtracting
    the corresponding DTM height from the height of each point. A horizontal slice is then extracted from the
    normalized point cloud, which contains all points within a certain height range above the terrain (the height range
    is defined by :code:`trunk_search_min_z` and :code:`trunk_search_max_z`). This layer should
    be chosen so that it contains all tree trunks and as few other objects as possible. The points within this slice are
    downsampled using voxel-based subsampling and then clustered in 2D using the DBSCAN algorithm proposed in
    `Ester, Martin, et al. "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise." \
    KDD. Vol. 96. No. 34, pp. 226-231. 1996. <https://dl.acm.org/doi/10.5555/3001460.3001507>`__ 

    Since, in some cases, several trunks that are located close to each other are assigned into a single cluster when
    applying the DBSCAN algorithm in 2D, an additional DBSCAN clustering step is done in 3D.

    After the clustering, points classified as noise are removed and the following filtering rules are applied to the
    clusters to filter out false positive clusters:

    1. Clusters with less than :code:`trunk_search_min_cluster_points` points are discarded.
    2. Clusters whose extent in the z-direction (i.e., the height difference between the highest and the lowest point in
       the cluster) is less than :code:`trunk_search_min_cluster_height` are discarded.
    3. The 80% quantile of the reflection intensities of the points in a cluster is calculated. Clusters that
       have an 80% quantile of intensities equal to or smaller than :code:`trunk_search_min_cluster_intensity` are
       discarded.
    4. From the remaining clusters :code:`trunk_search_circle_fitting_num_layers` horizontal and potentially
       overlapping layers are extracted, each with a height of :code:`trunk_search_circle_fitting_layer_height` and an
       overlap of :code:`trunk_search_circle_fitting_layer_overlap`. A circle or ellipse (if the circle fitting does not
       converge) is fitted to the points from each layer. Then, for all possible combinations of
       :code:`trunk_search_circle_fitting_std_num_layers` layers, the standard deviation of the fitted
       circle or ellipse diameters is computed. If for any of the combinations the standard deviation is smaller than or
       equal to :code:`trunk_search_circle_fitting_max_std_diameter`, the cluster is kept, otherwise it is discarded. A
       multi-stage process is used for the circle or ellipse fitting: First, preliminary circles or ellipses are fitted
       to the layers extracted from point clusters, which do not have the full point density due to the initial
       downsampling. Second, more exact circles or ellipses are fitted. To select the points used for the circle or
       ellipse fitting in this second step, clipping regions are determined based on the preliminary circles or
       ellipses. For this purpose, a buffer area is created around the outline of the respective circle or ellipse. This
       buffer has a width of :code:`trunk_search_circle_fitting_small_buffer_width` if the preliminary trunk diameter is
       less than or equal to :code:`trunk_search_circle_fitting_switch_buffer_threshold` and otherwise
       :code:`trunk_search_circle_fitting_large_buffer_width. All points from the full-resoultion point cloud that lie
       within the buffer area are determined and the fitting of the circles or ellipses is repeated using these points.
       If at least :code:`trunk_search_circle_fitting_std_num_layers` circles are found for a cluster, the filtering is
       done based on the fitted circles and the ellipses are not used for the filtering. Otherwise, the clusters are
       filtered based on the fitted ellipses.

    Parameters:
        trunk_search_min_z (float, optional): Height above the terrain at which the horizontal slice begins that is used
            for the DBSCAN clustering (in meters). Defaults to 1 m.
        trunk_search_max_z (float, optional): Height above the terrain at which the horizontal slice ends that is used
            for the DBSCAN clustering (in meters). Defaults to 3 m.
        trunk_search_voxel_size (float, optional): Voxel size with which the points from the horizontal slice are
            downsampled before performing the DBSCAN clustering (in meters). Defaults to 0.015 m.
        trunk_search_dbscan_2d_eps (float, optional): Parameter :math:`\epsilon` of the DBSCAN algorithm for the initial
            clustering in 2D. The parameter defined the radius of the circular neighborhood that is used to determine
            the number of neighbors for a given point (in meters). Defaults to 0.025 m.
        trunk_search_dbscan_2d_min_points (int, optional): Parameter :math:`MinPnts` of the DBSCAN algorithm for the
            initial clustering in 2D. The parameter defines the number of neighbors a given point must have in order to
            be considered as a core point. All neighbors of a core point are added to the clusters and then checked
            whether they are core points themselves. Defaults to 90.
        trunk_search_switch_clustering_3d_params_treshold (float, optional): Threshold on the horizontal extent of trunk
            clusters at which the parameters for the DBSCAN clustering in 3D are switched. Defaults to
            0.22 m<sup>2</sup>.
        trunk_search_dbscan_3d_eps_large (float, optional): Parameter :math:`\epsilon` of the DBSCAN algorithm that is
            used for the clustering in 3D if the horizontal extent of a cluster obtained from the initial clustering in
            2D is larger than or equal to :code:`trunk_search_switch_clustering_3d_params_treshold`. Defaults to 0.02 m.
        trunk_search_dbscan_3d_min_points_large (int, optional): Parameter :math:`MinPnts` of the DBSCAN algorithm that
            is used for the clustering in 3D if the horizontal extent of a cluster obtained from the initial clustering
            in 2D is larger than or equal to :code:`trunk_search_switch_clustering_3d_params_treshold`. Defaults to 20.
        trunk_search_min_cluster_points (int, optional): Minimum number of points a cluster must contain in order not to
            be discarded. Defaults to 300.
        trunk_search_min_cluster_height (float, optional): Minimum extent in the z-direction (i.e., the height
            difference between the highest and the lowest point in the cluster) a cluster must have in order not to be
            discarded (in meters). Defaults to 1.3 m.
        trunk_search_min_cluster_intensity (float, optional): Threshold for filtering of clusters based on reflection
            intensity values. Clusters are discarded if the 80 % percentile of the reflection intensities of the points
            in the cluster is below the given threshold. Defaults to 6000. If no reflection intensity values are input
            to the algorithm, the intensity-based filtering is skipped.
        trunk_search_min_explained_variance (float, optional): Minimum percentage of variance that the first principal
            component of an instance must explain in order to not be discarded (must be a value between zero and one).
            Defaults to 0.5.
        trunk_search_max_inclination (float, optional): Maximum inclination angle to the z-axis that the first
            principal component of an instance can have before being discarded (in degree). Defaults to 45 degree.
        trunk_search_circle_fitting_method (str, optional): Circle fitting method to use: :code:`"m-estimator"` |
            :code:`"ransac"`.
        trunk_search_circle_fitting_num_layers (int, optional): Number of horizontal layers used for the circle /
            ellipse fitting. Depending on the settings for :code:`trunk_search_circle_fitting_layer_height` and
            :code:`trunk_search_circle_fitting_layer_overlap`, this parameter controls which height range of the trunk
            clusters is considered for circle / ellipse fitting. Defaults to 14.
        trunk_search_circle_fitting_layer_height (float, optional): Height of the horizontal layers used for circle /
            ellipse fitting (in meters). Defaults to 0.15 m.
        trunk_search_circle_fitting_layer_overlap (float, optional): Overlap between adjacent horizontal layers used for
            circle / ellipse fitting (in meters). Defaults to 0.025 m.
        trunk_search_circle_fitting_min_points (int, optional): Minimum number of points that a horizontal layer must
            contain in order to perform circle / ellipse fitting on it. Defaults to 50.
        trunk_search_circle_fitting_min_trunk_diameter (float, optional): Minimum circle diameter for the circle fitting
            procedure. Defaults to 0.02 m.
        trunk_search_circle_fitting_max_trunk_diameter (float, optional): Maximum circle diameter for the circle fitting
            procedure. Defaults to 1 m.
        trunk_search_circle_fitting_min_completeness_idx (float, optional): Minimum circumferential completeness index
            that circles must achieve in the circle fitting procedure. If set to :code:`None`, circles are not filtered
            based on their circumferential completeness index. Defaults to :code:`None`.
        trunk_search_circle_fitting_small_buffer_width (float, optional): Width of the buffer area for constructing the
            clipping area for the circle or ellipse fitting on the full-resolution point cloud if the diameter of the
            preliminary circles or ellipses is less than or equal to
            :code:`trunk_search_circle_fitting_switch_buffer_threshold` (in meters). Defaults to 0.06 m.
        trunk_search_circle_fitting_large_buffer_width (float, optional): Width of the buffer area for constructing the
            clipping area for the circle or ellipse fitting on the full-resolution point cloud if the diameter of the
            preliminary circles or ellipses is greater than :code:`trunk_search_circle_fitting_switch_buffer_threshold`
            (in meters). Defaults to 0.09 m.
        trunk_search_circle_fitting_switch_buffer_threshold (float, optional): Threshold for the diameter of the
            preliminary circles or ellipses that controls when to switch between
            :code:`trunk_search_circle_fitting_small_buffer_width` and
            :code:`trunk_search_circle_fitting_large_buffer_width` (in meters). Defaults to 0.3 m.
        trunk_search_ellipse_filter_threshold (float, optional): In the final ellipse fitting, ellipses are only kept if
            the ratio of the radius along the semi-minor axis to the radius along the semi-major axis is greater than or
            equal to this threshold. Defaults to 0.6.
        trunk_search_circle_fitting_max_std_diameter (float, optional): Threshold for filtering the trunk clusters based
            on the standard deviation of the diameters of the fitted circles / ellipses. If there is at
            least one combination of :code:`trunk_search_circle_fitting_std_num_layers` layers for which the standard
            deviation of the diameters of the fitted circles / ellipses is below or equal to this threshold, the cluster
            is kept, otherwise it is discarded.
        trunk_search_circle_fitting_std_num_layers (int, optional): Number of horizontal layers to consider in each
            sample for calculating the standard deviation of the diameters of the fitted circles / ellipses. Defaults to
            6.
        trunk_search_gam_buffer_width: If the refined circle fitting is deactivated, all points in a buffer area around
            the outline of the initial circle / ellipse are cut out for the GAM fitting. This parameter defines the#
            width of the buffer area. Defaults to 0.03 m.
        trunk_search_gam_max_radius_diff: If the difference between the predicted radii of consecutive angles is greater
            than this parameter, the fitted GAM is considered invalid, and the diameter of the fitted circle / ellipse
            is used instead.

    .. rubric:: 4. Computation of Tree Positions and Trunk Diameters
    In this step, the trunk clusters obtained in the previous step are used to compute the trunk centers and trunk
    diameters at breast height. For this purpose, the circles and ellipses fitted in the previous step are used. For
    each trunk, the combination of those six circles or six ellipses (if not enough circles were detected) is selected
    whose radii have the lowest standard deviation. To estimate the trunk position at breast height, a linear model is
    fitted that predicts the center position of the selected circles or ellipses from the layer height. The prediction
    of the fitted model for a height of 1.3 m is used as an estimate of the trunk position at breast height. To estimate
    the trunk diameter at breast height, the trunk radius for each of the six selected layers is estimated by fitting a
    generalized additive model (GAM) to the points from each layer (only the points within the clipping area created in
    the previous step are used). Before fitting the GAM, the input points are centered around the circle / ellipse
    center and converted into polar coordinates. The GAM is then used to predict the radius part of the polar
    coordinates from the angle part. Using the fitted GAM, the bounding polygon of the trunk is predicted and the trunk
    radius is estimated based on the area of this bounding polygon. Finally, a linear model is fitted that predicts the
    trunk radius from the layer height. The prediction of the fitted model for a height of 1.3 m is used as an estimate
    of the trunk radius at breast height.

    .. rubric:: 5. Tree Segmentation Using Region Growing
    In this step, the points are downsampled using voxel-based subsampling and subsequently a point-wise segmentation of
    the individual trees is generated using a region growing procedure. In the
    first step, the region growing procedure selects an initial set of seed points for each tree. These should be points
    that are very likely to belong to the corresponding tree. In an iterative process, the sets of points assigned to
    each tree are then expanded. In each iteration, the neighboring points of each seed point within a certain search
    radius are determined. Neighboring points that are not yet assigned to any tree are added to the same tree as the
    seed point and become seed points in the next iteration. The region growing continues until there are no more seed
    points to be processed or the maximum number of iterations is reached.

    To select the initial seed points for a given tree, the following approach is used: A cylinder with a height of
    :code:`crown_seg_seed_layer_height` and a diameter of :code:`crown_seg_seed_diameter_factor * d` is
    considered, where :code:`d` is the tree's trunk diameter at breast height, which has been computed in the previous
    step. The cylinder's center is positioned at the trunk center at breast height, which also has been computed in the
    previous step. All points within the cylinder are selected as seed points.

    The search radius for the iterative region growing procedure is set as follows: First, the search radius is set to
    the voxel size used for voxel-based subsampling, which is performed before region growing. The search radius is
    increased by the voxel size if one of the following conditions is fulfilled at the end of a region growing
    iteration:

    1. The ratio between the number of points newly assigned to trees in the iteration and the number of remaining,
       unassigned points is below :code:`crown_seg_min_total_assignment_ratio`.
    2. The ratio between the number of trees to which new points have been assigned in the iteration and the total
       number of trees is below :code:`crown_seg_min_tree_assignment_ratio`.

    The search radius is increased up to a maximum radius of :code:`crown_seg_max_search_radius`. If the search
    radius has not been increased for :code:`crown_seg_decrease_search_radius_after_num_iter`, it is reduced by the
    voxel size.

    To promote upward growth, the z-coordinates of the points are divided by :code:`crown_seg_z_scale` before the
    region growing.

    Since the terrain filtering in the first step of the algorithm may be inaccurate and some tree points may be falsely
    classified as terrain points, both terrain and non-terrain points are considered by the region growing procedure.
    However, to prevent large portions of terrain points from being included in tree instances, terrain points are only
    assigned to if their cumulative search distance from the initial seed point is below the threshold defined by
    :code:`crown_seg_cum_search_dist_include_ground`. The cumulative search distance is defined as the total
    distance traveled between consecutive points until reaching a terrain point.

    Parameters:
        crown_seg_voxel_size (float, optional): Voxel size with which the points are downsampled before
            the region growing (in meters). Defaults to 0.05 m.
        crown_seg_z_scale (float, optional): Factor by which to divide the z-coordinates of the points before
            the region growing. To promote upward growth, this factor should be larger than 1. Defaults to 2.
        crown_seg_seed_layer_height (float, optional): Height of the cylinders that are placed around the trunk
            centers at breast height for seed point selection (in meters). Defaults to 0.6 m.
        crown_seg_seed_dbh_factor (float, optional): Factor to multiply with the trunk diameter at breast height
            to obtain the diameter of the cylinder used for seed point selection. Defaults to 1.05.
        crown_seg_seed_min_diameter (float, optional): Minimum diameter of the cylinder used for seed point
            selection. Defaults to 0.05 m.
        crown_seg_min_total_assignment_ratio (float, optional): Threshold controlling when to increase the search
            radius. If the ratio between the number of points newly assigned to trees in an iteration and the number of
            remaining, unassigned points is below this threshold, the search radius is increased by
            :code:`crown_seg_voxel_size` up to a maximum search radius of :code:`crown_seg_max_search_radius`.
            Defaults to 0.002.
        crown_seg_min_tree_assignment_ratio (float, optional): Threshold controlling when to increase the search
            radius. If the ratio between the number of trees to which new points have been assigned in the iteration and
            the total number of trees is below this threshold, the search radius is increased by
            :code:`crown_seg_voxel_size` up to a maximum search radius of :code:`crown_seg_max_search_radius`.
            Defaults to 0.3.
        crown_seg_max_search_radius (float, optional): Maximum search radius (in meters). Defaults to 0.5 m.
        crown_seg_decrease_search_radius_after_num_iter (int, optional): Number of region growing iterations after
            which to decrease the search radius by :code:`crown_seg_voxel_size` if it has not been increased in
            these iterations. Defaults to 10.
        crown_seg_max_iterations (int, optional): Maximum number of region growing iterations. Defaults to 1000.
        crown_seg_cum_search_dist_include_terrain (float, optional): Maximum cumulative search distance between the
            initial seed point and a terrain point to include that terrain point in a tree instance (in meters).
            Defaults to 0.9 m.

    Raises:
        ValueError: If :code:`trunk_search_min_z` is smaller than :code:`csf_tree_classification_threshold` or if
            :code:`trunk_search_circle_fitting_layer_start` is smaller than :code:`trunk_search_min_z`.
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
        dtm_p: float = 1,
        dtm_resolution: float = 0.25,
        dtm_voxel_size: float = 0.05,
        # parameters for the identification of trunk clusters
        trunk_search_min_z: float = 1.0,
        trunk_search_max_z: float = 4.0,
        trunk_search_voxel_size: float = 0.015,
        trunk_search_dbscan_2d_eps: float = 0.025,
        trunk_search_dbscan_2d_min_points: int = 100,
        trunk_search_dbscan_3d_eps: float = 0.1,
        trunk_search_dbscan_3d_min_points: int = 15,
        trunk_search_min_cluster_points: Optional[int] = 300,
        trunk_search_min_cluster_height: Optional[float] = 1.5,
        trunk_search_min_cluster_intensity: Optional[float] = 6000,
        trunk_search_min_explained_variance: Optional[float] = 0.5,
        trunk_search_max_trunk_inclination: Optional[float] = 45,
        trunk_search_circle_fitting_method: Literal["m-estimator", "ransac"] = "ransac",
        trunk_search_circle_fitting_layer_start: float = 1.0,
        trunk_search_circle_fitting_num_layers: int = 15,
        trunk_search_circle_fitting_layer_height: float = 0.225,
        trunk_search_circle_fitting_layer_overlap: float = 0.025,
        trunk_search_circle_fitting_min_points: int = 15,
        trunk_search_circle_fitting_min_trunk_diameter: float = 0.02,
        trunk_search_circle_fitting_max_trunk_diameter: float = 1.0,
        trunk_search_circle_fitting_min_completeness_idx: Optional[float] = 0.3,
        trunk_search_circle_fitting_refined_circle_fitting: bool = False,
        trunk_search_circle_fitting_small_buffer_width: float = 0.06,
        trunk_search_circle_fitting_large_buffer_width: float = 0.09,
        trunk_search_circle_fitting_switch_buffer_threshold: float = 0.3,
        trunk_search_ellipse_filter_threshold: float = 0.6,
        trunk_search_circle_fitting_max_std_diameter: float = 0.02,
        trunk_search_circle_fitting_max_std_position: Optional[float] = None,
        trunk_search_circle_fitting_std_num_layers: int = 6,
        trunk_search_gam_buffer_width: float = 0.03,
        trunk_search_gam_max_radius_diff: Optional[float] = 0.3,
        # region growing parameters
        crown_seg_voxel_size: float = 0.05,
        crown_seg_z_scale: float = 2,
        crown_seg_seed_layer_height: float = 0.6,
        crown_seg_seed_diameter_factor: float = 1.05,
        crown_seg_seed_min_diameter: float = 0.05,
        crown_seg_min_total_assignment_ratio: float = 0.002,
        crown_seg_min_tree_assignment_ratio: float = 0.3,
        crown_seg_max_search_radius: float = 0.5,
        crown_seg_decrease_search_radius_after_num_iter: int = 10,
        crown_seg_max_iterations: int = 1000,
        crown_seg_cum_search_dist_include_terrain: float = 0.9,
    ):
        super().__init__()

        self._csf_terrain_classification_threshold = csf_terrain_classification_threshold
        self._csf_tree_classification_threshold = csf_tree_classification_threshold
        self._csf_correct_steep_slope = csf_correct_steep_slope
        self._csf_iterations = csf_iterations
        self._csf_resolution = csf_resolution
        self._csf_rigidness = csf_rigidness

        self._dtm_k = dtm_k
        self._dtm_p = dtm_p
        self._dtm_resolution = dtm_resolution
        self._dtm_voxel_size = dtm_voxel_size

        if invalid_tree_id > 0:
            raise ValueError("invalid_tree_id must either be zero or a negative number.")

        if trunk_search_min_z < csf_tree_classification_threshold:
            raise ValueError("csf_tree_classification_threshold must be smaller than trunk_search_min_z.")

        if trunk_search_circle_fitting_layer_start < trunk_search_min_z:
            raise ValueError("trunk_search_min_z must be smaller than trunk_search_circle_fitting_layer_start.")

        self._trunk_search_min_z = trunk_search_min_z
        self._trunk_search_max_z = trunk_search_max_z
        self._trunk_search_voxel_size = trunk_search_voxel_size
        self._trunk_search_dbscan_2d_eps = trunk_search_dbscan_2d_eps
        self._trunk_search_dbscan_2d_min_points = trunk_search_dbscan_2d_min_points
        self._trunk_search_dbscan_3d_eps = trunk_search_dbscan_3d_eps
        self._trunk_search_dbscan_3d_min_points = trunk_search_dbscan_3d_min_points

        self._trunk_search_min_cluster_points = trunk_search_min_cluster_points
        self._trunk_search_min_cluster_height = trunk_search_min_cluster_height
        self._trunk_search_min_cluster_intensity = trunk_search_min_cluster_intensity
        self._trunk_search_min_explained_variance = trunk_search_min_explained_variance
        self._trunk_search_max_trunk_inclination = trunk_search_max_trunk_inclination
        self._trunk_search_circle_fitting_method = trunk_search_circle_fitting_method
        self._trunk_search_circle_fitting_num_layers = trunk_search_circle_fitting_num_layers
        self._trunk_search_circle_fitting_layer_start = trunk_search_circle_fitting_layer_start
        self._trunk_search_circle_fitting_layer_height = trunk_search_circle_fitting_layer_height
        self._trunk_search_circle_fitting_layer_overlap = trunk_search_circle_fitting_layer_overlap
        self._trunk_search_circle_fitting_min_points = trunk_search_circle_fitting_min_points
        self._trunk_search_circle_fitting_min_trunk_diameter = trunk_search_circle_fitting_min_trunk_diameter
        self._trunk_search_circle_fitting_max_trunk_diameter = trunk_search_circle_fitting_max_trunk_diameter
        self._trunk_search_circle_fitting_min_completeness_idx = trunk_search_circle_fitting_min_completeness_idx
        self._trunk_search_circle_fitting_refined_circle_fitting = trunk_search_circle_fitting_refined_circle_fitting
        self._trunk_search_circle_fitting_small_buffer_width = trunk_search_circle_fitting_small_buffer_width
        self._trunk_search_circle_fitting_large_buffer_width = trunk_search_circle_fitting_large_buffer_width
        self._trunk_search_circle_fitting_switch_buffer_threshold = trunk_search_circle_fitting_switch_buffer_threshold
        self._trunk_search_ellipse_filter_threshold = trunk_search_ellipse_filter_threshold
        self._trunk_search_circle_fitting_max_std_diameter = trunk_search_circle_fitting_max_std_diameter
        if trunk_search_circle_fitting_max_std_position:
            self._trunk_search_circle_fitting_max_std_position = trunk_search_circle_fitting_max_std_position
        else:
            self._trunk_search_circle_fitting_max_std_position = np.inf
        self._trunk_search_circle_fitting_std_num_layers = trunk_search_circle_fitting_std_num_layers
        self._trunk_search_gam_buffer_width = trunk_search_gam_buffer_width
        self._trunk_search_gam_max_radius_diff = trunk_search_gam_max_radius_diff

        self._crown_seg_voxel_size = crown_seg_voxel_size
        self._crown_seg_z_scale = crown_seg_z_scale
        self._crown_seg_seed_diameter_factor = crown_seg_seed_diameter_factor
        self._crown_seg_seed_min_diameter = crown_seg_seed_min_diameter
        self._crown_seg_seed_layer_height = crown_seg_seed_layer_height
        self._crown_seg_min_total_assignment_ratio = crown_seg_min_total_assignment_ratio
        self._crown_seg_min_tree_assignment_ratio = crown_seg_min_tree_assignment_ratio
        self._crown_seg_max_search_radius = crown_seg_max_search_radius
        self._crown_seg_decrease_search_radius_after_num_iter = crown_seg_decrease_search_radius_after_num_iter
        self._crown_seg_max_iterations = crown_seg_max_iterations
        self._crown_seg_cum_search_dist_include_terrain = crown_seg_cum_search_dist_include_terrain

        self._invalid_tree_id = invalid_tree_id
        self._num_workers = num_workers if num_workers is not None else 1

        if visualization_folder is None or isinstance(visualization_folder, Path):
            self._visualization_folder = visualization_folder
        else:
            self._visualization_folder = Path(visualization_folder)
            self._visualization_folder.mkdir(exist_ok=True, parents=True)
        self._random_seed = random_seed
        self._random_generator = np.random.default_rng(seed=random_seed)

    def export_dtm(
        self, dtm: npt.NDArray, dtm_offset: npt.NDArray, point_cloud_id: str, crs: Optional[str] = None
    ) -> None:
        r"""
        Exports the given digital terrain model as a GeoTIF file to :code:`visualization_folder`.

        Args:
            dtm: Digital terrain model.
            dtm_offset: X- and y-coordinate of the top left corner of the DTM grid.
            point_cloud_id: ID of the point cloud to be used in the file name.
            crs: Coordinate reference system to be used. Defaults to :code:`None`, which means that the output file is
                not georeferenced.

        Returns:
            ValueError: If :code:`self._visualization_folder` is :code:`None`.

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
        xyz: npt.NDArray,
        attributes: Dict[str, npt.NDArray],
        step_name: str,
        point_cloud_id: str,
        crs: Optional[str] = None,
    ) -> None:
        r"""
        Exports a point cloud representing intermediate results as LAZ file to :code:`visualization_folder`.

        Args:
            xyz: Point coordinates of the points.
            attributes: Dictionary with additional per-point attributes. The keys of the dictionary are the attribute
                names and the corresponding values must be numpy arrays of the same length as :code:`xyz`.
            step_name: Name of the processing step to be included in the file name.
            point_cloud_id: ID of the point cloud to be used in the file name.
            crs: Coordinate reference system to be used. Defaults to :code:`None`, which means that the output file is
                not georeferenced.

        Raises:
            ValueError: If :code:`self._visualization_folder` is :code:`None` or if the values in :code:`attributes`
            have a different length than :code:`xyz`.

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
        point_cloud.to(
            self._visualization_folder / f"{point_cloud_id}_{step_name}.laz",
        )

    def find_trunks(  # pylint: disable=too-many-locals, too-many-statements, too-many-branches
        self,
        trunk_layer_xyz: npt.NDArray,
        dtm: npt.NDArray,
        dtm_offset: npt.NDArray,
        intensities: Optional[npt.NDArray] = None,
        point_cloud_id: Optional[str] = None,
        crs: Optional[str] = None,
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray[np.int64]]:
        r"""
        Identifies tree trunks in a 3D point cloud.

        Args:
            trunk_layer_xyz: Point coordinates of the points within the trunk layer.
            dtm: Digital terrain model.
            dtm_offset: X- and y-coordinate of the top left corner of the DTM grid.
            intensities: Reflection intensities of all points in the point cloud. If set to :code:`None`, filtering
                steps that use intensity values are skipped. Defaults to :code:`None`.
            point_cloud_id: ID of the point cloud to be used in the file names of the visualization results. Defaults to
                :code:`None`, which means that no visualizations are created.
            crs: EPSG code of the coordinate reference system of the input point cloud. The EPSG code is used to set the
                coordinate reference system when exporting intermediate data. Defaults to :code:`None`, which means that
                no coordinate reference system is set for the exported data.

        Returns:
            Tuple of three arrays. The first contains the x- and y-coordinates of the position of each trunk at breast
            height (1.3 m). The second contains the diameter of each trunk at breast height. The third contains the
            cluster label of each point, with points that do not belong to any cluster being assigned the label
            :code:`-1`.

        Shape:
            - :code:`trunk_layer_xyz`: :math:`(N, 3)`
            - :code:`dtm`: :math:`(H, W)`
            - :code:`dtm_offset`: :math:`(2)`
            - :code:`intensities`: :math:`(N)`
            - Output: Tuple of three arrays. The first has shape :math:`(T, 2)`, the second has shape :math:`(T)`, and
                the third has shape :math:`(N)`.

            | where
            |
            | :math:`N = \text{ number of points}`
            | :math:`T = \text{ number of trunks}`
            | :math:`H = \text{ extent of the DTM grid in y-direction}`
            | :math:`W = \text{ extent of the DTM grid in x-direction}`
        """

        trunk_layer_xyz_downsampled, selected_indices, inverse_indices = voxel_downsampling(
            trunk_layer_xyz, voxel_size=self._trunk_search_voxel_size
        )
        if intensities is not None:
            intensities = intensities[selected_indices]

        with Profiler("2D DBSCAN", self._performance_tracker):
            self._logger.info("Cluster trunk points in 2D...")
            dbscan = DBSCAN(
                eps=self._trunk_search_dbscan_2d_eps,
                min_samples=self._trunk_search_dbscan_2d_min_points,
                n_jobs=self._num_workers,
            )
            dbscan.fit(trunk_layer_xyz_downsampled[:, :2])
            cluster_labels = dbscan.labels_.astype(np.int64)
            unique_cluster_labels = np.unique(cluster_labels)
            unique_cluster_labels = unique_cluster_labels[unique_cluster_labels != -1]

            self._logger.info("Found %d trunk candidates.", len(unique_cluster_labels))

            if self._visualization_folder is not None and point_cloud_id is not None:
                point_cloud_attributes = {"dbscan_2d": (cluster_labels + 1).astype(np.uint32)}

        # In the paper, it is not explicitely mentioned that a filtering step based on point count and vertical exten
        # is done between the 2D and the 3D DBSCAN clustering. However, including such filtering does not change the
        # final result because clusters can only get smaller during 3D clustering and saves computational resources.
        with Profiler("Filtering of trunk clusters based on point count", self._performance_tracker):
            cluster_labels, unique_cluster_labels = filter_instances_min_points(
                cluster_labels, unique_cluster_labels, min_points=self._trunk_search_min_cluster_points, inplace=True
            )

            self._logger.info(
                "%d trunk candidates remaining after discarding clusters with too few points.",
                len(unique_cluster_labels),
            )

            if self._visualization_folder is not None and point_cloud_id is not None:
                point_cloud_attributes["filtering_point_count_2d"] = (cluster_labels + 1).astype(np.uint32)

        with Profiler("Filtering of trunk clusters based on vertical extent", self._performance_tracker):
            cluster_labels, unique_cluster_labels = filter_instances_vertical_extent(
                trunk_layer_xyz_downsampled,
                cluster_labels,
                unique_cluster_labels,
                min_vertical_extent=self._trunk_search_min_cluster_height,
                inplace=True,
            )

            self._logger.info(
                "%d trunk candidates remaining after discarding clusters with too small " + "vertical extent.",
                len(unique_cluster_labels),
            )

            if self._visualization_folder is not None and point_cloud_id is not None:
                point_cloud_attributes["filtering_vertical_extent_2d"] = (cluster_labels + 1).astype(np.uint32)

        with Profiler("3D clustering of trunk points", self._performance_tracker):
            self._logger.info("Cluster trunk points in 3D...")

            next_label = unique_cluster_labels.max() + 1 if len(unique_cluster_labels) > 0 else 0
            for label in unique_cluster_labels:
                cluster_indices = np.flatnonzero(cluster_labels == label)

                cluster_xyz = trunk_layer_xyz_downsampled[cluster_indices]

                dbscan = DBSCAN(
                    eps=self._trunk_search_dbscan_3d_eps,
                    min_samples=self._trunk_search_dbscan_3d_min_points,
                    n_jobs=self._num_workers,
                )

                dbscan.fit(cluster_xyz)
                new_labels = dbscan.labels_.astype(np.int64)
                new_labels[new_labels != -1] += next_label
                next_label = new_labels.max() + 1
                cluster_labels[cluster_indices] = new_labels

            cluster_labels, unique_cluster_labels = make_labels_consecutive(
                cluster_labels, ignore_id=-1, inplace=True, return_unique_labels=True
            )

            self._logger.info(
                "%d trunk candidates after clustering in 3D.",
                len(unique_cluster_labels),
            )

        if self._visualization_folder is not None and point_cloud_id is not None:
            point_cloud_attributes["dbscan_3d"] = (cluster_labels + 1).astype(np.uint32)

        with Profiler("Filtering of trunk clusters based on point count", self._performance_tracker):
            cluster_labels, unique_cluster_labels = filter_instances_min_points(
                cluster_labels, unique_cluster_labels, min_points=self._trunk_search_min_cluster_points, inplace=True
            )

            self._logger.info(
                "%d trunk candidates remaining after discarding clusters with too few points.",
                len(unique_cluster_labels),
            )

            if self._visualization_folder is not None and point_cloud_id is not None:
                point_cloud_attributes["filtered_point_count_3d"] = (cluster_labels + 1).astype(np.uint32)

        with Profiler("Filtering of trunk clusters based on vertical extent", self._performance_tracker):
            cluster_labels, unique_cluster_labels = filter_instances_vertical_extent(
                trunk_layer_xyz_downsampled,
                cluster_labels,
                unique_cluster_labels,
                min_vertical_extent=self._trunk_search_min_cluster_height,
                inplace=True,
            )

            if self._visualization_folder is not None and point_cloud_id is not None:
                point_cloud_attributes["filtered_vertical_extent_3d"] = (cluster_labels + 1).astype(np.uint32)

            self._logger.info(
                "%d trunk candidates remaining after discarding clusters with too small vertical extent.",
                len(unique_cluster_labels),
            )

        if intensities is not None:
            with Profiler("Filtering of trunk clusters based on intensity values", self._performance_tracker):
                cluster_labels, unique_cluster_labels = filter_instances_intensity(
                    intensities,
                    cluster_labels,
                    unique_cluster_labels,
                    min_intensity=self._trunk_search_min_cluster_intensity,
                    threshold_percentile=0.8,
                    inplace=True,
                )

                if self._visualization_folder is not None and point_cloud_id is not None:
                    point_cloud_attributes["filtered_intensity_3d"] = (cluster_labels + 1).astype(np.uint32)

                self._logger.info(
                    "%d trunk candidates remaining after discarding clusters with small intensity.",
                    len(unique_cluster_labels),
                )

        with Profiler("Filtering of trunk clusters based on PCA", self._performance_tracker):
            cluster_labels, unique_cluster_labels = filter_instances_pca(
                trunk_layer_xyz_downsampled,
                cluster_labels,
                unique_cluster_labels,
                min_explained_variance=self._trunk_search_min_explained_variance,
                max_inclination=self._trunk_search_max_trunk_inclination,
                inplace=True,
            )

            if self._visualization_folder is not None and point_cloud_id is not None:
                point_cloud_attributes["filtered_linearity_3d"] = (cluster_labels + 1).astype(np.uint32)

            self._logger.info(
                "%d trunk candidates remaining after discarding clusters with low linearity.",
                len(unique_cluster_labels),
            )

        (
            layer_circles,
            layer_ellipses,
            terrain_heights,
            layer_heights,
            trunk_layer_xy,
            batch_lengths_xy,
        ) = self.fit_circles_or_ellipses_to_trunks(
            trunk_layer_xyz_downsampled,
            cluster_labels,
            unique_cluster_labels,
            dtm,
            dtm_offset,
            point_cloud_id=point_cloud_id,
        )

        if self._trunk_search_circle_fitting_refined_circle_fitting:
            (
                layer_circles,
                layer_ellipses,
                trunk_layer_xy,
                batch_lengths_xy,
            ) = self.fit_refined_circles_and_ellipses_to_trunks(
                trunk_layer_xyz, layer_circles, layer_ellipses, terrain_heights, point_cloud_id=point_cloud_id
            )

        with Profiler(
            "Filtering of trunk clusters based on standard deviation of circle / ellipse diameters",
            self._performance_tracker,
        ):
            filter_mask, best_circle_combination, best_ellipse_combination = self.filter_instances_trunk_layer_std(
                layer_circles, layer_ellipses
            )
            layer_circles = layer_circles[filter_mask]
            layer_circles[best_circle_combination[:, 0] == -1] = -1
            layer_ellipses = layer_ellipses[filter_mask]
            unique_cluster_labels = unique_cluster_labels[filter_mask]

            self.rename_visualizations_after_filtering(filter_mask, point_cloud_id=point_cloud_id)

            filter_mask = np.repeat(filter_mask, self._trunk_search_circle_fitting_num_layers)
            trunk_layer_xy = trunk_layer_xy[np.repeat(filter_mask, batch_lengths_xy)]
            batch_lengths_xy = batch_lengths_xy[filter_mask]

            self._logger.info(
                "%d trunks remaining after discarding clusters with too high standard deviation.", len(layer_circles)
            )

        if self._visualization_folder is not None and point_cloud_id is not None:
            cluster_labels[~np.isin(cluster_labels, unique_cluster_labels, assume_unique=True)] = -1
            point_cloud_attributes["filtered_circle_fitting_3d"] = (cluster_labels + 1).astype(np.uint32)
            self.export_point_cloud(
                trunk_layer_xyz_downsampled,
                point_cloud_attributes,
                "trunk_clusters",
                point_cloud_id,
                crs=crs,
            )
            del point_cloud_attributes
        del unique_cluster_labels

        if not self._trunk_search_circle_fitting_refined_circle_fitting:
            if not trunk_layer_xyz_downsampled.flags.f_contiguous:
                trunk_layer_xyz_downsampled = trunk_layer_xyz_downsampled.copy(order="F")

            trunk_layer_xy, batch_lengths_xy = collect_inputs_trunk_layers_refined_fitting_cpp(
                trunk_layer_xyz_downsampled,
                layer_circles.reshape((-1, 3)).astype(trunk_layer_xyz_downsampled.dtype, order="F"),
                layer_ellipses.reshape((-1, 5)).astype(trunk_layer_xyz_downsampled.dtype, order="F"),
                terrain_heights,
                float(self._trunk_search_circle_fitting_layer_start),
                int(self._trunk_search_circle_fitting_num_layers),
                float(self._trunk_search_circle_fitting_layer_height),
                float(self._trunk_search_circle_fitting_layer_overlap),
                float(self._trunk_search_circle_fitting_switch_buffer_threshold),
                float(self._trunk_search_gam_buffer_width),
                float(self._trunk_search_gam_buffer_width),
                0,
                int(self._num_workers),
            )
        del trunk_layer_xyz_downsampled

        with Profiler("Computation of trunk positions", self._performance_tracker):
            self._logger.info("Compute trunk positions...")
            trunk_positions = self.compute_trunk_positions(
                layer_circles, layer_ellipses, layer_heights, best_circle_combination, best_ellipse_combination
            )

        with Profiler("Computation of trunk diameters", self._performance_tracker):
            self._logger.info("Compute trunk diameters...")
            trunk_diameters = self.compute_trunk_diameters(
                layer_circles,
                layer_ellipses,
                layer_heights,
                trunk_layer_xy,
                batch_lengths_xy,
                best_circle_combination,
                best_ellipse_combination,
                point_cloud_id=point_cloud_id,
            )

        cluster_labels = make_labels_consecutive(cluster_labels, ignore_id=-1)  # type: ignore[assignment]

        return trunk_positions, trunk_diameters, cluster_labels[inverse_indices]

    def fit_circles_or_ellipses_to_trunks(  # pylint: disable=too-many-locals, too-many-statements, too-many-branches
        self,
        trunk_layer_xyz: npt.NDArray,
        cluster_labels: npt.NDArray[np.int64],
        unique_cluster_labels: npt.NDArray[np.int64],
        dtm: npt.NDArray,
        dtm_offset: npt.NDArray,
        point_cloud_id: Optional[str] = None,
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        r"""
        Given a set of point clusters that may represent individual tree trunks, circles are fitted to multiple
        horinzontal layers of each cluster. If the circle fitting does not converge, an ellipse is fitted instead. If a
        horizontal layer contains less than :code:`self._trunk_search_circle_fitting_min_points` points, neither a
        circle nor an ellipse is fitted. To obtain the horizontal layers,
        :code:`self._trunk_search_circle_fitting_num_layers` horizontal layers with a height of
        :code:`self._trunk_search_circle_fitting_layer_height` are created starting at a height of
        :code:`self._trunk_search_circle_fitting_layer_start`. The layers have an overlap of
        :code:`self._trunk_search_circle_fitting_layer_overlap` to the previous layer. Hence, the last layer ends at
        :code:`self._trunk_search_circle_fitting_layer_start + self._trunk_search_circle_fitting_num_layers * \
        (self._trunk_search_circle_fitting_layer_height - self._trunk_search_circle_fitting_layer_overlap)`.

        Args:
            trunk_layer_xyz: Coordinates of the points from the trunk layer.
            cluster_labels: Indices indicating to which cluster each point belongs. Points not belonging to any cluster
                should be assigned the ID -1.
            unique_cluster_labels: Unique cluster labels, i.e., an array that should contain each cluster ID once. The
                cluster IDs are expected to start with zero and to be in a continuous range.
            dtm: Digital terrain model.
            dtm_offset: X- and y-coordinate of the top left corner of the DTM grid.
            point_cloud_id: ID of the point cloud to be used in the file names of the visualization results. Defaults to
                :code:`None`, which means that no visualizations are created.

        Returns:
            Tuple of six arrays. The first array contains the parameters of the circles that were fitted to the layers
            of each cluster. Each circle is represented by three values, namely the x- and y-coordinates of its center
            and its radius. If the circle fitting did not converge for a layer, all parameters are set to -1. The second
            array contains the parameters of the ellipses that were fitted to the layers of each cluster. Each ellipse
            is represented by five values, namely the x- and y-coordinates of its center, its radius along the
            semi-major and along the semi-minor axis, and the counterclockwise angle of rotation from the x-axis to
            the semi-major axis of the ellipse. If the ellipse fitting results in an ellipse whose axis ratio is smaller
            than :code:`self._trunk_search_ellipse_filter_threshold`, all parameters are set to -1. The third array
            contains the terrain height at the center of each cluster. The fourth contains the z-coordinate of the
            midpoint of each horizontal layer. The fifth array contains the x- and y-coordinates of the points in each
            horizontal layer of each cluster that were retrieved based on the cluster labels. Points belonging to the
            same layer of the same cluster are stored consecutively. The sixth array contains the number of points
            belonging to each horizontal layer of each cluster.

        Shape:
            - :code:`trunk_layer_xyz`: :math:`(N, 3)`
            - :code:`cluster_labels`: :math:`(N)`
            - :code:`unique_cluster_labels`: :math:`(T)`
            - Output: Tuple of six elements. The first has shape :math:`(T, L, 3)`, the second has shape
              :math:`(T, L, 5)`, the third has shape :math:`(T)`, the fourth has shape :math:`(L)`, the fifth has shape
              :math:`(N_{0,0} + ... + N_{T,L}, 2)`, and the sixth has shape :math:`(T \cdot L)`

            | where
            |
            | :math:`N = \text{ number of points in the trunk layer}`
            | :math:`T = \text{ number of trunk clusters}`
            | :math:`L = \text{ number of horinzontal layers to which circles / ellipses are fitted}`
            | :math:`N_{t,l} = \text{ number of points selected from the l-th layer of cluster } t`
        """

        with Profiler("Fitting of circles or ellipses to downsampled trunk candidates", self._performance_tracker):
            self._logger.info("Fitting circles / ellipses to downsampled trunk candidates...")

            num_layers = self._trunk_search_circle_fitting_num_layers

            layer_circles = np.full(
                (len(unique_cluster_labels), num_layers, 3),
                fill_value=-1,
                dtype=trunk_layer_xyz.dtype,
            )

            layer_ellipses = np.full(
                (len(unique_cluster_labels), num_layers, 5),
                fill_value=-1,
                dtype=trunk_layer_xyz.dtype,
            )

            if not trunk_layer_xyz.flags.f_contiguous:
                trunk_layer_xyz = trunk_layer_xyz.copy(order="F")

            if not dtm.flags.f_contiguous:
                dtm = dtm.copy(order="F")

            trunk_layer_xy, batch_lengths_xy, terrain_heights, layer_heights = collect_inputs_trunk_layers_fitting_cpp(
                trunk_layer_xyz,
                cluster_labels,
                unique_cluster_labels,
                dtm,
                dtm_offset,
                float(self._dtm_resolution),
                float(self._trunk_search_circle_fitting_layer_start),
                int(num_layers),
                float(self._trunk_search_circle_fitting_layer_height),
                float(self._trunk_search_circle_fitting_layer_overlap),
                int(self._trunk_search_circle_fitting_min_points),
                int(self._num_workers),
            )

            if len(unique_cluster_labels) == 0:
                return layer_circles, layer_ellipses, terrain_heights, layer_heights, trunk_layer_xy, batch_lengths_xy

            min_radius = self._trunk_search_circle_fitting_min_trunk_diameter / 2
            max_radius = self._trunk_search_circle_fitting_max_trunk_diameter / 2
            min_completeness_idx = self._trunk_search_circle_fitting_min_completeness_idx
            bandwidth = 0.01

            circle_detector: Union[MEstimator, Ransac]
            if self._trunk_search_circle_fitting_method == "m-estimator":
                circle_detector = MEstimator(
                    bandwidth=bandwidth,
                    break_min_change=1e-6,
                    min_step_size=1e-10,
                    max_iterations=300,
                    armijo_min_decrease_percentage=0.5,
                    armijo_attenuation_factor=0.25,
                )
                circle_detector.detect(
                    trunk_layer_xy,
                    batch_lengths=batch_lengths_xy,
                    n_start_x=3,
                    n_start_y=3,
                    min_start_radius=min_radius,
                    max_start_radius=max_radius,
                    n_start_radius=3,
                    num_workers=self._num_workers,
                )
            else:
                circle_detector = Ransac(bandwidth=bandwidth)
                circle_detector.detect(
                    trunk_layer_xy,
                    batch_lengths=batch_lengths_xy,
                    num_workers=self._num_workers,
                    seed=self._random_seed,
                )
            circle_detector.filter(
                max_circles=1,
                deduplication_precision=4,
                min_circumferential_completeness_idx=min_completeness_idx,
                circumferential_completeness_idx_max_dist=bandwidth,
                circumferential_completeness_idx_num_regions=int(365 / 5),
                non_maximum_suppression=True,
                num_workers=self._num_workers,
            )

            ellipses = fit_ellipse(trunk_layer_xy, batch_lengths_xy)

            visualization_tasks: List[Tuple[Any, ...]] = []

            batch_starts_xy = np.cumsum(np.concatenate((np.array([0], dtype=np.int64), batch_lengths_xy)))[:-1]
            batch_starts_circles = np.cumsum(
                np.concatenate((np.array([0], dtype=np.int64), circle_detector.batch_lengths_circles))
            )[:-1]
            for label_idx, label in enumerate(unique_cluster_labels):
                for layer in range(self._trunk_search_circle_fitting_num_layers):
                    flat_idx = label_idx * num_layers + layer
                    batch_start_idx_xy = batch_starts_xy[flat_idx]
                    batch_end_idx_xy = batch_start_idx_xy + batch_lengths_xy[flat_idx]
                    circle_idx = batch_starts_circles[flat_idx]
                    if batch_lengths_xy[flat_idx] < self._trunk_search_circle_fitting_min_points:
                        self._logger.info(
                            "Layer %d of trunk cluster %d contains too few points to fit a circle or an ellipse.",
                            layer,
                            label,
                        )
                        continue

                    has_circle = circle_detector.batch_lengths_circles[flat_idx] > 0

                    has_ellipse = ellipses[flat_idx, 2] != -1

                    if has_ellipse:
                        # filter out ellipses if radius is outside the accepted range
                        radius_major, radius_minor = ellipses[flat_idx, 2:4]

                        if radius_minor / radius_major < self._trunk_search_ellipse_filter_threshold:
                            has_ellipse = False

                    if not has_circle and not has_ellipse:
                        self._logger.info(
                            "Neither a circle nor an ellipse was found for layer %d of trunk cluster %d.",
                            layer,
                            label,
                        )
                        continue

                    if has_ellipse:
                        ellipse = ellipses[flat_idx]

                        center_x, center_y, radius_major, radius_minor, theta = ellipse
                        layer_ellipses[label_idx, layer] = [
                            center_x,
                            center_y,
                            radius_major,
                            radius_minor,
                            theta,
                        ]

                        if self._visualization_folder is not None and point_cloud_id is not None:
                            visualization_path = (
                                self._visualization_folder
                                / point_cloud_id
                                / f"preliminary_ellipse_trunk_{label}_layer_{layer}.png"
                            )
                            visualization_tasks.append(
                                (trunk_layer_xy[batch_start_idx_xy:batch_end_idx_xy], visualization_path, None, ellipse)
                            )
                    if has_circle:
                        layer_circles[label_idx, layer, :3] = circle_detector.circles[circle_idx]
                        if self._visualization_folder is not None and point_cloud_id is not None:
                            visualization_path = (
                                self._visualization_folder
                                / point_cloud_id
                                / f"preliminary_circle_trunk_{label}_layer_{layer}.png"
                            )
                            visualization_tasks.append(
                                (
                                    trunk_layer_xy[batch_start_idx_xy:batch_end_idx_xy],
                                    visualization_path,
                                    circle_detector.circles[circle_idx],
                                )
                            )

        if len(visualization_tasks) > 0:
            with Profiler(
                "Visualization of circles and ellipses fitted to downsampled trunk candidates",
                self._performance_tracker,
            ):
                self._logger.info("Visualize circles / ellipses fitted to downsampled trunk candidates...")
                num_workers = self._num_workers if self._num_workers > 0 else multiprocessing.cpu_count()
                with multiprocessing.Pool(processes=num_workers) as pool:
                    pool.starmap(plot_fitted_shape, visualization_tasks)

        return layer_circles, layer_ellipses, terrain_heights, layer_heights, trunk_layer_xy, batch_lengths_xy

    def fit_refined_circles_and_ellipses_to_trunks(  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        self,
        trunk_layer_xyz: npt.NDArray,
        preliminary_layer_circles: npt.NDArray,
        preliminary_layer_ellipses: npt.NDArray,
        terrain_heights: npt.NDArray,
        point_cloud_id: Optional[str] = None,
    ) -> Tuple[
        npt.NDArray,
        npt.NDArray,
        npt.NDArray,
        npt.NDArray[np.int64],
    ]:
        r"""
        Given a set of point clusters that may represent individual tree trunks, circles are fitted to multiple
        horinzontal layers of each cluster. To obtain the horizontal layers,
        :code:`self._trunk_search_circle_fitting_num_layers` horizontal layers with a height of
        :code:`self._trunk_search_circle_fitting_layer_height` are created starting at a height of
        :code:`self._trunk_search_circle_fitting_layer_start`. The layers have an overlap of
        :code:`self._trunk_search_circle_fitting_layer_overlap` to the previous layer. Hence, the last layer ends at
        :code:`self._trunk_search_circle_fitting_layer_start + self._trunk_search_circle_fitting_num_layers * \
        (self._trunk_search_circle_fitting_layer_height - self._trunk_search_circle_fitting_layer_overlap)`.

        The points used for the circle and ellipse fitting in each layer are selected based on the results of a
        preliminary circle or ellipse fitting. For this purpose, a buffer area is created around the outline of the
        respective circle or ellipse. This buffer has a width of :code:`trunk_search_circle_fitting_small_buffer_width`
        if the preliminary circle or ellipse diameter is less than or equal to
        :code:`trunk_search_circle_fitting_switch_buffer_threshold` and otherwise
        :code:`trunk_search_circle_fitting_large_buffer_width. All points from the respective layer that lie
        within the buffer area are determined and the fitting of the circles or ellipses is done using only these
        points.

        In the ellipse fitting, ellipses are only kept if the ratio of the radius along the semi-minor axis to the
        radius along the semi-major axis is greater than or equal to `self._trunk_search_ellipse_filter_threshold`.

        Args:
            trunk_layer_xyz: Coordinates of the points from the trunk layer.
            preliminary_layer_circles: Parameters of the preliminary circles. Each circle must be represented by three
                values, namely the x- and y-coordinates of its center and its radius. If the preliminary circle fitting
                was unsucessfull for the respective layer, all values must be set to -1.
            preliminary_layer_ellipses: Parameters of the preliminary ellipses.  Each ellipse must represented by five
                values, namely the x- and y-coordinates of its center, its radius along the semi-major and along the
                semi-minor axis, and the counterclockwise angle of rotation from the x-axis to the semi-major axis of
                the ellipse. If the preliminary circle fitting was unsucessfull for the respective layer, all values
                must be set to -1.
            terrain_heights: Terrain height at the center of each cluster.
            point_cloud_id: ID of the point cloud to be used in the file names of the visualization results. Defaults to
                :code:`None`, which means that no visualizations are created.

        Returns:
            Tuple of four arrays. The first array contains the parameters of the circles that were fitted to the layers
            of each cluster. Each circle is represented by three values, namely the x- and y-coordinates of its center
            and its radius. If the circle fitting did not converge for a layer, all parameters are set to -1. The second
            array contains the parameters of the ellipses that were fitted to the layers of each cluster. Each ellipse
            is represented by five values, namely the x- and y-coordinates of its center, its radius along the
            semi-major and along the semi-minor axis, and the counterclockwise angle of rotation from the x-axis to
            the semi-major axis of the ellipse. If the ellipse fitting results in an ellipse whose axis ratio is smaller
            than :code:`self._trunk_search_ellipse_filter_threshold`, all parameters are set to -1. The third array
            contains the z-coordinate of the midpoint of each horizontal layer. The fourth array contains the
            x- and y-coordinates of the points in each horizontal layer of each cluster that were selected for the
            circle and ellipse fitting in that layer based on the preliminary circles or ellipses. Points belonging to
            the same layer of the same cluster are stored consecutively. The fifth array contains the number of points
            belonging to each horizontal layer of each cluster.

        Shape:
            - :code:`trunk_layer_xyz`: :math:`(N, 3)`
            - :code:`preliminary_layer_circles`: :math:`(T, L, 3)`
            - :code:`preliminary_layer_ellipses`: :math:`(T, L, 5)`
            - :code:`terrain_heights`: :math:`(T)` 
            - Output: Tuple of four arrays. The first has shape :math:`(T, L, 3)`, the second has shape
              :math:`(T, L, 5)`, the third has shape :math:`(L)`, and the fourth has shape
              :math:`(N_{0,0} + ... + N_{T,L}, 2)`.

            | where
            |
            | :math:`N = \text{ number of points in the trunk layer}`
            | :math:`T = \text{ number of trunk clusters}`
            | :math:`L = \text{ number of horinzontal layers to which circles and ellipses are fitted}`
            | :math:`N_{t,l} = \text{ number of points selected from the l-th layer of cluster } t`
        """

        with Profiler("Fitting of circles and ellipses to full-resolution trunk candidates", self._performance_tracker):
            self._logger.info("Fitting circles / ellipses to full-resolution trunk candidates...")

            num_instances = len(preliminary_layer_circles)
            num_layers = self._trunk_search_circle_fitting_num_layers

            layer_circles = np.full(
                (num_instances, self._trunk_search_circle_fitting_num_layers, 3),
                fill_value=-1,
                dtype=trunk_layer_xyz.dtype,
            )
            layer_ellipses = np.full(
                (num_instances, self._trunk_search_circle_fitting_num_layers, 5),
                fill_value=-1,
                dtype=trunk_layer_xyz.dtype,
            )

            if not trunk_layer_xyz.flags.f_contiguous:
                trunk_layer_xyz = trunk_layer_xyz.copy(order="F")
            preliminary_layer_circles = preliminary_layer_circles.reshape((-1, 3)).astype(
                trunk_layer_xyz.dtype, order="F"
            )
            preliminary_layer_ellipses = preliminary_layer_ellipses.reshape((-1, 5)).astype(
                trunk_layer_xyz.dtype, order="F"
            )

            trunk_layer_xy, batch_lengths_xy = collect_inputs_trunk_layers_refined_fitting_cpp(
                trunk_layer_xyz,
                preliminary_layer_circles,
                preliminary_layer_ellipses,
                terrain_heights,
                float(self._trunk_search_circle_fitting_layer_start),
                int(num_layers),
                float(self._trunk_search_circle_fitting_layer_height),
                float(self._trunk_search_circle_fitting_layer_overlap),
                float(self._trunk_search_circle_fitting_switch_buffer_threshold),
                float(self._trunk_search_circle_fitting_small_buffer_width),
                float(self._trunk_search_circle_fitting_large_buffer_width),
                0,
                int(self._num_workers),
            )

            if num_instances == 0:
                return (
                    layer_circles,
                    layer_ellipses,
                    np.empty((0, 2), dtype=trunk_layer_xyz.dtype),
                    np.empty(0, dtype=np.int64),
                )

            min_radius = self._trunk_search_circle_fitting_min_trunk_diameter / 2
            max_radius = self._trunk_search_circle_fitting_max_trunk_diameter / 2
            min_completeness_idx = self._trunk_search_circle_fitting_min_completeness_idx
            bandwidth = 0.01

            with Profiler("Circle fitting to full-resolution trunk candidates", self._performance_tracker):
                self._logger.info("Fit circles...")

                circle_detector: Union[MEstimator, Ransac]
                if self._trunk_search_circle_fitting_method == "m-estimator":
                    circle_detector = MEstimator(
                        bandwidth=bandwidth,
                        break_min_change=1e-6,
                        min_step_size=1e-10,
                        max_iterations=300,
                        armijo_min_decrease_percentage=0.5,
                        armijo_attenuation_factor=0.25,
                    )
                    circle_detector.detect(
                        trunk_layer_xy,
                        batch_lengths=batch_lengths_xy,
                        n_start_x=3,
                        n_start_y=3,
                        min_start_radius=min_radius,
                        max_start_radius=max_radius,
                        n_start_radius=3,
                        num_workers=self._num_workers,
                    )
                else:
                    circle_detector = Ransac(bandwidth=bandwidth)
                    circle_detector.detect(
                        trunk_layer_xy,
                        batch_lengths=batch_lengths_xy,
                        num_workers=self._num_workers,
                        seed=self._random_seed,
                    )

                circle_detector.filter(
                    max_circles=1,
                    deduplication_precision=4,
                    min_circumferential_completeness_idx=min_completeness_idx,
                    circumferential_completeness_idx_max_dist=bandwidth,
                    circumferential_completeness_idx_num_regions=int(365 / 5),
                    non_maximum_suppression=True,
                    num_workers=self._num_workers,
                )

            with Profiler("Ellipse fitting to full-resolution trunk candidates", self._performance_tracker):
                self._logger.info("Fit ellipses...")

            ellipses = fit_ellipse(trunk_layer_xy, batch_lengths_xy, num_workers=self._num_workers)

            visualization_tasks: List[Tuple[Any, ...]] = []

            circle_idx = 0
            batch_start_idx = 0
            for label in range(num_instances):
                for layer in range(self._trunk_search_circle_fitting_num_layers):
                    flat_idx = label * num_layers + layer
                    batch_end_idx = batch_start_idx + batch_lengths_xy[flat_idx]

                    if circle_detector.batch_lengths_circles[flat_idx] > 0:
                        layer_circles[label, layer] = circle_detector.circles[circle_idx]

                        if self._visualization_folder is not None and point_cloud_id is not None:
                            visualization_path = (
                                self._visualization_folder
                                / point_cloud_id
                                / f"refined_circle_trunk_{label}_layer_{layer}.png"
                            )
                            visualization_tasks.append(
                                (
                                    trunk_layer_xy[batch_start_idx:batch_end_idx],
                                    visualization_path,
                                    circle_detector.circles[circle_idx],
                                )
                            )

                    if ellipses[flat_idx, 2] != -1:
                        radius_major, radius_minor = ellipses[flat_idx, 2:4]

                        if radius_minor / radius_major >= self._trunk_search_ellipse_filter_threshold:
                            layer_ellipses[label, layer] = ellipses[flat_idx]

                        if self._visualization_folder is not None and point_cloud_id is not None:
                            visualization_path = (
                                self._visualization_folder
                                / point_cloud_id
                                / f"refined_ellipse_trunk_{label}_layer_{layer}.png"
                            )
                            visualization_tasks.append(
                                (
                                    trunk_layer_xy[batch_start_idx:batch_end_idx],
                                    visualization_path,
                                    None,
                                    ellipses[flat_idx],
                                )
                            )

                    circle_idx += circle_detector.batch_lengths_circles[flat_idx]
                    batch_start_idx = batch_end_idx

        if len(visualization_tasks) > 0:
            with Profiler(
                "Visualization of circles and ellipses fitted to full-resolution trunk candidates",
                self._performance_tracker,
            ):
                self._logger.info("Visualize circles and ellipses fitted to full-resolution trunk candidates...")
                num_workers = self._num_workers if self._num_workers > 0 else multiprocessing.cpu_count()
                with multiprocessing.Pool(processes=num_workers) as pool:
                    pool.starmap(plot_fitted_shape, visualization_tasks)

        return layer_circles, layer_ellipses, trunk_layer_xy, batch_lengths_xy

    def filter_instances_trunk_layer_std(  # pylint: disable=too-many-locals
        self, layer_circles: npt.NDArray, layer_ellipses: npt.NDArray
    ) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        r"""
        Filters the point clusters that may represent individual tree trunks based on the circles and ellipses fitted to
        different horizontal layers of the clusters. For each cluster, the standard deviation of the fitted circle or
        ellipse diameters is computed for all possible combinations of
        :code:`self._trunk_search_circle_fitting_std_num_layers` layers. If for any of the combinations the standard
        deviation is smaller than or equal to :code:`self._trunk_search_circle_fitting_max_std_diameter`, the cluster is
        kept, otherwise it is discarded. If at least :code:`self._trunk_search_circle_fitting_std_num_layers` circles
        are have been found for a trunk cluster, only the fitted circles are considered for this filtering step.
        Otherwise, the filtering is done based on the fitted ellipses.

        Args:
            layer_circles: Parameters of the circles that were fitted to the horizontal layers of each cluster. Each
                circle must be represented by three values, namely the x- and y-coordinates of its center and its
                radius. If no circle was found for a certain layer, the circle parameters for that layer must be set to
                -1.
            layer_ellipses: Parameters of the ellipses that were fitted to the horizontal layers of each cluster. Each
                ellipse must represented by five values, namely the x- and y-coordinates of its center, its
                radius along the semi-major and along the semi-minor axis, and the counterclockwise angle of rotation
                from the x-axis to the semi-major axis of the ellipse. If no ellipse was found for a certain layer, the
                ellipse parameters for that layer must be set to -1.

        Returns:
            Tuple of three arrays. The first array contains a boolean mask indicating which trunk clusters are retained
            after the filtering. The second contains the indices of the combination of layers with the lowest standard
            deviation of the circle diameters for each cluster. If for a trunk cluster less than
            :code:`self._trunk_search_circle_fitting_std_num_layers` circles were found, the indices for that cluster
            are set to -1. The third array contains the indices of the combination of layers with the lowest standard
            deviation of the ellipse diameters for each cluster. If more than
            :code:`self._trunk_search_circle_fitting_std_num_layers` circles were found for a trunk cluster, the
            ellipses are not considered for the filtering and the indices for that cluster are set to -1.

        Shape:
            - :code:`layer_circles`: :math:`(T, L, 3)`
            - :code:`layer_ellipses`: :math:`(T, L, 5)`
            - Output: Tuple of three arrays. The first array has shape :math:`(T)`, and the second and the third have
              shape :math:`(T', L')`.

            | where
            |
            | :math:`T = \text{ number of trunk clusters before the filtering}`
            | :math:`T' = \text{ number of trunk clusters after the filtering}`
            | :math:`L = \text{ number of horinzontal layers to which circles and ellipses are fitted}`
            | :math:`L' = \text{ number of horinzontal layers sampled for the calculation of the standard deviation}`
        """

        num_instances = len(layer_circles)
        num_layers = layer_circles.shape[1]

        filter_mask = np.zeros(num_instances, dtype=bool)
        best_circle_combination = np.full(
            (num_instances, self._trunk_search_circle_fitting_std_num_layers), dtype=np.int64, fill_value=-1
        )
        best_ellipse_combination = np.full(
            (num_instances, self._trunk_search_circle_fitting_std_num_layers), dtype=np.int64, fill_value=-1
        )

        for label in range(num_instances):
            existing_circle_layers = np.arange(num_layers, dtype=np.int64)[layer_circles[label][:, 2] != -1]
            existing_ellipse_layers = np.arange(num_layers, dtype=np.int64)[layer_ellipses[label][:, 2] != -1]

            if (
                len(existing_circle_layers) < self._trunk_search_circle_fitting_std_num_layers
                and len(existing_ellipse_layers) < self._trunk_search_circle_fitting_std_num_layers
            ):
                continue

            if len(existing_circle_layers) >= self._trunk_search_circle_fitting_std_num_layers:
                circle_diameters = layer_circles[label, :, 2] * 2
                combinations = list(
                    itertools.combinations(existing_circle_layers, self._trunk_search_circle_fitting_std_num_layers)
                )
                minimum_std = np.inf
                for combination in combinations:
                    diameter_std = np.std(circle_diameters[np.array(combination)])
                    position_std = np.zeros(2, dtype=diameter_std.dtype)
                    if self._trunk_search_circle_fitting_max_std_position is not None:
                        position_std = np.std(layer_circles[label, np.array(combination), :2], axis=0)
                    if (
                        diameter_std <= self._trunk_search_circle_fitting_max_std_diameter
                        and (position_std <= self._trunk_search_circle_fitting_max_std_position).all()
                    ):
                        filter_mask[label] = True
                        if diameter_std < minimum_std:
                            minimum_std = diameter_std
                            best_circle_combination[label] = combination
            if not filter_mask[label]:
                ellipse_diameters = (layer_ellipses[label, :, 2:4]).sum(axis=-1)
                combinations = list(
                    itertools.combinations(existing_ellipse_layers, self._trunk_search_circle_fitting_std_num_layers)
                )
                minimum_std = np.inf
                for combination in combinations:
                    diameter_std = np.std(ellipse_diameters[np.array(combination)])

                    position_std = np.zeros(2, dtype=diameter_std.dtype)
                    if self._trunk_search_circle_fitting_max_std_position is not None:
                        position_std = np.std(layer_ellipses[label, np.array(combination), :2], axis=0)

                    if (
                        diameter_std <= self._trunk_search_circle_fitting_max_std_diameter
                        and (position_std <= self._trunk_search_circle_fitting_max_std_position).all()
                    ):
                        filter_mask[label] = True
                        if diameter_std < minimum_std:
                            minimum_std = diameter_std
                            best_ellipse_combination[label] = combination

        return filter_mask, best_circle_combination[filter_mask], best_ellipse_combination[filter_mask]

    def rename_visualizations_after_filtering(
        self, filter_mask: npt.NDArray[np.bool_], point_cloud_id: Optional[str]
    ) -> None:
        r"""
        Renames visualization files that plot the circles / ellipses fitted to trunk layers after the filtering of the
        trunks. In the course of this, the trunk IDs in the file names are updated and the postfix :code:`_valid` is
        added to the file names for clusters that were kept during filtering, while the postfix :code:`_invalid` is
        added to the file names for clusters that were filtered out.

        Args:
            filter_mask: Boolean mask indicating which trunks were kept during filtering.
            point_cloud_id: ID of the point cloud used in the file names of the visualizations. Defaults to
                :code:`None`, which means that no visualizations were created.

        Shape:
            - :code:`filter_mask`: :math:`(T)`

            | where
            |
            | :math:`T = \text{ number of trunk clusters}`
        """

        if self._visualization_folder is None or point_cloud_id is None:
            return

        next_valid_label = 0
        next_invalid_label = filter_mask.sum()
        for label, is_valid in enumerate(filter_mask):
            for layer in range(self._trunk_search_circle_fitting_num_layers):
                visualization_paths = [
                    self._visualization_folder / point_cloud_id / f"preliminary_circle_trunk_{label}_layer_{layer}.png",
                    self._visualization_folder
                    / point_cloud_id
                    / f"preliminary_ellipse_trunk_{label}_layer_{layer}.png",
                    self._visualization_folder / point_cloud_id / f"refined_circle_trunk_{label}_layer_{layer}.png",
                    self._visualization_folder / point_cloud_id / f"refined_ellipse_trunk_{label}_layer_{layer}.png",
                ]
                for visualization_path in visualization_paths:
                    if not visualization_path.exists():
                        continue

                    if is_valid:
                        new_visualization_path_str = str(visualization_path)
                        new_visualization_path_str = new_visualization_path_str.replace(
                            f"trunk_{label}_", f"trunk_{next_valid_label}_"
                        )
                        new_visualization_path_str = new_visualization_path_str.replace(
                            f"layer_{layer}.png", f"layer_{layer}_valid.png"
                        )
                    else:
                        new_visualization_path_str = str(visualization_path)
                        new_visualization_path_str = new_visualization_path_str.replace(
                            f"trunk_{label}_", f"trunk_{next_invalid_label}_"
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

    def compute_trunk_positions(
        self,
        layer_circles: npt.NDArray,
        layer_ellipses: npt.NDArray,
        layer_heights: npt.NDArray,
        best_circle_combination: npt.NDArray[np.int64],
        best_ellipse_combination: npt.NDArray[np.int64],
    ) -> npt.NDArray:
        r"""
        Calculates the trunk positions using the circles or ellipses fitted to the horizontal layers of the trunks. For
        this purpose, the combination of those :code:`self._trunk_search_circle_fitting_std_num_layers` circles or
        ellipses with the smallest standard deviation of the diameters is selected. A linear model is fitted to these
        circles or ellipses to predict the centers of the circles or ellipses as a function of the height above the
        ground. The prediction of the linear model for a height of 1.3 m above the ground is returned as the trunk
        position.

        Args:
            layer_circles: Parameters of the circles that were fitted to the horizontal layers of each cluster. Each
                circle must be represented by three values, namely the x- and y-coordinates of its center and its
                radius. If no circle was found for a certain layer, the circle parameters for that layer must be set to
                -1.
            layer_ellipses: Parameters of the ellipses that were fitted to the horizontal layers of each cluster. Each
                ellipse must represented by five values, namely the x- and y-coordinates of its center, its
                radius along the semi-major and along the semi-minor axis, and the counterclockwise angle of rotation
                from the x-axis to the semi-major axis of the ellipse. If no ellipse was found for a certain layer, the
                ellipse parameters for that layer must be set to -1.
            layer_heights: Heights above the ground of the horizontal layers to which the circles or ellipses were
                fitted.
            best_circle_combination: Indices of the combination of layers with the lowest standard deviation of the
                circle diameters for each trunk cluster. If less than
                :code:`self._trunk_search_circle_fitting_std_num_layers` circles were found for a trunk cluster, the
                indices for that cluster must be set to -1.
            best_ellipse_combination: Indices of the combination of layers with the lowest standard deviation of the
                ellipse diameters for each cluster. If more than
                :code:`self._trunk_search_circle_fitting_std_num_layers` circles were found for a trunk cluster, the
                ellipses are not considered for calculating the trunk position.

        Returns:
            X- and y-coordinates of the position of each trunk at breast height (1.3 m).

        Shape:
            - :code:`layer_circles`: :math:`(T, L, 3)`
            - :code:`layer_ellipses`: :math:`(T, L, 5)`
            - :code:`layer_heights`: :math:`(L)`
            - :code:`best_circle_combination`: :math:`(T, L')`
            - :code:`best_ellipse_combination`: :math:`(T, L')`
            - Output: :math:`(T, 2)`.

            | where
            |
            | :math:`T = \text{ number of trunk clusters}`
            | :math:`L = \text{ number of horinzontal layers to which circles and ellipses are fitted}`
            | :math:`L' = \text{ number of horinzontal layers sampled for the calculation of the standard deviation}`
        """

        num_instances = len(layer_circles)

        trunk_positions = np.empty((num_instances, 2), dtype=layer_circles.dtype, order="F")

        for label in range(num_instances):
            has_circle_combination = best_circle_combination[label, 0] != -1
            if has_circle_combination:
                circles_or_ellipses = layer_circles[label, best_circle_combination[label]]
                layer_heights_combination = layer_heights[best_circle_combination[label]]
            else:
                circles_or_ellipses = layer_ellipses[label, best_ellipse_combination[label]]
                layer_heights_combination = layer_heights[best_ellipse_combination[label]]

            centers = circles_or_ellipses[:, :2]

            prediction_x, _ = estimate_with_linear_model(
                layer_heights_combination, centers[:, 0], np.array([1.3], dtype=layer_circles.dtype)
            )
            trunk_positions[label, 0] = prediction_x[0]
            prediction_y, _ = estimate_with_linear_model(
                layer_heights_combination, centers[:, 1], np.array([1.3], dtype=layer_circles.dtype)
            )
            trunk_positions[label, 1] = prediction_y[0]

        return trunk_positions

    def compute_trunk_diameters(  # pylint: disable=too-many-locals,
        self,
        layer_circles: npt.NDArray,
        layer_ellipses: npt.NDArray,
        layer_heights: npt.NDArray,
        trunk_layer_xy: npt.NDArray,
        batch_lengths_xy: npt.NDArray[np.int64],
        best_circle_combination: npt.NDArray[np.int64],
        best_ellipse_combination: npt.NDArray[np.int64],
        *,
        point_cloud_id: Optional[str] = None,
    ) -> npt.NDArray:
        r"""
        Calculates the trunk diameters using the circles or ellipses fitted to the horizontal layers of the trunks. For
        this purpose, the combination of those :code:`self._trunk_search_circle_fitting_std_num_layers` circles or
        ellipses with the smallest standard deviation of the diameters is selected. The trunk diameter for each selected
        layer is computed by fitting a GAM to the points of that layer, using the centers of the previously fitted
        circles or ellipses to normalize the points. A linear model is then fitted to the trunk diameters obtained from
        the GAM to predict the diameters of the circles or ellipses as a function of the height above the ground. The
        prediction of the linear model for a height of 1.3 m above the ground is returned as the trunk diameter.

        Args:
            layer_circles: Parameters of the circles that were fitted to the horizontal layers of each cluster. Each
                circle must be represented by three values, namely the x- and y-coordinates of its center and its
                radius. If no circle was found for a certain layer, the circle parameters for that layer must be set to
                -1.
            layer_ellipses: Parameters of the ellipses that were fitted to the horizontal layers of each cluster. Each
                ellipse must represented by five values, namely the x- and y-coordinates of its center, its
                radius along the semi-major and along the semi-minor axis, and the counterclockwise angle of rotation
                from the x-axis to the semi-major axis of the ellipse. If no ellipse was found for a certain layer, the
                ellipse parameters for that layer must be set to -1.
            layer_heights: Heights above the ground of the horizontal layers to which the circles or ellipses were
                fitted.
            trunk_layer_xy: X- and y-coordinates of the points belonging to the different horizontal layers of the
                trunks. Points that belong to the same layer of the same trunk must be stored consecutively and the
                number of points belonging to each layer must be specified using :code:`batch_lengths_xy`.
            batch_lengths_xy: Number of points belonging to each horizontal layer of each trunk.
            best_circle_combination: Indices of the combination of layers with the lowest standard deviation of the
                circle diameters for each trunk cluster. If less than
                :code:`self._trunk_search_circle_fitting_std_num_layers` circles were found for a trunk cluster, the
                indices for that cluster must be set to -1.
            best_ellipse_combination: Indices of the combination of layers with the lowest standard deviation of the
                ellipse diameters for each cluster. If more than
                :code:`self._trunk_search_circle_fitting_std_num_layers` circles were found for a trunk cluster, the
                ellipses are not considered for calculating the trunk position.
            point_cloud_id: ID of the point cloud to be used in the file names of the visualization results. Defaults to
                :code:`None`, which means that no visualizations are created.

        Returns:
            Diameters at breast height of each trunk (1.3 m).

        Shape:
            - :code:`layer_circles`: :math:`(T, L, 3)`
            - :code:`layer_ellipses`: :math:`(T, L, 5)`
            - :code:`layer_heights`: :math:`(L)`
            - :code:`trunk_layer_xy`: :math:`(N, 2)`
            - :code:`batch_lengths_xy`: :math:`(T \cdot L)`
            - :code:`best_circle_combination`: :math:`(T, L')`
            - :code:`best_ellipse_combination`: :math:`(T, L')`
            - Output: :math:`(T)`.

            | where
            |
            | :math:`N = \text{ number of points in the trunk layer}`
            | :math:`T = \text{ number of trunk clusters}`
            | :math:`L = \text{ number of horinzontal layers to which circles and ellipses are fitted}`
            | :math:`L' = \text{ number of horinzontal layers sampled for the calculation of the standard deviation}`
        """

        num_instances = len(layer_circles)
        len_layer_combination = best_circle_combination.shape[1]
        num_layers = self._trunk_search_circle_fitting_num_layers

        trunk_diameters = np.empty(num_instances, dtype=trunk_layer_xy.dtype)

        visualization_tasks = []

        batch_starts = np.cumsum(np.concatenate((np.array([0], dtype=np.int64), batch_lengths_xy)))[:-1]
        for label in range(num_instances):
            has_circle_combination = best_circle_combination[label, 0] != -1
            if has_circle_combination:
                circles_or_ellipses = layer_circles[label, best_circle_combination[label]]
                layer_heights_combination = layer_heights[best_circle_combination[label]]
            else:
                circles_or_ellipses = layer_ellipses[label, best_ellipse_combination[label]]
                layer_heights_combination = layer_heights[best_ellipse_combination[label]]

            centers = circles_or_ellipses[:, :2]

            layer_diameters = np.empty(len_layer_combination, dtype=trunk_layer_xy.dtype)

            best_combination = (
                best_circle_combination[label] if has_circle_combination else best_ellipse_combination[label]
            )
            for layer_idx, layer in enumerate(best_combination):
                flat_idx = label * num_layers + layer
                batch_start_idx = batch_starts[flat_idx]
                batch_end_idx = batch_start_idx + batch_lengths_xy[flat_idx]
                diameter_gam = None
                polygon_vertices = None
                if batch_start_idx < batch_end_idx:
                    diameter_gam, polygon_vertices = self.diameter_estimation_gam(
                        trunk_layer_xy[batch_start_idx:batch_end_idx], centers[layer_idx]
                    )
                if diameter_gam is not None:
                    layer_diameters[layer_idx] = diameter_gam
                else:
                    if has_circle_combination:
                        layer_diameters[layer_idx] = circles_or_ellipses[layer_idx, 2] * 2
                    else:
                        ellipse_diameter = circles_or_ellipses[layer_idx, 2:4].sum()
                        layer_diameters[layer_idx] = ellipse_diameter

                if (
                    polygon_vertices is not None
                    and self._visualization_folder is not None
                    and point_cloud_id is not None
                ):
                    if diameter_gam is not None:
                        file_name = f"gam_trunk_{label}_layer_{layer}.png"
                    else:
                        file_name = f"gam_trunk_{label}_layer_{layer}_invalid.png"

                    visualization_path = self._visualization_folder / point_cloud_id / file_name
                    visualization_tasks.append(
                        (
                            trunk_layer_xy[batch_start_idx:batch_end_idx],
                            visualization_path,
                            None,
                            None,
                            polygon_vertices,
                        )
                    )

            prediction, _ = estimate_with_linear_model(
                layer_heights_combination, layer_diameters, np.array([1.3], dtype=trunk_layer_xy.dtype)
            )
            trunk_diameters[label] = prediction[0]

        if len(visualization_tasks) > 0:
            num_workers = self._num_workers if self._num_workers > 0 else multiprocessing.cpu_count()
            with multiprocessing.Pool(processes=num_workers) as pool:
                pool.starmap(plot_fitted_shape, visualization_tasks)

        return trunk_diameters

    def diameter_estimation_gam(  # pylint: disable=too-many-locals
        self,
        points: npt.NDArray,
        center: npt.NDArray,
        eps: Optional[float] = None,
    ) -> Tuple[Optional[float], npt.NDArray]:
        r"""
        Estimates the diameter of a tree trunk using a GAM. It is assumed that a circle or an ellipse has already been
        fitted to the points of the tree trunk. To create the GAM, the points are converted into polar coordinates,
        using the center of the previously fitted circle or ellipse as the coordinate origin. The GAM is then fitted to
        predict the radius of the points based on the angles. The fitted GAM is then used to predict the trunk's
        boundary polygon and the trunk diameter is computed from the area of the boundary polygon. If the difference
        between the predicted radii of consecutive angles is greater than the value of
        :code:`trunk_search_gam_max_radius_diff`, the fitted GAM is considered invalid, and None is returned for the
        trunk diameter.

        Args:
            points: Points belonging to the trunk slice for which to estimate the diameter.
            center: Center of the circle or ellipse that has been fitted to the trunk.
            eps: Small epsilon value that is added to some terms to avoid division by zero. Defaults to
                :code:`sys.float_info.epsilon`.

        Returns:
            Tuple with two elements: The first is the estimated trunk diameter and the second is an array containing the
            sorted vertices of the trunk's boundary polygon predicted by the GAM. The estimated trunk diameter may be
            :code:`None` if the fitted GAM is invalid.

        Shape:
            - :code:`points`: :math:`(N, 2)` or :math:`(N, 3)`
            - :code:`center`: :math:`(2)`

            | where
            |
            | :math:`N = \text{ number of points}`
        """

        if eps is None:
            eps = sys.float_info.epsilon

        points_centered = points[:, :2] - center.reshape((-1, 2))

        # calculate polar coordinates
        polar_radius = np.linalg.norm(points_centered[:, :2], axis=-1)

        # add small random offset to avoid perfect separation
        polar_radius += self._random_generator.normal(0, 1e-8, len(points))

        polar_angle = np.arctan2(points_centered[:, 1], points_centered[:, 0])

        # fit GAM
        xy = np.column_stack((polar_angle, polar_radius))
        gam = LinearGAM(s(0, basis="cp", edge_knots=[-np.pi, np.pi])).fit(xy[:, 0], xy[:, 1])

        # predict stem outline using fitted GAM
        polar_angles = np.asarray([-np.pi + 2 * np.pi * k / 360 for k in range(360)])
        polar_radii = gam.predict(polar_angles)

        cartesian_coords_x = polar_radii * np.cos(polar_angles)
        cartesian_coords_y = polar_radii * np.sin(polar_angles)

        cartesian_coords = np.column_stack((cartesian_coords_x, cartesian_coords_y))
        cartesian_coords = cartesian_coords + center.reshape((-1, 2))

        radius_diff = polar_radii.max() - polar_radii.min()

        if (
            self.trunk_search_gam_max_radius_diff is not None and radius_diff > self._trunk_search_gam_max_radius_diff
        ) or (polar_radii < 0).any():
            return None, cartesian_coords

        trunk_area = polygon_area(cartesian_coords_x, cartesian_coords_y)
        diameter_gam = 2 * np.sqrt(trunk_area / np.pi)

        return diameter_gam, cartesian_coords

    def segment_crowns(
        self,
        xyz: npt.NDArray,
        dists_to_dtm: npt.NDArray,
        is_tree: npt.NDArray[np.bool_],
        tree_positions: npt.NDArray,
        trunk_diameters: npt.NDArray,
        cluster_labels: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
        r"""
        Computes a point-wise segmentation of the individual trees using a region growing procedure. In the first step,
        the region growing procedure selects an initial set of seed points for each tree. These should be points
        that are very likely to belong to the corresponding tree. In an iterative process, the sets of points assigned
        to each tree are then expanded. In each iteration, the neighboring points of each seed point within a certain
        search radius are determined. Neighboring points that are not yet assigned to any tree are added to the same
        tree as the seed point and become seed points in the next iteration. The region growing continues until there
        are no more seed points to be processed or the maximum number of iterations is reached.

        To select the initial seed points for a given tree, the following approach is used: A cylinder with a height of
        :code:`crown_seg_seed_layer_height` and a diameter of :code:`self._crown_seg_seed_diameter_factor * d`
        is considered, where :code:`d` is the tree's trunk diameter at breast height, which has been computed in the
        previous step. The cylinder's center is positioned at the trunk center at breast height, which also has been
        computed in the previous step. All points within the cylinder are selected as seed points.

        The search radius for the iterative region growing procedure is set as follows: First, the search radius is set
        to the voxel size used for voxel-based subsampling, which is performed before region growing. The search radius
        is increased by the voxel size if one of the following conditions is fulfilled at the end of a region growing
        iteration:

        1. The ratio between the number of points newly assigned to trees in the iteration and the number of remaining,
        unassigned points is below :code:`self._crown_seg_min_total_assignment_ratio`.
        2. The ratio between the number of trees to which new points have been assigned in the iteration and the total
        number of trees is below :code:`self._crown_seg_min_tree_assignment_ratio`.

        The search radius is increased up to a maximum radius of :code:`self._crown_seg_max_search_radius`. If the
        search radius has not been increased for :code:`self._crown_seg_decrease_search_radius_after_num_iter`, it
        is reduced by the voxel size.

        To promote upward growth, the z-coordinates of the points are divided by :code:`crown_seg_z_scale` before
        the region growing.

        Since the terrain filtering in the first step of the algorithm may be inaccurate and some tree points may be
        falsely classified as terrain points, both terrain and non-terrain points are considered by the region growing
        procedure. However, to prevent large portions of terrain points from being included in tree instances, terrain
        points are only assigned to if their cumulative search distance from the initial seed point is below the
        threshold defined by :code:`self._crown_seg_cum_search_dist_include_ground`. The cumulative search distance
        is defined as the total distance traveled between consecutive points until reaching a terrain point.

        Args:
            xyz: Non-normalized coordinates of the points which to consider in the region growing. This can include
                both terrain and non-terrain points.
            dists_to_dtm: Height of each point above the ground.
            is_tree: Boolean array indicating which points have been identified as potential tree points, i.e.,
                non-vegetation points. The points for which the corresponding entry is :code:`True` are considered in
                all region growing iterations, while terrain points are only considered if the cumulative search
                distance is below the threshold defined by :code:`_crown_seg_cum_search_dist_include_ground`.
            tree_positions: X- and y-coordinates of the positions of the trees to be used for seed point selection.
            trunk_diameters: Trunk diameters of of the trees to be used for seed point selection.
            cluster_labels: Cluster labels for each point that represent the detected clusters of trunk points. Points
                that do not belong to any cluster must have the cluster label :code:`-1`.

        Returns:
            Tree instance labels for all points. For points not belonging to any tree, the label is set to the invalid
            tree ID specified in the algorithm's constructor.

        Raises:
            ValueError: If :code:`xyz`, :code:`dists_to_dtm`, and :code:`is_tree` have different lengths or if
                :code:`tree_positions` and :code:`trunk_diameters` have different lengths.

        Shape:
            - :code:`xyz`: :math:`(N, 3)`
            - :code:`dists_to_dtm`: :math:`(N)`
            - :code:`is_tree`: :math:`(N)`
            - :code:`tree_positions`: :math:`(T, 2)`
            - :code:`trunk_diameters`: :math:`(T)`
            - :code:`cluster_labels`: :math:`(N)`
            - Output: :math:`(N)`

            | where
            |
            | :math:`N = \text{ number of points}`
            | :math:`T = \text{ number of tree instances}`
        """

        if len(xyz) != len(dists_to_dtm):
            raise ValueError("xyz and dists_to_dtm must have the same length.")
        if len(xyz) != len(is_tree):
            raise ValueError("xyz and is_tree must have the same length.")
        if len(xyz) != len(cluster_labels):
            raise ValueError("xyz and cluster_labels must have the same length.")

        downsampled_xyz, downsampled_indices, inverse_indices = voxel_downsampling(
            xyz, voxel_size=self._crown_seg_voxel_size
        )
        dists_to_dtm = dists_to_dtm[downsampled_indices]
        is_tree = is_tree[downsampled_indices]
        cluster_labels = cluster_labels[downsampled_indices]

        if not downsampled_xyz.flags.f_contiguous:
            downsampled_xyz = downsampled_xyz.copy(order="F")
        if not tree_positions.flags.f_contiguous:
            tree_positions = tree_positions.copy(order="F")
        dists_to_dtm = dists_to_dtm.astype(downsampled_xyz.dtype)
        tree_positions = tree_positions.astype(downsampled_xyz.dtype)
        trunk_diameters = trunk_diameters.astype(downsampled_xyz.dtype)

        instance_ids = segment_tree_crowns_cpp(
            downsampled_xyz,
            dists_to_dtm,
            is_tree,
            tree_positions,
            trunk_diameters,
            cluster_labels,
            float(self._crown_seg_voxel_size),
            float(self._crown_seg_z_scale),
            float(self._crown_seg_seed_layer_height),
            float(self._crown_seg_seed_diameter_factor),
            float(self._crown_seg_seed_min_diameter),
            float(self._crown_seg_min_total_assignment_ratio),
            float(self._crown_seg_min_tree_assignment_ratio),
            float(self._crown_seg_max_search_radius),
            int(self._crown_seg_decrease_search_radius_after_num_iter),
            int(self._crown_seg_max_iterations),
            float(self._crown_seg_cum_search_dist_include_terrain),
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
        xyz: npt.NDArray,
        intensities: Optional[npt.NDArray] = None,
        point_cloud_id: Optional[str] = None,
        crs: Optional[str] = None,
    ) -> Tuple[npt.NDArray[np.int64], npt.NDArray, npt.NDArray]:
        r"""
        Runs the tree instance segmentation for the given point cloud.

        Args:
            xyz: 3D coordinates of all points in the point cloud.
            intensities: Reflection intensities of all points in the point cloud. If set to :code:`None`, filtering
                steps that use intensity values are skipped. Defaults to :code:`None`.
            point_cloud_id: ID of the point cloud to be used in the file names of the visualization results. Defaults to
                :code:`None`, which means that no visualizations are created.
            crs: EPSG code of the coordinate reference system of the input point cloud. The EPSG code is used to set the
                coordinate reference system when exporting intermediate data, such as a digital terrain model file.
                Defaults to :code:`None`, which means that no coordinate reference system is set for the exported data.

        Returns:
            Tree instance labels for all points. For points not belonging to any tree, the label is set to -1.

        Raises:
            ValueError: If :code:`intensities` is not :code:`None`, and :code:`xyz` and :code:`intensities` have
                different lengths.

        Shape:
            - :code:`xyz`: :math:`(N, 3)`
            - :code:`intensities`: :math:`(N)`
            - Output: :math:`(N)`

            | where
            |
            | :math:`N = \text{ number of points}`
        """

        self._random_generator = np.random.default_rng(seed=self._random_seed)

        if intensities is not None and len(xyz) != len(intensities):
            raise ValueError("xyz and intensities must have the same length.")

        with Profiler("Construction of a digital terrain model", self._performance_tracker):
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

            with Profiler("Rasterization", self._performance_tracker):
                self._logger.info("Construct rasterized digital terrain model...")
                dtm, dtm_offset = create_digital_terrain_model(
                    xyz[terrain_classification == 0],
                    grid_resolution=self._dtm_resolution,
                    k=self._dtm_k,
                    p=self._dtm_p,
                    voxel_size=self._dtm_voxel_size,
                    num_workers=self._num_workers,
                )
                del terrain_classification

                if self._visualization_folder is not None and point_cloud_id is not None:
                    self.export_dtm(dtm, dtm_offset, point_cloud_id, crs=crs)

        with Profiler("Height normalization", self._performance_tracker):
            self._logger.info("Normalize point heights...")
            dists_to_dtm = distance_to_dtm(xyz, dtm, dtm_offset, self._dtm_resolution)
            is_tree = dists_to_dtm >= self._csf_tree_classification_threshold

        with Profiler("Detection of tree trunks", self._performance_tracker):
            self._logger.info("Detect trunks...")
            trunk_layer_filter = np.flatnonzero(
                np.logical_and(
                    dists_to_dtm >= self._trunk_search_min_z,
                    dists_to_dtm < self._trunk_search_max_z,
                )
            )
            trunk_layer_xyz = xyz[trunk_layer_filter]

            if self._visualization_folder is not None and point_cloud_id is not None:
                self.export_point_cloud(
                    trunk_layer_xyz,
                    {"dist_to_dtm": dists_to_dtm[trunk_layer_filter]},
                    "trunk_layer",
                    point_cloud_id,
                    crs=crs,
                )

            trunk_positions, trunk_diameters, cluster_labels = self.find_trunks(
                trunk_layer_xyz,
                dtm,
                dtm_offset,
                intensities=intensities[trunk_layer_filter] if intensities is not None else None,
                point_cloud_id=point_cloud_id,
                crs=crs,
            )
            cluster_labels_full = np.full(len(xyz), fill_value=-1, dtype=np.int64)
            cluster_labels_full[trunk_layer_filter] = cluster_labels
            del dtm
            del dtm_offset
            del trunk_layer_filter
            del trunk_layer_xyz
            del cluster_labels

        if len(trunk_positions) == 0:
            return (
                np.full(len(xyz), fill_value=-1, dtype=xyz.dtype),
                trunk_positions,
                trunk_diameters,
            )

        with Profiler("Crown segmentation", self._performance_tracker):
            self._logger.info("Segment tree crowns...")
            instance_ids = self.segment_crowns(
                xyz, dists_to_dtm, is_tree, trunk_positions, trunk_diameters, cluster_labels_full
            )

        self._logger.info("Finished segmentation.")

        print(self.performance_metrics())

        return instance_ids, trunk_positions, trunk_diameters
