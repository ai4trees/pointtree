"""Presets for the treeX algorithm."""

__all__ = ["TreeXPreset", "TreeXPresetOriginal", "TreeXPresetTLS", "TreeXPresetULS"]

from collections.abc import Mapping
from dataclasses import dataclass, make_dataclass, field, asdict
import inspect
from typing import Any, List, Optional, Tuple

from ._tree_x_algorithm import TreeXAlgorithm


def create_default_preset() -> "TreeXPreset":
    """
    Returns:
        Preset class that contains the default settings of the TreeX algorithm.
    """

    constructor_signature = inspect.signature(TreeXAlgorithm.__init__)
    fields: List[Tuple[str, Any, Any]] = []

    for name, param in constructor_signature.parameters.items():
        if name == "self":
            continue
        if param.default is not inspect.Parameter.empty:
            type_annotation = param.annotation if param.annotation is not inspect.Parameter.empty else Any
            fields.append((name, type_annotation, field(default=param.default)))  # pylint: disable=invalid-field-call

    return make_dataclass("TreeXPresetDefault", fields=fields)  # type: ignore[return-value]


class TreeXPreset(create_default_preset(), Mapping):  # type: ignore[misc]
    """
    Preset for the treeX algorithm containing the default settings of the algorithm.
    """

    def __len__(self):
        """
        Returns:
            Number of parameters included in the preset.
        """

        return len(asdict(self))

    def __iter__(self):
        """
        Returns:
            Iterator over the preset.
        """

        return iter(asdict(self))

    def __getitem__(self, key: str) -> Any:
        """
        Args:
            key: Parameter name.

        Returns: Parameter value for the given parameter name.
        """

        return asdict(self)[key]


@dataclass
class TreeXPresetOriginal(TreeXPreset):  # pylint: disable=too-many-instance-attributes
    """
    Preset for the treeX algorithm with settings similar to those used in the papers `Tockner, Andreas, et al. \
    "Automatic Tree Crown Segmentation Using Dense Forest Point Clouds from Personal Laser Scanning (PLS)." \
    International Journal of Applied Earth Observation and Geoinformation 114 (2022): 103025. \
    <https://doi.org/10.1016/j.jag.2022.103025>`__. and 
    """

    stem_search_min_z: float = 1.0
    stem_search_max_z: float = 3.0
    stem_search_voxel_size: float = 0.015
    stem_search_dbscan_2d_eps: float = 0.025
    stem_search_dbscan_2d_min_points: int = 90
    stem_search_min_cluster_points: Optional[int] = 300
    stem_search_min_cluster_height: Optional[float] = 1.3
    stem_search_min_cluster_intensity: Optional[float] = 6000
    stem_search_pc1_min_explained_variance: Optional[float] = None
    stem_search_max_stem_inclination: Optional[float] = None
    stem_search_refined_circle_fitting: bool = True
    stem_search_ellipse_fitting: bool = True
    stem_search_circle_fitting_layer_start: float = 1.0
    stem_search_circle_fitting_num_layers: int = 14
    stem_search_circle_fitting_layer_height: float = 0.125
    stem_search_circle_fitting_layer_overlap: float = 0.025
    stem_search_circle_fitting_min_points: int = 50
    stem_search_circle_fitting_small_buffer_width: float = 0.06
    stem_search_circle_fitting_large_buffer_width: float = 0.09
    stem_search_circle_fitting_switch_buffer_threshold: float = 0.3
    stem_search_ellipse_filter_threshold: float = 0.6
    stem_search_circle_fitting_max_std_diameter: float = 0.0185
    stem_search_circle_fitting_max_std_position: Optional[float] = None
    stem_search_circle_fitting_std_num_layers: int = 6
    stem_search_gam_max_radius_diff: Optional[float] = None


@dataclass
class TreeXPresetTLS(TreeXPreset):
    """
    Preset for the treeX algorithm for dense terrestrial point clouds (e.g., from stationary or mobile scanning).
    """


@dataclass
class TreeXPresetULS(TreeXPreset):  # pylint: disable=too-many-instance-attributes
    """
    Preset for the treeX algorithm for sparser UAV-borne point clouds.
    """

    # parameters for the identification of stem clusters
    stem_search_min_z: float = 1.0
    stem_search_max_z: float = 7.0
    stem_search_voxel_size: float = 0.03
    stem_search_dbscan_2d_eps: float = 0.05
    stem_search_dbscan_2d_min_points: int = 20
    stem_search_dbscan_3d_eps: float = 0.3
    stem_search_dbscan_3d_min_points: int = 3
    stem_search_min_cluster_points: Optional[int] = 80
    stem_search_min_cluster_height: float = 2.0
    stem_search_circle_fitting_num_layers: int = 5
    stem_search_circle_fitting_layer_height: float = 1.4
    stem_search_circle_fitting_layer_overlap: float = 0.3
    stem_search_circle_fitting_min_points = 3
    stem_search_circle_fitting_max_std_diameter: float = 0.05
    stem_search_circle_fitting_std_num_layers: int = 2
