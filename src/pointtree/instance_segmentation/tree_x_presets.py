"""Presets for the treeX algorithm."""

__all__ = ["TreeXPreset", "TreeXPresetTLS", "TreeXPresetULS"]

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
class TreeXPresetTLS(TreeXPreset):
    """
    Preset for the treeX algorithm for dense terrestrial point clouds (e.g., from stationary or mobile scanning).
    """


@dataclass
class TreeXPresetULS(TreeXPreset):  # pylint: disable=too-many-instance-attributes
    """
    Preset for the treeX algorithm for sparser UAV-borne point clouds.
    """

    # parameters for the identification of trunk clusters
    trunk_search_min_z: float = 1.0
    trunk_search_max_z: float = 7.0
    trunk_search_voxel_size: float = 0.03
    trunk_search_dbscan_2d_eps: float = 0.05
    trunk_search_dbscan_2d_min_points: int = 20
    trunk_search_dbscan_3d_eps: float = 0.3
    trunk_search_dbscan_3d_min_points: int = 3
    trunk_search_min_cluster_points: Optional[int] = 80
    trunk_search_circle_fitting_num_layers: int = 5
    trunk_search_circle_fitting_layer_height: float = 1.4
    trunk_search_circle_fitting_layer_overlap: float = 0.3
    trunk_search_circle_fitting_min_points = 3
    trunk_search_circle_fitting_max_std_diameter: float = 0.05
    trunk_search_circle_fitting_std_num_layers: int = 2
    trunk_search_min_cluster_height: float = 2.0

    # region growing parameters
    crown_seg_voxel_size: float = 0.05
