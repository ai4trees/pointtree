"""Cloth simulation filtering."""

__all__ = ["cloth_simulation_filtering"]

import CSF
import numpy as np

from pointtree.type_aliases import FloatArray, LongArray


def cloth_simulation_filtering(  # pylint: disable=too-few-public-methods
    coords: FloatArray,
    classification_threshold: float,
    resolution: float,
    rigidness: int,
    correct_steep_slope: bool = False,
    iterations: int = 100,
) -> LongArray:
    r"""
    Detects ground points using the Cloth Simulation Filtering (CSF) algorithm proposed in `Zhang, Wuming, et al. \
    "An Easy-to-Use Airborne LiDAR Data Filtering Method Based on Cloth Simulation." Remote Sensing 8.6 (2016): 501
    <https://doi.org/10.3390/rs8060501>`__.

    Args:
        coords: Point coordinates.
        classification_threshold: Maximum height above the cloth a point can have in order to be classified as
            terrain point. All points whose distance to the cloth is equal or below this threshold are
            classified as terrain points.
        resolution: Resolution of the cloth grid (in meter).
        rigidness: Rigidness of the cloth (the three levels 1, 2, and 3 are available, where 1 is
            the lowest and 3 the highest rigidness).
        correct_steep_slope: Whether the cloth should be corrected for steep slopes in a post-pressing step.
        iterations: Maximum number of iterations.

    Returns:
        Class IDs for each point. For terrain points, the class ID is set to 0 and for non-terrain points to 1.

    Raises:
        ValueError: If :code:`rigidness` is not 1, 2, or 3.

    Shape:
        - :code:`coords`: :math:`(N, 3)`
        - Output: :math:`(N)`.

        | where
        |
        | :math:`N = \text{ number of points}`
    """

    if rigidness < 1 or rigidness > 3:
        raise ValueError(f"Invalid rigidness value: {rigidness}")

    csf = CSF.CSF()
    csf.params.class_threshold = classification_threshold
    csf.params.bSloopSmooth = correct_steep_slope
    csf.params.interations = iterations
    csf.params.cloth_resolution = resolution
    csf.params.rigidness = rigidness
    csf.params.time_step = 1

    csf.setPointCloud(coords)
    ground_indices = CSF.VecInt()
    non_ground_indices = CSF.VecInt()
    csf.do_filtering(ground_indices, non_ground_indices, exportCloth=False)

    classification = np.full(len(coords), dtype=np.int64, fill_value=-1)
    classification[ground_indices] = 0
    classification[non_ground_indices] = 1

    return classification
