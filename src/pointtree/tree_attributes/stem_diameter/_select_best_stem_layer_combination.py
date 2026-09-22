"""Selection of the most consistent combination of horizontal stem layers."""

__all__ = ["select_best_stem_layer_combination"]

import itertools
from typing import Optional, Tuple

import numpy as np

from pointtree.type_aliases import FloatArray, LongArray


def select_best_stem_layer_combination(
    existing_layers: LongArray,
    diameters: FloatArray,
    combination_size: int,
    max_std_diameter: float,
    positions: Optional[FloatArray] = None,
    max_std_position: float = np.inf,
) -> Tuple[Optional[LongArray], float]:
    r"""
    Selects the combination of :code:`combination_size` horizontal stem layers with the lowest standard deviation of
    the fitted circle or ellipse diameters, among all combinations whose diameter standard deviation does not exceed
    :code:`max_std_diameter`. If :code:`positions` is provided, a combination is additionally only considered valid
    if the standard deviation of the x- and y-coordinates of the circle / ellipse centers within that combination
    does not exceed :code:`max_std_position`.

    Args:
        existing_layers: Indices of the layers for which a circle or ellipse was fitted successfully.
        diameters: Diameter of the fitted circle or ellipse for each layer (including layers without a valid fit,
            which must not be referenced by :code:`existing_layers`).
        combination_size: Number of layers to select.
        max_std_diameter: Maximum standard deviation of the diameters within a combination for that combination to
            be considered valid.
        positions: X- and y-coordinates of the circle / ellipse center for each layer. If :code:`None`, the standard
            deviation of the layer positions is not used for filtering.
        max_std_position: Maximum standard deviation of the x- and y-coordinates of the circle / ellipse centers
            within a combination for that combination to be considered valid. Only used if :code:`positions` is not
            :code:`None`.

    Returns:
        :Tuple of two elements:
            - Indices of the layers belonging to the combination with the lowest diameter standard deviation among
              the valid combinations. :code:`None` if :code:`existing_layers` contains fewer than
              :code:`combination_size` layers or if no valid combination was found.
            - Standard deviation of the diameters within the selected combination. :code:`NaN` if no valid
              combination was found.

    Shape:
        - :code:`existing_layers`: :math:`(L')`
        - :code:`diameters`: :math:`(L)`
        - :code:`positions`: :math:`(L, 2)`
        - Output: :math:`(C)` and scalar

        | where
        |
        | :math:`C` = Number of layers to select
        | :math:`L` = number of horizontal layers
        | :math:`L'` = number of layers with a valid circle / ellipse fit
    """

    if len(existing_layers) < combination_size:
        return None, float("nan")

    best_combination = None
    best_std = np.inf

    for combination in itertools.combinations(existing_layers, combination_size):
        combination_array = np.asarray(combination, dtype=np.int64)

        diameter_std = float(np.std(diameters[combination_array]))
        if diameter_std > max_std_diameter:
            continue

        if positions is not None:
            position_std = np.std(positions[combination_array], axis=0)
            if (position_std > max_std_position).any():
                continue

        if diameter_std < best_std:
            best_std = diameter_std
            best_combination = combination_array

    if best_combination is None:
        return None, float("nan")

    return best_combination, best_std
