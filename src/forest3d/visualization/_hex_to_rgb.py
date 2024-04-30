""" Color conversion. """

__all__ = ["hex_to_rgb"]

from typing import Tuple

import numpy as np


def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """
    Converts hexadecimal color code to RGB color value.

    Args:
        hex: Hexadecimal color code.

    Returns:
        RGBA color value.
    """

    rgb_color = []
    if hex_color.startswith("#"):
        hex_color = hex_color.lstrip("#")
    for i in (0, 2, 4):
        decimal = int(hex_color[i : i + 2], 16)
        rgb_color.append(decimal)
    rgb_color.append(255)

    return tuple((np.array(rgb_color).astype(float) / 255))
