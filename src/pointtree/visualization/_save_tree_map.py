""" Visualization of 2D maps. """

__all__ = ["save_tree_map"]

import os
from typing import Optional

import numpy as np
from PIL import Image

from ._color_palette import acm_red, color_palette


def save_tree_map(  # pylint: disable=too-many-branches
    image: np.ndarray,
    output_path: str,
    *,
    is_label_image: bool = False,
    crown_borders: Optional[np.ndarray] = None,
    border_mask: Optional[np.ndarray] = None,
    seed_mask: Optional[np.ndarray] = None,
    core_mask: Optional[np.ndarray] = None,
    tree_positions: Optional[np.ndarray] = None,
    crown_positions: Optional[np.ndarray] = None,
    trunk_positions: Optional[np.ndarray] = None,
) -> None:
    """
    Saves a 2D map showing canopy height models, tree positions etc. as image file.
    """

    if is_label_image:
        color_image = np.full((image.shape[0], image.shape[1], 3), fill_value=255, dtype=np.uint8)
        for instance_id in np.unique(image):
            if instance_id <= 0:
                continue
            color = color_palette[instance_id % len(color_palette)]
            color_image[image == instance_id, :] = color[:3]
        image = color_image
    else:
        image = image.astype(float)
        image -= image.min()
        image /= image.max()
        image *= 255
        image = 255 - image
        image = np.expand_dims(image, axis=-1)
        image = np.tile(image, (1, 1, 3))
        image = image.astype(np.uint8)

    if core_mask is not None:
        image[core_mask, 0] = 0
        image[core_mask, 1] = 127
        image[core_mask, 2] = 95

    if border_mask is not None:
        image[border_mask, 0] = 252
        image[border_mask, 1] = 146
        image[border_mask, 2] = 0

    if seed_mask is not None:
        image[seed_mask, 0] = 170
        image[seed_mask, 1] = 204
        image[seed_mask, 2] = 0

    if crown_borders is not None:
        image[crown_borders, 0] = 252
        image[crown_borders, 1] = 146
        image[crown_borders, 2] = 0

    pil_image = Image.fromarray(image).convert("RGB")

    if trunk_positions is not None:
        for coord in trunk_positions:
            pil_image.putpixel((coord[1], coord[0]), acm_red)

    if crown_positions is not None:
        for coord in crown_positions:
            pil_image.putpixel((coord[1], coord[0]), acm_red)

    if tree_positions is not None:
        for coord in tree_positions:
            pil_image.putpixel((coord[1], coord[0]), acm_red)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pil_image.save(output_path)
