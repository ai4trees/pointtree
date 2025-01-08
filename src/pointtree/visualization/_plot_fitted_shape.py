""" Plots a shape fitted to a set of 2D points. """

__all__ = ["plot_fitted_shape"]

import gc

from pathlib import Path
from typing import Optional, Union


import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches
import numpy as np
import numpy.typing as npt


def plot_fitted_shape(  # pylint: disable=too-many-locals
    xy: npt.NDArray[np.float64],
    file_path: Union[str, Path],
    circle: Optional[npt.NDArray[np.float64]] = None,
    ellipse: Optional[npt.NDArray[np.float64]] = None,
    polygon: Optional[npt.NDArray[np.float64]] = None,
) -> None:
    r"""
    Plots a shape fitted to a set of 2D points. The shape to be plotted can either be a circle, an ellipse or polygon.

    Args:
        xy: Coordinates of the points to which the circle was fitted.
        file_path: File path under which the image is to be saved.
        circle: Parameters of the circle in the following order: x-coordinate of the center, y-coordinate of the center,
            radius. Defaults to :code:`None`, which means that no circle is plotted.
        ellipse: Parameters of the ellipse in the following order: x- and y-coordinates of the center, radius along the
            semi-major, radius along the semi-minor axis, and the counterclockwise angle of rotation from the x-axis to
            the semi-major axis of the ellipse.
        polygon: Sorted vertices (x- and y-coordinates) of the polygon. Defaults to :code:`None`, which means that no
            polygon is plotted.

    Shape:
        - :code:`xy`: :math:`(N, 2)`
        - :code:`circle`: :math:`(3)`
        - :code:`ellipse`: :math:`(5)`
        - :code:`polygon`: :math:`(V)`

        | where
        |
        | :math:`N = \text{ number of points}`
        | :math:`V = \text{ number of polygon vertices}`
    """

    plt.close("all")
    matplotlib.use("agg")
    plt.ioff()

    plt.clf()
    plt.plot(xy[:, 0], xy[:, 1], "o", color="black", markersize=2, zorder=1)

    axis = plt.gca()

    fig_width_x = xy[:, 0].max() - xy[:, 0].min()
    fig_width_y = xy[:, 1].max() - xy[:, 1].min()
    fig_width = max(fig_width_x, fig_width_y)
    fig_center_x = xy[:, 0].min() + fig_width_x / 2
    fig_center_y = xy[:, 1].min() + fig_width_y / 2
    padding = max(0.2 * fig_width, 0.1)

    axis.set_xlim((fig_center_x - fig_width / 2 - padding, fig_center_x + fig_width / 2 + padding))
    axis.set_ylim((fig_center_y - fig_width / 2 - padding, fig_center_y + fig_width / 2 + padding))

    if circle is not None:
        if len(circle) != 3:
            raise ValueError("A circle must contain three parameters.")

        center = circle[:2]
        radius = circle[2]
        circle_patch = plt.Circle(center, radius, color="red", linewidth=3, fill=False, zorder=2)
        axis.add_patch(circle_patch)

    if ellipse is not None:
        if len(ellipse) != 5:
            raise ValueError("An ellipse must contain five parameters.")

        center = ellipse[:2]
        width, height, angle = ellipse[2:]
        angle = angle / (2 * np.pi) * 365
        ellipse_patch = matplotlib.patches.Ellipse(
            (center[0], center[1]), width * 2, height * 2, angle=angle, color="red", linewidth=3, fill=False, zorder=2
        )
        axis.add_patch(ellipse_patch)

    if polygon is not None:
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise ValueError("A polygon must have shape (V, 2).")
        line_closed_xy = np.concatenate((polygon, polygon[:1]))
        plt.plot(line_closed_xy[:, 0], line_closed_xy[:, 1], "-", color="red", lw=3, zorder=2)

    fig = plt.gcf()
    fig.set_size_inches(5, 5)

    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    file_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(file_path)
    plt.close("all")
    gc.collect()
