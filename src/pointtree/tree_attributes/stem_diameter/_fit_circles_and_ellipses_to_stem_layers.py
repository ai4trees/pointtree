"""Circle and ellipse fitting for horizontal stem layers."""

__all__ = ["fit_circles_and_ellipses_to_stem_layers"]

from typing import Literal, Optional, Tuple, Union

from circle_detection import MEstimator, Ransac
import numpy as np

from pointtree.type_aliases import FloatArray, LongArray

from ...operations._fit_ellipse import fit_ellipse


def fit_circles_and_ellipses_to_stem_layers(  # pylint: disable=too-many-locals, too-many-arguments
    layer_xy: FloatArray,
    batch_lengths: LongArray,
    *,
    circle_fitting_method: Literal["ransac", "m-estimator"] = "ransac",
    bandwidth: float = 0.01,
    min_fitting_score: float = 100.0,
    min_stem_diameter: float = 0.02,
    max_stem_diameter: float = 1.0,
    min_completeness_idx: Optional[float] = 0.3,
    fit_ellipses: bool = False,
    ellipse_filter_threshold: float = 0.6,
    num_workers: int = 1,
    seed: Optional[int] = None,
) -> Tuple[FloatArray, FloatArray]:
    r"""
    Fits a circle and, optionally, an ellipse to each of a set of horizontal stem layers.

    Args:
        layer_xy: X- and y-coordinates of the points of all layers. Points belonging to the same layer must be
            stored consecutively, with the number of points belonging to each layer given by :code:`batch_lengths`.
        batch_lengths: Number of points belonging to each layer.
        circle_fitting_method: Circle fitting method to use: :code:`"m-estimator"` | :code:`"ransac"`.
        bandwidth: Bandwidth for circle fitting. It is used in the calculation of the circle fitting score and, for
            the M-estimator method, also for kernel density estimation.
        min_fitting_score: Minimum fitting score that circles must achieve in the circle fitting to be considered
            valid. Only used when :code:`circle_fitting_method` is set to :code:`"ransac"`.
        min_stem_diameter: Minimum circle / ellipse diameter to be considered a valid fit.
        max_stem_diameter: Maximum circle / ellipse diameter to be considered a valid fit.
        min_completeness_idx: Minimum circumferential completeness index that circles must have to be considered
            valid. If :code:`None`, the circumferential completeness index is not used for filtering.
        fit_ellipses: Whether an ellipse should additionally be fitted to each layer.
        ellipse_filter_threshold: Ellipses are only kept if the ratio of the radius along the semi-minor axis to the
            radius along the semi-major axis is greater than or equal to this threshold. Only used when
            :code:`fit_ellipses` is set to :code:`True`.
        num_workers: Number of worker threads to use for the circle fitting. If set to -1, all CPU threads are used.
        seed: Random seed for the circle fitting. If :code:`None`, the fitting is not seeded.

    Returns:
        :Tuple of two arrays:
            - Parameters of the circles that were fitted to the layers. Each circle is represented by three values,
              namely the x- and y-coordinates of its center and its radius. If the circle fitting is not successful
              for a layer, all parameters are set to :code:`-1`.
            - Parameters of the ellipses that were fitted to the layers. Each ellipse is represented by five values,
              namely the x- and y-coordinates of its center, its radius along the semi-major and along the
              semi-minor axis, and the counterclockwise angle of rotation from the x-axis to the semi-major axis of
              the ellipse. If :code:`fit_ellipses` is :code:`False`, the ellipse fitting was not successful for a
              layer, or the ellipse's axis ratio is smaller than :code:`ellipse_filter_threshold`, all parameters
              are set to :code:`-1`.

    Shape:
        - :code:`layer_xy`: :math:`(N, 2)`
        - :code:`batch_lengths`: :math:`(L)`
        - Output: :math:`(L, 3)`, :math:`(L, 5)`

        | where
        |
        | :math:`N` = number of points
        | :math:`L` = number of layers
    """

    num_layers = len(batch_lengths)

    layer_circles = np.full((num_layers, 3), fill_value=-1, dtype=layer_xy.dtype)
    layer_ellipses = np.full((num_layers, 5), fill_value=-1, dtype=layer_xy.dtype)

    if not layer_xy.flags.f_contiguous:
        layer_xy = layer_xy.copy(order="F")

    min_radius = min_stem_diameter / 2
    max_radius = max_stem_diameter / 2
    min_start_radius = min_radius + min(2 * bandwidth, (max_radius - min_radius) / 4)
    max_start_radius = max_radius - min(2 * bandwidth, (max_radius - min_radius) / 4)

    circle_detector: Union[MEstimator, Ransac]
    if circle_fitting_method == "m-estimator":
        circle_detector = MEstimator(
            bandwidth=bandwidth,
            break_min_change=1e-6,
            min_step_size=1e-10,
            max_iterations=300,
            armijo_min_decrease_percentage=0.5,
            armijo_attenuation_factor=0.25,
        )
        circle_detector.detect(
            layer_xy,
            batch_lengths=batch_lengths,
            n_start_x=3,
            n_start_y=3,
            min_start_radius=min_start_radius,
            max_start_radius=max_start_radius,
            break_min_radius=min_radius,
            break_max_radius=max_radius,
            n_start_radius=3,
            num_workers=num_workers,
        )
    else:
        circle_detector = Ransac(bandwidth=bandwidth, min_fitting_score=min_fitting_score)
        circle_detector.detect(
            layer_xy,
            batch_lengths=batch_lengths,
            break_min_radius=min_radius,
            break_max_radius=max_radius,
            num_workers=num_workers,
            seed=seed,
        )

    circle_detector.filter(
        max_circles=1,
        deduplication_precision=4,
        min_circumferential_completeness_idx=min_completeness_idx,
        circumferential_completeness_idx_max_dist=bandwidth,
        circumferential_completeness_idx_num_regions=int(365 / 5),
        non_maximum_suppression=True,
        num_workers=num_workers,
    )

    ellipses = None
    if fit_ellipses:
        ellipses = fit_ellipse(layer_xy, batch_lengths)

    batch_starts_circles = np.cumsum(
        np.concatenate((np.array([0], dtype=np.int64), circle_detector.batch_lengths_circles))
    )[:-1]

    for layer in range(num_layers):
        if batch_lengths[layer] == 0:
            continue

        if circle_detector.batch_lengths_circles[layer] > 0:
            layer_circles[layer] = circle_detector.circles[batch_starts_circles[layer]]

        if ellipses is not None and ellipses[layer, 2] != -1:
            radius_major, radius_minor = ellipses[layer, 2:4]
            if radius_minor / radius_major >= ellipse_filter_threshold:
                layer_ellipses[layer] = ellipses[layer]

    return layer_circles, layer_ellipses
