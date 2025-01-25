""" Utilities for generating test data. """

__all__ = ["generate_circle_points", "generate_ellipse_points", "generate_grid_points"]

from typing import Union, Tuple

import numpy as np
import numpy.typing as npt


def generate_circle_points(  # pylint: disable=too-many-locals
    circles: npt.NDArray[np.float64],
    min_points: int,
    max_points: int,
    add_noise_points: bool = False,
    seed: int = 0,
    variance: Union[float, npt.NDArray[np.float64]] = 0,
) -> npt.NDArray[np.float64]:
    """
    Generates a set of 2D points that are randomly sampled around the outlines of the specified circles.

    Args:
        circles: Parameters of the circles from which to sample (in the following order: x-coordinate of the center,
            y-coordinate of the center, radius).
        min_points: Minimum number of points to sample from each circle.
        max_points: Maximum number of points to sample from each circle.
        add_noise_points: Whether randomly placed noise points not sampled from a circle should be added to the set
            of 2D points. Defaults to :code:`False`.
        seed: Random seed. Defaults to 0.
        variance: Variance of the distance of the sampled points to the circle outlines. Can be either a scalar
            value or an array of values whose length is equal to the length of :code:`circles`.

    Returns:
        X- and y-coordinates of the generated 2D points.

    Raises:
        ValueError: If :code:`variance` is an arrays whose length is not equal to the length of :code:`circles`.
    """
    xy = []
    random_generator = np.random.default_rng(seed=seed)

    if isinstance(variance, np.ndarray) and len(variance) != len(circles):
        raise ValueError("Length of variance must be equal to the number of circles.")

    for circle_idx, circle in enumerate(circles):
        num_points = int(random_generator.uniform(min_points, max_points))

        angles = np.linspace(0, 2 * np.pi, num_points)
        point_radii = np.full(num_points, fill_value=circle[2], dtype=np.float64)

        if isinstance(variance, (float, int)):
            current_variance = float(variance)
        else:
            current_variance = variance[circle_idx]

        point_radii += random_generator.normal(0, current_variance, num_points)

        x = point_radii * np.cos(angles)
        y = point_radii * np.sin(angles)
        xy.append(np.column_stack([x, y]) + circle[:2])

    if add_noise_points:
        num_points = int(random_generator.uniform(min_points * 0.1, max_points * 0.1))
        min_xy = (circles[:, :2] - circles[:, 2]).min(axis=0)
        max_xy = (circles[:, :2] + circles[:, 2]).max(axis=0)
        noise_points = random_generator.uniform(min_xy, max_xy, (num_points, 2))
        xy.append(noise_points)

    return np.concatenate(xy)


def generate_ellipse_points(  # pylint: disable=too-many-locals
    ellipses: npt.NDArray[np.float64],
    min_points: int,
    max_points: int,
    add_noise_points: bool = False,
    seed: int = 0,
    variance: Union[float, npt.NDArray[np.float64]] = 0,
) -> npt.NDArray[np.float64]:
    """
    Generates a set of 2D points that are randomly sampled around the outlines of the specified ellipses.

    Args:
        ellipses: Parameters of the ellipses from which to sample (in the following order: x-coordinate of the center,
            y-coordinate of the center, radius along the semi-major axis, radius along the semi-minor axis, and the
            counterclockwise angle of rotation from the x-axis to the semi-major axis of the ellipse).
        min_points: Minimum number of points to sample from each ellipse.
        max_points: Maximum number of points to sample from each ellipse.
        add_noise_points: Whether randomly placed noise points not sampled from a ellipse should be added to the set
            of 2D points. Defaults to :code:`False`.
        seed: Random seed. Defaults to 0.
        variance: Variance of the distance of the sampled points to the ellipse outlines. Can be either a scalar
            value or an array of values whose length is equal to the length of :code:`ellipses`.

    Returns:
        X- and y-coordinates of the generated 2D points.

    Raises:
        ValueError: If :code:`variance` is an arrays whose length is not equal to the length of :code:`ellipses`.
    """

    xy = []
    random_generator = np.random.default_rng(seed=seed)

    if isinstance(variance, np.ndarray) and len(variance) != len(ellipses):
        raise ValueError("Length of variance must be equal to the number of ellipses.")

    for ellipse_idx, ellipse in enumerate(ellipses):

        num_points = int(random_generator.uniform(min_points, max_points))

        # to equally distribute the sampled points over the outline of the ellipse, the sampling density is
        # based on the rate-of-change of the ellipse's arc length
        # see https://math.stackexchange.com/questions/3710402/generate-random-points-on-perimeter-of-ellipse
        theta_lookup = np.linspace(0, 2 * np.pi, num_points * 10)

        derivative_arc_len_angle = np.sqrt(
            ellipse[2] ** 2 * np.sin(theta_lookup) ** 2 + ellipse[3] ** 2 * np.cos(theta_lookup) ** 2
        )
        cumulative_distribution = (derivative_arc_len_angle).cumsum()
        cumulative_distribution = cumulative_distribution / cumulative_distribution[-1]

        theta = np.linspace(0, 1, num_points)
        lookup_indices = np.empty(len(theta), dtype=np.int64)

        for idx, theta_val in enumerate(theta):
            lookup_idx = min((theta_val >= cumulative_distribution).sum(), len(cumulative_distribution) - 1)
            lookup_indices[idx] = lookup_idx

        theta_corrected = theta_lookup[lookup_indices]

        point_major_radii = np.full(num_points, fill_value=ellipse[2], dtype=np.float64)
        point_minor_radii = np.full(num_points, fill_value=ellipse[3], dtype=np.float64)

        if isinstance(variance, (float, int)):
            current_variance = float(variance)
        else:
            current_variance = variance[ellipse_idx]

        point_major_radii += random_generator.normal(0, current_variance, num_points)
        point_minor_radii += random_generator.normal(0, current_variance, num_points)

        x = point_major_radii * np.cos(theta_corrected)
        y = point_minor_radii * np.sin(theta_corrected)

        rotation_matrix = np.array(
            [[np.cos(ellipse[4]), np.sin(ellipse[4])], [-np.sin(ellipse[4]), np.cos(ellipse[4])]]
        )
        current_xy = np.matmul(np.column_stack([x, y]), rotation_matrix)
        current_xy = current_xy + ellipse[:2]

        xy.append(current_xy)

    if add_noise_points:
        num_points = int(random_generator.uniform(min_points * 0.1, max_points * 0.1))
        min_xy = (ellipses[:, :2] - ellipses[:, 2]).min(axis=0)
        max_xy = (ellipses[:, :2] + ellipses[:, 2]).max(axis=0)
        noise_points = random_generator.uniform(min_xy, max_xy, (num_points, 2))
        xy.append(noise_points)

    return np.concatenate(xy)


def generate_grid_points(num_points: Tuple[int, ...], point_spacing: float) -> npt.NDArray[np.float64]:
    """
    Generates a set of 2D points that are regularly spaced.

    Args:
        num_points: Number of points to generate along each coordinate dimension.
        point_spacing: Spacing of the points to be generated.

    Returns:
        X- and y-coordinates of the generated 2D points.
    """

    coord_dims = []

    for n_dim in num_points:
        coord_dims.append(np.arange(n_dim).astype(np.float64) * point_spacing)

    mesh_coords = np.meshgrid(*coord_dims)

    return np.column_stack([x.flatten() for x in mesh_coords])
