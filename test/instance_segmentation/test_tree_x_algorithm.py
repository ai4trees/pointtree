""" Tests for pointtree.instance_segmentation.TreeXAlgorithm. """

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import shutil

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from pointtree.instance_segmentation import TreeXAlgorithm


class TestTreeXAlgorithm:
    """Tests for pointtree.instance_segmentation.TreeXAlgorithm."""

    @pytest.fixture
    def cache_dir(self):
        cache_dir = "./tmp/test/io/TestTreeXAlgorithm"
        os.makedirs(cache_dir, exist_ok=True)
        yield cache_dir
        shutil.rmtree(cache_dir)

    @staticmethod
    def _generate_circle_points(  # pylint: disable=too-many-locals
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
                value or an array of values whose length is equal to :code:`num_circles`.

        Returns:
            Tuple of two arrays. The first contains the parameters of the generated circles (in the order x, y and
            radius). The second contains the x- and y-coordinates of the generated 2D points.

        Raises:
            ValueError: If :code:`variance` is an arrays whose length is not equal to :code:`circles`.
        """
        xy = []
        random_generator = np.random.default_rng(seed=seed)

        if isinstance(variance, np.ndarray) and len(variance) != len(circles):
            raise ValueError("Length of variance must be equal to num_circles.")

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
            min_xy = (circles[:, :2] - circles[:, 2]).min(axis=0).min()
            max_xy = (circles[:, :2] + circles[:, 2]).max(axis=0).max()
            noise_points = random_generator.uniform(min_xy, max_xy, (num_points, 2))
            xy.append(noise_points)

        return np.concatenate(xy)

    @staticmethod
    def _generate_ellipse_points(  # pylint: disable=too-many-locals
        ellipses: npt.NDArray[np.float64],
        min_points: int,
        max_points: int,
        add_noise_points: bool = False,
        seed: int = 0,
        variance: Union[float, npt.NDArray[np.float64]] = 0,
    ) -> npt.NDArray[np.float64]:
        xy = []
        random_generator = np.random.default_rng(seed=seed)

        if isinstance(variance, np.ndarray) and len(variance) != len(ellipses):
            raise ValueError("Length of variance must be equal to num_circles.")

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

            current_xy = np.column_stack([x, y]) + ellipse[:2]
            rotation_matrix = np.array(
                [[np.cos(ellipse[4]), np.sin(ellipse[4])], [-np.sin(ellipse[4]), np.cos(ellipse[4])]]
            )
            current_xy = np.matmul(current_xy, rotation_matrix)

            xy.append(current_xy)

        if add_noise_points:
            num_points = int(random_generator.uniform(min_points * 0.1, max_points * 0.1))
            min_xy = (ellipses[:, :2] - ellipses[:, 2]).min(axis=0).min()
            max_xy = (ellipses[:, :2] + ellipses[:, 2]).max(axis=0).max()
            noise_points = random_generator.uniform(min_xy, max_xy, (num_points, 2))
            xy.append(noise_points)

        return np.concatenate(xy)

    @staticmethod
    def _generate_grid_points(num_points: Tuple[int, ...], point_spacing: float) -> npt.NDArray[np.float64]:

        coord_dims = []

        for n_dim in num_points:
            coord_dims.append(np.arange(n_dim).astype(np.float64) * point_spacing)

        mesh_coords = np.meshgrid(*coord_dims)

        return np.column_stack([x.flatten() for x in mesh_coords])

    def generate_layer_circles_and_ellipses(  # pylint: disable=too-many-locals
        self, add_noise_points: bool, variance: float
    ) -> Tuple[
        Dict[str, Any],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """
        Generates inputs and expected values for testing the methods
        :code:`TreeXAlgorithm.fit_preliminary_circles_or_ellipses_to_trunks` and
        :code:`TreeXAlgorithm.fit_exact_circles_and_ellipses_to_trunks`.

        Args:
            add_noise_points: Whether randomly placed noise points not sampled from a circle / ellipse should be added
                to the generated  input points.
            variance: Variance of the distance of the generated points to the circle / ellipse outlines.

        Returns:
            Tuple of six values. The first is a dictionary containing the keyword arguments to be passed to the
            :code:`TreeXAlgorithm` class to produce the expected results returned by this method. The second is an array
            containing the input point coordinates to be passed to the
            :code:`TreeXAlgorithm.fit_preliminary_circles_or_ellipses_to_trunks` method. The third is an array
            containing IDs indicating which points belong to which trunk cluster. The fourth is an array containing the
            unique cluster IDs. The fifth contains the true parameters of the circles / ellipses from which the input
            points were sampled. The sixth contains the heights of the horizontal layers at which points were generated.
        """

        trunk_search_min_z = 1.0
        layer_height = 0.1
        layer_overlap = 0.02
        num_trees = 2
        num_layers = 5
        min_points = 50
        skip_layers = [2]
        generate_valid_ellipses = [0]
        generate_invalid_ellipses = [4]

        settings = {
            "trunk_search_min_z": trunk_search_min_z,
            "trunk_search_circle_fitting_num_layers": num_layers,
            "trunk_search_circle_fitting_layer_height": layer_height + layer_overlap,
            "trunk_search_circle_fitting_layer_overlap": layer_overlap,
            "trunk_search_circle_fitting_min_points": min_points,
            "trunk_search_circle_fitting_min_trunk_diameter": 0.02,
            "trunk_search_circle_fitting_max_trunk_diameter": 1,
            "trunk_search_circle_fitting_min_completeness_idx": 0.6,
        }

        circle_or_ellipse_points: List[float] = []
        cluster_labels: List[npt.NDArray[np.int64]] = []
        expected_circles_or_ellipses = np.full((num_trees, num_layers, 5), fill_value=-1, dtype=np.float64)
        expected_layer_heigths = np.empty(num_layers, dtype=np.float64)

        for tree in range(num_trees):
            offset = tree * 1.5
            for layer_idx in range(num_layers):
                current_height = trunk_search_min_z + layer_height * layer_idx + layer_height / 2 + layer_overlap / 2
                expected_layer_heigths[layer_idx] = current_height
                if layer_idx not in skip_layers:
                    if layer_idx in generate_valid_ellipses or layer_idx in generate_invalid_ellipses:
                        if layer_idx in generate_invalid_ellipses:
                            ellipses = np.array([[0, 0, 0.8, 0.2, np.pi / 4]], dtype=np.float64)
                        else:
                            ellipses = np.array([[0, 0, 0.6, 0.4, np.pi / 4]], dtype=np.float64)
                        points_2d = self._generate_ellipse_points(
                            ellipses,
                            min_points=10 * min_points,
                            max_points=20 * min_points,
                            seed=layer_idx,
                            add_noise_points=False,
                            variance=variance,
                        )
                        expected_circles_or_ellipses[tree, layer_idx] = ellipses[0]
                    else:
                        circles = np.array([[0, 0, 0.5 - 0.05 * layer_idx]], dtype=np.float64)
                        points_2d = self._generate_circle_points(
                            circles,
                            min_points=10 * min_points,
                            max_points=20 * min_points,
                            seed=layer_idx,
                            add_noise_points=add_noise_points,
                            variance=variance,
                        )
                        expected_circles_or_ellipses[tree, layer_idx, :3] = circles[0]
                    expected_circles_or_ellipses[tree, layer_idx, :2] += offset

                points_3d = np.empty(
                    (len(points_2d), 3), dtype=np.float64  # pylint: disable=possibly-used-before-assignment
                )
                points_3d[:, :2] = points_2d + offset
                points_3d[:, 2] = current_height
                if layer_idx in skip_layers:
                    points_3d = points_3d[: int(min_points / 2)]
                cluster_labels.extend(np.full(len(points_3d), fill_value=tree, dtype=np.int64))
                circle_or_ellipse_points.extend(points_3d)

        trunk_layer_xyz = np.array(circle_or_ellipse_points)
        cluster_labels_np = np.array(cluster_labels)
        unique_cluster_labels = np.unique(cluster_labels)

        return (
            settings,
            trunk_layer_xyz,
            cluster_labels_np,
            unique_cluster_labels,
            expected_circles_or_ellipses,
            expected_layer_heigths,
        )

    @pytest.mark.parametrize("variance, add_noise_points", [(0.0, False), (0.01, True)])
    @pytest.mark.parametrize("create_visualization, use_pathlib", [(False, False), (True, False), (True, True)])
    def test_fit_circles_or_ellipses_to_trunks(  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        self, variance: float, add_noise_points: bool, create_visualization: bool, use_pathlib: bool, cache_dir
    ):
        (
            settings,
            trunk_layer_xyz,
            cluster_labels,
            unique_cluster_labels,
            expected_circles_or_ellipses,
            expected_layer_heigths,
        ) = self.generate_layer_circles_and_ellipses(add_noise_points, variance)

        num_trees = len(expected_circles_or_ellipses)
        num_layers = expected_circles_or_ellipses.shape[1]

        visualization_folder = None
        point_cloud_id = None
        if create_visualization:
            point_cloud_id = "test"
            visualization_folder = cache_dir
            if use_pathlib:
                visualization_folder = Path(visualization_folder)

        trunk_search_ellipse_filter_threshold = 0.6

        algorithm = TreeXAlgorithm(
            visualization_folder=visualization_folder,
            trunk_search_ellipse_filter_threshold=trunk_search_ellipse_filter_threshold,
            **settings,
        )

        preliminary_circles_or_ellipses = algorithm.fit_preliminary_circles_or_ellipses_to_trunks(
            trunk_layer_xyz, cluster_labels, unique_cluster_labels, point_cloud_id=point_cloud_id
        )

        decimal = 1 if variance > 0 else 5

        np.testing.assert_almost_equal(expected_circles_or_ellipses, preliminary_circles_or_ellipses, decimal=decimal)

        if create_visualization:
            if not use_pathlib:
                visualization_folder = Path(visualization_folder)  # type: ignore[arg-type]
            visualization_folder = cast(Path, visualization_folder)
            for tree in range(num_trees):
                for layer in range(num_layers):
                    if expected_circles_or_ellipses[tree, layer, 4] == -1:
                        continue
                    if expected_circles_or_ellipses[tree, layer, 3] == -1:
                        expected_visualization_path = (
                            visualization_folder / "test" / f"preliminary_circle_trunk_{tree}_layer_{layer}.png"
                        )
                    else:
                        expected_visualization_path = (
                            visualization_folder / "test" / f"preliminary_ellipse_trunk_{tree}_layer_{layer}.png"
                        )

                    assert expected_visualization_path.exists()

        (
            layer_circles,
            layer_ellipses,
            layer_heights,
            trunk_layers_xy,
        ) = algorithm.fit_exact_circles_and_ellipses_to_trunks(
            trunk_layer_xyz, preliminary_circles_or_ellipses, point_cloud_id=point_cloud_id
        )

        if create_visualization:
            if not use_pathlib:
                visualization_folder = Path(visualization_folder)  # type: ignore[arg-type]
            visualization_folder = cast(Path, visualization_folder)
            for tree in range(num_trees):
                for layer in range(num_layers):
                    if expected_circles_or_ellipses[tree, layer, 4] == -1:
                        continue
                    if expected_circles_or_ellipses[tree, layer, 3] == -1:
                        expected_visualization_path = (
                            visualization_folder / "test" / f"exact_circle_trunk_{tree}_layer_{layer}.png"
                        )
                    else:
                        expected_visualization_path = (
                            visualization_folder / "test" / f"exact_ellipse_trunk_{tree}_layer_{layer}.png"
                        )

                    assert expected_visualization_path.exists()

        np.testing.assert_array_equal(expected_layer_heigths, layer_heights)

        assert len(layer_circles) == num_trees
        assert len(layer_ellipses) == num_trees
        assert layer_circles.shape[1] == num_layers
        assert layer_ellipses.shape[1] == num_layers
        assert len(trunk_layers_xy) == num_trees
        assert len(trunk_layers_xy[0]) == num_layers

        for tree in range(num_trees):
            for layer in range(num_layers):
                ellipse_radius_ratio = (
                    expected_circles_or_ellipses[tree, layer, 3] / expected_circles_or_ellipses[tree, layer, 2]
                )
                is_skip_layer = expected_circles_or_ellipses[tree, layer, 2] == -1
                is_invalid_ellipse_layer = (
                    expected_circles_or_ellipses[tree, layer, 3] != -1
                    and ellipse_radius_ratio < trunk_search_ellipse_filter_threshold
                )
                if is_skip_layer or is_invalid_ellipse_layer:
                    assert (layer_circles[tree, layer] == -1).all()
                    assert (layer_ellipses[tree, layer] == -1).all()
                    continue
                if expected_circles_or_ellipses[tree, layer, 3] != -1:
                    np.testing.assert_almost_equal(
                        expected_circles_or_ellipses[tree, layer], layer_ellipses[tree, layer], decimal=decimal
                    )
                    assert (layer_circles[tree, layer] == -1).all()
                    continue

                np.testing.assert_almost_equal(
                    expected_circles_or_ellipses[tree, layer, :3], layer_circles[tree, layer], decimal=decimal
                )
                assert (layer_ellipses[tree, layer, 2:4] > 0).all()

    def test_filter_instances_trunk_layer_variance(self):
        algorithm = TreeXAlgorithm(
            trunk_search_circle_fitting_std_num_layers=3, trunk_search_circle_fitting_max_std=0.1
        )

        layer_circles = np.array(
            [
                [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
                [[0, 0, 2], [0, 0, 0.95], [0, 0, 1.4], [0, 0.1, 0.1]],
                [[-1, -1, -1], [0, 0, 1], [-1, -1, -1], [-1, -1, -1]],
                [[0, 0, 1], [0, 0, 0.95], [0, 0, 1.2], [0, 0.1, 0.9]],
            ],
            dtype=np.float64,
        )
        layer_ellipses = np.array(
            [
                [[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]],
                [[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]],
                [[0, 0, 1, 0.9, 0], [0, 0, 0.95, 0.9, 0], [0, 0, 0.9, 0.85, 0], [0, 0, 1.2, 0.85, 0]],
                [[0, 0, 1, 0.9, 0], [0, 0, 1, 0.9, 0], [0, 0, 1.2, 1.1, 0], [0, 0.1, 0.9, 0.8, 0]],
            ],
            dtype=np.float64,
        )

        expected_filter_mask = np.array([False, False, True, True])
        expected_best_circle_combinations = np.array([[-1, -1, -1], [0, 1, 3]], dtype=np.int64)
        expected_best_ellipse_combinations = np.array([[0, 1, 2], [-1, -1, -1]], dtype=np.int64)

        filter_mask, best_circle_combination, best_ellipse_combination = algorithm.filter_instances_trunk_layer_std(
            layer_circles, layer_ellipses
        )

        np.testing.assert_array_equal(expected_filter_mask, filter_mask)
        np.testing.assert_array_equal(expected_best_circle_combinations, np.sort(best_circle_combination, axis=-1))
        np.testing.assert_array_equal(expected_best_ellipse_combinations, np.sort(best_ellipse_combination, axis=-1))

    def test_rename_visualizations_after_filtering(self, cache_dir):
        cache_dir = Path(cache_dir)
        algorithm = TreeXAlgorithm(trunk_search_circle_fitting_num_layers=2, visualization_folder=cache_dir)
        filter_mask = np.array([False, True])
        point_cloud_id = "test"

        paths = [
            (
                cache_dir / point_cloud_id / "preliminary_ellipse_trunk_0_layer_1.png",
                cache_dir / point_cloud_id / "preliminary_ellipse_trunk_1_layer_1_invalid.png",
            ),
            (
                cache_dir / point_cloud_id / "preliminary_circle_trunk_1_layer_0.png",
                cache_dir / point_cloud_id / "preliminary_circle_trunk_0_layer_0_valid.png",
            ),
            (
                cache_dir / point_cloud_id / "preliminary_circle_trunk_1_layer_1.png",
                cache_dir / point_cloud_id / "preliminary_circle_trunk_0_layer_1_valid.png",
            ),
            (
                cache_dir / point_cloud_id / "exact_ellipse_trunk_0_layer_1.png",
                cache_dir / point_cloud_id / "exact_ellipse_trunk_1_layer_1_invalid.png",
            ),
            (
                cache_dir / point_cloud_id / "exact_circle_trunk_1_layer_0.png",
                cache_dir / point_cloud_id / "exact_circle_trunk_0_layer_0_valid.png",
            ),
            (
                cache_dir / point_cloud_id / "exact_circle_trunk_1_layer_1.png",
                cache_dir / point_cloud_id / "exact_circle_trunk_0_layer_1_valid.png",
            ),
        ]

        for existing_path, _ in paths:
            existing_path.parent.mkdir(exist_ok=True, parents=True)
            existing_path.touch()

        algorithm.rename_visualizations_after_filtering(filter_mask, point_cloud_id=point_cloud_id)

        for _, renamed_path in paths:
            assert renamed_path.exists()

    def test_compute_trunk_positions(self):
        algorithm = TreeXAlgorithm(trunk_search_circle_fitting_std_num_layers=3)

        layer_circles = np.array([[[0, 0, 1], [0.5, 0, 1], [1, 0, 1], [2, 0, 0.1]]], dtype=np.float64)
        layer_ellipses = np.array(
            [[[-1, -1, -1, -1, -1], [0.5, 0, 1.1, 0.99, 0], [1, 0, 1.1, 0.99, 0], [2, 0, 1.1, 0.99, 0.1]]],
            dtype=np.float64,
        )
        layer_heights = np.array([1, 1.5, 2, 2.5])

        best_circle_combination = np.array([[0, 2, 1]], dtype=np.int64)
        best_ellipse_combination = np.array([[-1, -1, -1]], dtype=np.int64)

        expected_trunk_positions = np.array([[0.3, 0]], dtype=np.float64)

        trunk_positions = algorithm.compute_trunk_positions(
            layer_circles, layer_ellipses, layer_heights, best_circle_combination, best_ellipse_combination
        )

        np.testing.assert_almost_equal(expected_trunk_positions, trunk_positions)

    def test_radius_estimation_gam_circle(self):
        algorithm = TreeXAlgorithm()

        circles = np.array([[1, 1, 1]])
        points = self._generate_circle_points(circles, min_points=50, max_points=50)

        radius_with_full_circle = algorithm.radius_estimation_gam(points, circles[0, :2], 6)

        assert circles[0, 2] == pytest.approx(radius_with_full_circle, abs=0.001)

        radius_with_missing_part = algorithm.radius_estimation_gam(points[:30], circles[0, :2], 7)

        assert circles[0, 2] == pytest.approx(radius_with_missing_part, abs=0.001)

    def test_radius_estimation_gam_ellipse(self):
        algorithm = TreeXAlgorithm()

        ellipses = np.array([[1, 1, 1.2, 0.9, 0]])
        points = self._generate_ellipse_points(ellipses, min_points=50, max_points=50)

        radius_with_full_ellipse = algorithm.radius_estimation_gam(points, ellipses[0, :2], 8)

        expected_radius = (ellipses[0, 2] + ellipses[0, 3]) / 2

        assert expected_radius == pytest.approx(radius_with_full_ellipse, abs=0.02)

        radius_with_missing_part = algorithm.radius_estimation_gam(points[:30], ellipses[0, :2], 9)

        assert expected_radius == pytest.approx(radius_with_missing_part, abs=0.1)

    @pytest.mark.parametrize("create_visualization, use_pathlib", [(False, False), (True, False), (True, True)])
    def test_radius_estimation_gam_visualization(self, create_visualization: bool, use_pathlib: bool, cache_dir):
        visualization_path: Optional[Union[str, Path]] = None
        if create_visualization:
            visualization_path = os.path.join(cache_dir, "test.png")
            if use_pathlib:
                visualization_path = Path(visualization_path)

        algorithm = TreeXAlgorithm()

        circles = np.array([[1, 1, 1]])
        points = self._generate_circle_points(circles, min_points=50, max_points=50)

        algorithm.radius_estimation_gam(points, circles[0, :2], 6, visualization_path=visualization_path)

        if create_visualization:
            if not use_pathlib:
                visualization_path = Path(visualization_path)
            visualization_path = cast(Path, visualization_path)
            assert visualization_path.exists()

    def test_compute_trunk_diameters(self):
        num_instances = 2
        num_layers = 4
        algorithm = TreeXAlgorithm(trunk_search_circle_fitting_std_num_layers=num_layers - 1)

        trunk_layers_xy = [[] for i in range(num_instances)]
        layer_heights = np.array([1, 2, 3, 4])
        layer_circles = np.full((num_instances, num_layers, 3), fill_value=-1, dtype=np.float64)
        layer_ellipses = np.full((num_instances, num_layers, 5), fill_value=-1, dtype=np.float64)

        for layer in range(num_layers):
            if layer == num_layers - 1:
                circles = np.array([[0, 0, 1.3]])
            else:
                circles = np.array([[0, 0, 1 - 0.1 * layer]])
            layer_circles[0, layer] = circles[0]
            trunk_layers_xy[0].append(self._generate_circle_points(circles, min_points=50, max_points=50))

            if layer == 0:
                ellipses = np.array([[0, 0, 1.3, 0.8, 0]])
            else:
                ellipses = np.array([[0, 0, 1.0 - 0.1 * layer, 1.0 - 0.1 * layer, 0]])
            layer_ellipses[1, layer] = ellipses[0]
            trunk_layers_xy[1].append(self._generate_ellipse_points(ellipses, min_points=50, max_points=50))

        best_circle_combination = np.array([[0, 2, 1], [-1, -1, -1]], dtype=np.int64)
        best_ellipse_combination = np.array([[-1, -1, -1], [2, 1, 3]], dtype=np.int64)

        expected_trunk_radii = np.array([1 - 0.03, 1 - 0.03], dtype=np.float64)

        trunk_diameters = algorithm.compute_trunk_diameters(
            layer_circles,
            layer_ellipses,
            layer_heights,
            trunk_layers_xy,
            best_circle_combination,
            best_ellipse_combination,
        )

        np.testing.assert_almost_equal(expected_trunk_radii * 2, trunk_diameters, decimal=4)

    def test_segment_crowns(self):  # pylint: disable=too-many-locals
        point_spacing = 0.1
        ground_plane = self._generate_grid_points((100, 100), point_spacing)
        ground_plane = np.column_stack([ground_plane, np.zeros(len(ground_plane), dtype=np.float64)])

        trunk_1 = np.full((30, 3), fill_value=2.05, dtype=np.float64)
        trunk_1[:, 2] = np.arange(30).astype(np.float64) * point_spacing

        trunk_2 = np.full((35, 3), fill_value=4.65, dtype=np.float64)
        trunk_2[:, 2] = np.arange(35).astype(np.float64) * point_spacing

        crown_1 = self._generate_grid_points((20, 20, 20), point_spacing)
        crown_1[:, :2] += 1.05
        crown_1[:, 2] += 3.1
        crown_2 = self._generate_grid_points((30, 30, 30), point_spacing)
        crown_2[:, :2] += 3.15
        crown_2[:, 2] += 3.6

        xyz = np.concatenate([ground_plane, crown_1, trunk_1, crown_2, trunk_2])
        expected_instance_ids = np.full(len(xyz), fill_value=-1, dtype=np.int64)
        start_tree_1 = len(ground_plane)
        end_tree_1 = start_tree_1 + len(crown_1) + len(trunk_1)
        expected_instance_ids[start_tree_1:end_tree_1] = 0
        expected_instance_ids[end_tree_1:] = 1

        distance_to_dtm = xyz[:, 2]

        is_tree = np.zeros(len(xyz), dtype=bool)
        is_tree[len(ground_plane) :] = True

        tree_positions = np.array([[2.05, 2.05], [4.65, 4.65]], dtype=np.float64)
        trunk_diameters = np.array([0.01, 0.01], dtype=np.float64)

        region_growing_z_scale = 2
        region_growing_seed_layer_height = 0.6
        max_cum_search_dist_without_terrain = (1.3 - region_growing_seed_layer_height / 2) / region_growing_z_scale

        algorithm = TreeXAlgorithm(
            region_growing_voxel_size=0.025,
            region_growing_cum_search_dist_include_terrain=max_cum_search_dist_without_terrain + 0.2,
            region_growing_seed_layer_height=region_growing_seed_layer_height,
            region_growing_z_scale=region_growing_z_scale,
        )

        instance_ids = algorithm.segment_crowns(xyz, distance_to_dtm, is_tree, tree_positions, trunk_diameters)

        assert len(xyz) == len(instance_ids)
        assert len(np.unique(instance_ids)) == 3
        np.testing.assert_array_equal(expected_instance_ids[start_tree_1:], instance_ids[start_tree_1:])
        # some of the terrain points around the trunks are assigned to the tree instances
        assert (instance_ids[:start_tree_1] != -1).sum() == 8

    @pytest.mark.parametrize("create_visualizations", [True, False])
    def test_full_algorithm(self, create_visualizations: bool, cache_dir):  # pylint: disable=too-many-locals
        point_spacing = 0.1
        ground_plane = self._generate_grid_points((100, 100), point_spacing)
        ground_plane = np.column_stack([ground_plane, np.zeros(len(ground_plane), dtype=np.float64)])

        points_per_layer = 200

        num_layers_trunk_1 = 30
        trunk_1: List[float] = []
        for layer in range(num_layers_trunk_1):
            layer_height = layer * point_spacing
            layer_points = self._generate_circle_points(
                np.array([[2.05, 2.05, 0.15]]),
                min_points=points_per_layer,
                max_points=points_per_layer,
                variance=0.01,
                seed=layer,
            )
            layer_points = np.column_stack(
                (layer_points, np.full(len(layer_points), fill_value=layer_height, dtype=np.float64))
            )
            trunk_1.extend(layer_points)

        num_layers_trunk_2 = 35
        trunk_2: List[float] = []
        for layer in range(num_layers_trunk_2):
            layer_height = layer * point_spacing
            layer_points = self._generate_circle_points(
                np.array([[5.65, 5.65, 0.25]]),
                min_points=points_per_layer,
                max_points=points_per_layer,
                variance=0.01,
                seed=layer,
            )
            layer_points = np.column_stack(
                (layer_points, np.full(len(layer_points), fill_value=layer_height, dtype=np.float64))
            )
            trunk_2.extend(layer_points)

        crown_1 = self._generate_grid_points((20, 20, 20), point_spacing)
        crown_1[:, :2] += 1.05
        crown_1[:, 2] += 3.1
        crown_2 = self._generate_grid_points((30, 30, 30), point_spacing)
        crown_2[:, :2] += 4.15
        crown_2[:, 2] += 3.6

        expected_trunk_positions = np.array([[2.05, 2.05], [5.65, 5.65]], dtype=np.float64)
        expected_trunk_diameters = np.array([0.3, 0.5], dtype=np.float64)
        expected_tree_heights = np.array([5.0, 6.5], dtype=np.float64)

        xyz = np.concatenate([ground_plane, crown_1, np.array(trunk_1), crown_2, np.array(trunk_2)])

        visualization_folder = None
        point_cloud_id = None
        if create_visualizations:
            visualization_folder = cache_dir
            point_cloud_id = "test"

        algorithm = TreeXAlgorithm(trunk_search_dbscan_eps=0.05, visualization_folder=visualization_folder)

        instance_ids, trunk_positions, trunk_diameters = algorithm(xyz, point_cloud_id=point_cloud_id)

        point_cloud = pd.DataFrame(xyz, columns=["x", "y", "z"])
        point_cloud["instance_id"] = instance_ids
        point_cloud.to_csv("test_point_cloud.csv", index=False)

        tree_heights = np.empty(2, dtype=np.float64)
        for instance_id in [0, 1]:
            instance_points = xyz[instance_ids == instance_id]
            tree_heights[instance_id] = instance_points[:, 2].max() - instance_points[:, 2].min()

        assert len(xyz) == len(instance_ids)
        assert len(np.unique(instance_ids)) == 3
        np.testing.assert_almost_equal(expected_trunk_positions, trunk_positions, decimal=2)
        np.testing.assert_almost_equal(expected_trunk_diameters, trunk_diameters, decimal=2)
        np.testing.assert_almost_equal(expected_tree_heights, tree_heights, decimal=2)

        algorithm = TreeXAlgorithm(trunk_search_min_cluster_points=10000)

        instance_ids, trunk_positions, trunk_diameters = algorithm(xyz)

        assert len(xyz) == len(instance_ids)
        assert (instance_ids == -1).all()
