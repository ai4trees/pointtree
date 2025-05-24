"""Tests for pointtree.visualization.plot_fitted_shape."""

import os
from pathlib import Path
import shutil
from typing import Union

import numpy as np
import pytest

from pointtree.visualization import plot_fitted_shape


class TestPlotFittedShape:
    """Tests for pointtree.visualization.plot_fitted_shape."""

    @pytest.fixture
    def cache_dir(self):
        cache_dir = "./tmp/test/visualization/TestPlotFittedShape"
        os.makedirs(cache_dir, exist_ok=True)
        yield cache_dir
        shutil.rmtree(cache_dir)

    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_plot_circle(self, use_pathlib: bool, cache_dir):
        xy = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.float64)
        circle = np.array([0, 0, 1], dtype=np.float64)

        file_path: Union[str, Path] = os.path.join(cache_dir, "circle.png")

        if use_pathlib:
            file_path = Path(file_path)

        plot_fitted_shape(xy, file_path, circle=circle)

        assert os.path.exists(file_path)

    def test_invalid_circle(self, cache_dir):
        xy = np.zeros((4, 2), dtype=np.float64)
        circle = np.array([0, 0, 1, 2], dtype=np.float64)

        file_path = os.path.join(cache_dir, "circle.png")

        with pytest.raises(ValueError):
            plot_fitted_shape(xy, file_path, circle=circle)

    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_plot_ellipse(self, use_pathlib: bool, cache_dir):
        xy = np.array([[-2, 2], [2, -2], [3, 3], [-3, -3]], dtype=np.float64)
        ellipse = np.array([0, 0, np.sqrt(2 * 9), np.sqrt(2 * 4), np.pi / 4], dtype=np.float64)

        file_path: Union[str, Path] = os.path.join(cache_dir, "ellipse.png")

        if use_pathlib:
            file_path = Path(file_path)

        plot_fitted_shape(xy, file_path, ellipse=ellipse)

        assert os.path.exists(file_path)

    def test_invalid_ellipse(self, cache_dir):
        xy = np.zeros((4, 2), dtype=np.float64)
        ellipse = np.array([0, 0, 1], dtype=np.float64)

        file_path = os.path.join(cache_dir, "ellipse.png")

        with pytest.raises(ValueError):
            plot_fitted_shape(xy, file_path, ellipse=ellipse)

    @pytest.mark.parametrize("use_pathlib", [True, False])
    def test_plot_polygon(self, use_pathlib: bool, cache_dir):
        xy = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.float64)
        polygon = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.float64)

        file_path: Union[str, Path] = os.path.join(cache_dir, "line.png")

        if use_pathlib:
            file_path = Path(file_path)

        plot_fitted_shape(xy, file_path, polygon=polygon)

        assert os.path.exists(file_path)

    def test_invalid_polygon(self, cache_dir):
        xy = np.zeros((4, 2), dtype=np.float64)
        polygon = np.array([0, 0, 1], dtype=np.float64)

        file_path = os.path.join(cache_dir, "line.png")

        with pytest.raises(ValueError):
            plot_fitted_shape(xy, file_path, polygon=polygon)
