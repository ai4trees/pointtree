"""Tests for pointtree.operations.fit_ellipse."""

from typing import List

import numpy as np
import numpy.typing as npt
import pytest

from pointtree.operations import fit_ellipse

from ..utils import generate_ellipse_points


class TestFitEllipse:
    """Tests for pointtree.operations.fit_ellipse."""

    @pytest.mark.parametrize("num_workers", [1, -1])
    @pytest.mark.parametrize("storage_format", ["C", "F"])
    @pytest.mark.parametrize("scalar_type", [np.float32, np.float64])
    def test_ellipse_fitting(self, num_workers: int, storage_format: str, scalar_type: np.dtype):
        batch_size = 3
        batch_lengths = np.zeros(batch_size, dtype=np.int64)
        xy_list: List[npt.NDArray[np.float64]] = []
        expected_ellipses: List[npt.NDArray[np.float64]] = []

        start_idx = 0
        for batch_idx in range(batch_size):
            ellipses = np.array([[0, batch_idx, 2 * (batch_idx + 1), 1, np.pi * batch_idx / 3]], dtype=np.float64)
            expected_ellipses.append(ellipses)
            current_ellipse_points = generate_ellipse_points(ellipses, min_points=100, max_points=200)
            xy_list.append(current_ellipse_points)
            batch_lengths[batch_idx] = len(current_ellipse_points)
            start_idx += len(current_ellipse_points)

        xy = np.concatenate(xy_list).astype(scalar_type).copy(order=storage_format)
        fitted_ellipses = fit_ellipse(xy, batch_lengths, num_workers=num_workers)

        assert fitted_ellipses.dtype == scalar_type
        assert fitted_ellipses.flags.owndata is False

        decimal = 12 if scalar_type == np.float64 else 3
        np.testing.assert_almost_equal(  # type: ignore[call-overload]
            np.concatenate(expected_ellipses).astype(scalar_type), fitted_ellipses, decimal=decimal
        )

    def test_invalid_xy(self):
        xy = np.random.rand(10, 3).astype(np.float64)
        batch_lengths = np.array([10], dtype=np.int64)

        with pytest.raises(TypeError):
            fit_ellipse(xy, batch_lengths)

    def test_invalid_batch_length(self):
        xy = np.random.rand(10, 2).astype(np.float64)
        batch_lengths = np.array([9], dtype=np.int64)

        with pytest.raises(ValueError):
            fit_ellipse(xy, batch_lengths)
