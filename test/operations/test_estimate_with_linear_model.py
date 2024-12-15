""" Tests for pointtree.operations.estimate_with_linear_model. """

import numpy as np
import pytest

from pointtree.operations import estimate_with_linear_model


class TestEstimateWithLinearModel:  # pylint: disable=too-few-public-methods
    """Tests for pointtree.operations.estimate_with_linear_model."""

    @pytest.mark.parametrize("expand_input_shape", [True, False])
    def test_estimate_with_linear_model(self, expand_input_shape: bool):
        x_train = np.array([0, 1, 3], dtype=np.float64)
        y_train = np.array([1, 2, 4], dtype=np.float64)
        x_predict = np.array([2, 4], dtype=np.float64)

        if expand_input_shape:
            x_train = x_train.reshape(-1, 1)
            x_predict = x_predict.reshape(-1, 1)

        expected_predictions = np.array([3, 5], dtype=np.float64)

        predictions, linear_model = estimate_with_linear_model(x_train, y_train, x_predict)

        np.testing.assert_array_equal(expected_predictions, predictions)
        if not expand_input_shape:
            x_predict = x_predict.reshape(-1, 1)
        np.testing.assert_array_equal(expected_predictions, linear_model.predict(x_predict))
