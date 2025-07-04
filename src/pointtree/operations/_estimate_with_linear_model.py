"""Trains a linear model and computes the predictions of the model for the given data."""

__all__ = ["estimate_with_linear_model"]

from typing import Tuple

from sklearn.linear_model import LinearRegression

from pointtree.type_aliases import FloatArray


def estimate_with_linear_model(
    x_train: FloatArray, y_train: FloatArray, x_predict: FloatArray
) -> Tuple[FloatArray, LinearRegression]:
    r"""
    Fits a linear model to the training data :code:`x_train` and :code:`y_train`, and computes the predictions of
    the model for :code:`x_predict`.

    Args:
        x_train: Training data.
        y_train: Target values.
        x_predict: Inference data.

    Returns:
        :Tuple of two elements:
            - An array containing the predictions for :code:`x_predict`
            - The fitted linear model

    Shape:
        - :code:`x_train`: :math:`(N)` or :math:`(N, D)`
        - :code:`y_train`: :math:`(N)`
        - :code:`x_predict`: :math:`(N')` or :math:`(N', D)`
        - Output: :math:`(N')`

        | where
        |
        | :math:`N = \text{ number of training samples}`
        | :math:`N' = \text{ number of inference samples}`
        | :math:`D = \text{ number of input features}`
    """

    if x_train.ndim == 1:
        x_train = x_train.reshape((-1, 1))
    if x_predict.ndim == 1:
        x_predict = x_predict.reshape((-1, 1))

    regression = LinearRegression().fit(x_train, y_train)

    return regression.predict(x_predict), regression
