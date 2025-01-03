""" Ellipse fitting. """

__all__ = ["fit_ellipse"]

import numpy as np
import numpy.typing as npt

from pointtree._operations_cpp import fit_ellipse as fit_ellipse_cpp  # type: ignore[import-untyped] # pylint: disable=import-error, no-name-in-module


def fit_ellipse(
    xy: npt.NDArray[np.float64], batch_lengths: npt.NDArray[np.int64], num_workers: int = 1
) -> npt.NDArray[np.float64]:
    r"""
    Fits an ellipse to a set of 2D points using the least-squares method described in `Halir, Radim, and Jan Flusser. \
    "Numerically Stable Direct Least Squares Fitting of Ellipses." Proc. 6th International Conference in Central \
    Europe on Computer Graphics and Visualization. WSCG. Vol. 98. Plzen-Bory: Citeseer, 1998. \
    <https://autotrace.sourceforge.net/WSCG98.pdf>`_ This method supports batch processing, i.e., ellipses can be fitted
    to separate sets of points (batch items) in parallel. For this purpose, :code:`batch_lengths` must be set to specify
    which point belongs to which set.

    Args:
        xy: X- and y- coordinates of the points to which the ellipses are to be fitted.
        batch_lengths: Number of points in each item of the input batch. For batch processing, it is
            expected that all points belonging to the same batch item are stored consecutively in the :code:`xy`
            input array. For example, if a batch comprises two batch items with :math:`N_1` points and :math:`N_2`
            points, then :code:`batch_lengths` should be set to :code:`[N_1, N_2]` and :code:`xy[:N_1]`
            should contain the points of the first batch item and :code:`xy[N_1:]` the points of the second batch
            item. If :code:`batch_lengths` is set to :code:`None`, it is assumed that the input points
            belong to a single batch item and batch processing is disabled. Defaults to :code:`None`.
        num_workers: Number of workers threads to use for parallel processing. If set to -1, all CPU threads are used.
            Defaults to 1.

    Returns:
        Parameters of the fitted ellipses in the following order: X- and y-coordinates of the center, radius along the
        semi-major and along the semi-minor axis, and the counterclockwise angle of rotation from the x-axis to the
        semi-major axis of the ellipse. If no ellipse is detected for a batch item, all ellipse parameters for this
        batch item are set to -1.

    Raises:
        TypeError: If :code:`xy` or :code:`batch_length` have an invalid shape or data type.
        ValueError: If the length of :code:`xy` is not equal to the sum of :code:`batch_lengths`.

    Shape:
        - :code:`xy`: :math:`(N, 2)`
        - :code:`batch_lengths`: :math:`(B)`
        - Output: :math:`(B, 5)`

          | where
          |
          | :math:`B = \text{ batch size}`
          | :math:`N = \text{ number of points}`
    """

    return fit_ellipse_cpp(xy, batch_lengths, num_workers)
