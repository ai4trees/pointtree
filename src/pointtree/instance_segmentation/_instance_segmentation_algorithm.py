""" Base class for implementing tree instance segmentation algorithms. """

__all__ = ["InstanceSegmentationAlgorithm"]

import abc
import logging
import sys
from typing import Any

import pandas as pd

from pointtree.evaluation import PerformanceTracker


class InstanceSegmentationAlgorithm(abc.ABC):
    """
    Base class for implementing tree instance segmentation algorithms.
    """

    def __init__(self):
        self._performance_tracker = PerformanceTracker()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s:%(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

    def performance_metrics(
        self,
    ) -> pd.DataFrame:
        """
        Returns:
            Tracked performance metrics as
            `pandas.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`__ with the columns
            :code:`"Description"`, :code:`"Wallclock Time [s]"`, :code:`"CPU Time [s]"`, :code:`"Memory Usage [GB]"`,
            and :code:`"Memory Increment [GB]"`.
        """

        return self._performance_tracker.to_pandas()

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """
        This method should be overwritten in subclasses to implement the specific tree instance segmentation algorithm.
        """
