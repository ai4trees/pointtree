""" Base class for implementing tree instance segmentation algorithms. """

__all__ = ["InstanceSegmentationAlgorithm"]

import abc
import logging
import sys
from typing import Any

import pandas as pd

from pointtree.evaluation import TimeTracker


class InstanceSegmentationAlgorithm(abc.ABC):
    """
    Base class for implementing tree instance segmentation algorithms.
    """

    def __init__(self):
        self._time_tracker = TimeTracker()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s:%(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

    def runtime_stats(
        self,
    ) -> pd.DataFrame:
        """
        Returns:
            Tracked execution times as
            `pandas.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`__ with the columns
            :code:`"Description"` and :code:`"Runtime"`.
        """

        return self._time_tracker.to_pandas()

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """
        This method should be overwritten in subclasses to implement the specific tree instance segmentation algorithm.
        """
