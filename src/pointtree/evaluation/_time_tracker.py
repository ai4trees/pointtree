"""A tracker that stores the execution times of different code sections."""

__all__ = ["TimeTracker"]

import pandas as pd


class TimeTracker:
    """A tracker that stores the execution times of different code sections."""

    def __init__(self):
        self._time_tracking_results = {}

    def reset(self):
        """
        Deletes all tracked execution times.
        """

        self._time_tracking_results = {}

    def save(self, desc: str, value: float):
        """
        Save the execution time of a certain code section.

        Args:
            desc: Description of the tracked code. If a value has already been saved for the description, the values are
                summed.
            value: Execution time of the tracked code.
        """

        if desc in self._time_tracking_results:
            self._time_tracking_results[desc] += value
        else:
            self._time_tracking_results[desc] = value

    def to_pandas(self) -> pd.DataFrame:
        """
        Returns:
            Tracked execution times as
            `pandas.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`__ with the columns
            :code:`"Description"` and :code:`"Runtime"`.
        """

        time_tracking_list = [[key, value] for key, value in self._time_tracking_results.items()]
        return pd.DataFrame(time_tracking_list, columns=["Description", "Runtime"])
