"""A tracker that stores the execution time and memory usage of different code sections."""

__all__ = ["PerformanceTracker"]

import pandas as pd


class PerformanceTracker:
    """A tracker that stores the execution time and memory usage of different code sections."""

    def __init__(self):
        self._performance_metrics = {}
        self.reset()

    def reset(self):
        """
        Deletes all tracked execution times.
        """

        self._performance_metrics = {
            "Wallclock Time [s]": {},
            "CPU Time [s]": {},
            "Memory Usage [GB]": {},
            "Memory Increment [GB]": {},
        }

    def save(self, desc: str, wall_clock_time: float, cpu_time: float, memory_usage: float, memory_increment: float):
        """
        Save the execution time of a certain code section.

        Args:
            desc: Description of the tracked code. If a value has already been saved for the description, the values are
                summed.
            value: Execution time of the tracked code.
        """

        for metric_name, metric_value in (
            ("Wallclock Time [s]", wall_clock_time),
            ("CPU Time [s]", cpu_time),
            ("Memory Usage [GB]", memory_usage),
            ("Memory Increment [GB]", memory_increment),
        ):
            if desc in self._performance_metrics[metric_name]:
                self._performance_metrics[metric_name][desc] += metric_value
            else:
                self._performance_metrics[metric_name][desc] = metric_value

    def to_pandas(self) -> pd.DataFrame:
        """
        Returns:
            Tracked execution times and memory usage data as
            `pandas.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`__ with the columns
            :code:`"Description"`, :code:`"Wallclock Time [s]"`, :code:`"CPU Time [s]"`, :code:`"Memory Usage [GB]"`,
            and :code:`"Memory Increment [GB]"`.
        """

        performance_metrics_list = [[key] for key in self._performance_metrics["Wallclock Time [s]"].keys()]

        metric_names = self._performance_metrics.keys()
        for metric_name in metric_names:
            for idx, metric_value in enumerate(self._performance_metrics[metric_name].values()):
                performance_metrics_list[idx].append(metric_value)

        return pd.DataFrame(performance_metrics_list, columns=["Description", *metric_names])
