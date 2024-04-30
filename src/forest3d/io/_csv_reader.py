""" Point cloud file reader for csv and txt files. """

__all__ = ["CsvReader"]

from typing import List, Optional, Tuple

import pandas

from ._base_point_cloud_reader import BasePointCloudReader
from ._point_cloud_io_data import PointCloudIoData


class CsvReader(BasePointCloudReader):
    """Point cloud file reader for csv and txt files."""

    def supported_file_formats(self) -> List[str]:
        """
        Returns:
            File formats supported by the point cloud file reader.
        """

        return ["csv", "txt"]

    def read(self, file_path: str, columns: Optional[List[str]] = None) -> PointCloudIoData:
        """
        Reads a point cloud file.

        Args:
            file_path: Path of the point cloud file to be read.
            columns: Name of the point cloud columns to be read. The x, y, and z columns are always read.

        Returns:
            Point cloud object.

        Raises:
            ValueError: If the point cloud format is not supported by the reader.
        """
        # The method from the base is called explicitly so that the read method appears in the documentation of this
        # class.
        return super().read(file_path, columns=columns)

    def _read_points(self, file_path: str, columns: Optional[List[str]] = None) -> pandas.DataFrame:
        """
        Reads point data from a point cloud file in csv or txt format.

        Args:
            file_path: Path of the point cloud file to be read.
            columns: Name of the point cloud columns to be read. The x, y, and z columns are always read.

        Returns:
            Point cloud data.
        """

        file_format = file_path.split(".")[-1]
        return pandas.read_csv(file_path, usecols=columns, sep="," if file_format == "csv" else " ")

    @staticmethod
    def _read_max_resolutions(file_path: str) -> Tuple[float, float, float]:
        """
        Reads the maximum resolution for each coordinate dimension from the point cloud file.

        Args:
            file_path: Path of the point cloud file to be read.

        Returns:
            Maximum resolution of the x-, y-, and z-coordinates of the point cloud.
        """

        file_format = file_path.split(".")[-1]
        df = pandas.read_csv(file_path, usecols=["x", "y", "z"], sep="," if file_format == "csv" else " ", dtype=str)

        # The precision of each coordinate is calculated by counting the digits after the decimal.
        x_max_resolution = (
            df["x"].str.split(".").apply(lambda x: round(0.1 ** len(x[1]), len(x)) if len(x) > 1 else 1).min()
        )
        y_max_resolution = (
            df["y"].str.split(".").apply(lambda x: round(0.1 ** len(x[1]), len(x)) if len(x) > 1 else 1).min()
        )
        z_max_resolution = (
            df["z"].str.split(".").apply(lambda x: round(0.1 ** len(x[1]), len(x)) if len(x) > 1 else 1).min()
        )

        return x_max_resolution, y_max_resolution, z_max_resolution
