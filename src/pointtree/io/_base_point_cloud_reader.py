""" Abstract base class for implementing point cloud file readers. """

__all__ = ["BasePointCloudReader"]

import abc
import os
from typing import List, Optional, Tuple

import pandas

from ._point_cloud_io_data import PointCloudIoData


class BasePointCloudReader(abc.ABC):
    """Abstract base class for implementing point cloud file readers."""

    @abc.abstractmethod
    def supported_file_formats(self) -> List[str]:
        """
        Returns:
            File formats supported by the point cloud file reader.
        """

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

        file_name = os.path.basename(file_path)
        file_format = file_name.split(".")[-1]
        file_id = file_name.rstrip(file_format)
        if file_format not in self.supported_file_formats():
            raise ValueError(f"The {file_format} format is not supported by the point cloud reader.")

        if columns is not None:
            columns = columns.copy()

            # The x, y, z coordinates are always loaded.
            for idx, coord in enumerate(["x", "y", "z"]):
                if coord not in columns:
                    columns.insert(idx, coord)

        point_cloud_df = self._read_points(file_path, columns=columns)
        (x_max_resolution, y_max_resolution, z_max_resolution) = self._read_max_resolutions(file_path)

        return PointCloudIoData(
            point_cloud_df,
            identifier=file_id,
            x_max_resolution=x_max_resolution,
            y_max_resolution=y_max_resolution,
            z_max_resolution=z_max_resolution,
        )

    @abc.abstractmethod
    def _read_points(self, file_path: str, columns: Optional[List[str]] = None) -> pandas.DataFrame:
        """
        Reads point data from a point cloud file. This method has to be overriden by child classes.

        Args:
            file_path: Path of the point cloud file to be read.
            columns: Name of the point cloud columns to be read.

        Returns:
            Point cloud data.
        """

    @staticmethod
    @abc.abstractmethod
    def _read_max_resolutions(file_path: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Reads the maximum resolution for each coordinate dimension from the point cloud file.

        Args:
            file_path: Path of the point cloud file to be read.

        Returns:
            Maximum resolution of the x-, y-, and z-coordinates of the point cloud.
        """
