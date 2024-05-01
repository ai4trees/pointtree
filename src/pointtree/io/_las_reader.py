""" Point cloud file reader for las and laz files. """

__all__ = ["LasReader"]

from typing import List, Optional, Tuple

import laspy
import numpy
import pandas

from ._base_point_cloud_reader import BasePointCloudReader
from ._point_cloud_io_data import PointCloudIoData


class LasReader(BasePointCloudReader):
    """Point cloud file reader for las and laz files."""

    _standard_field_defaults = {
        "return_number": 1,
        "number_of_returns": 1,
    }

    def __init__(self, ignore_default_columns: bool = True) -> None:
        """
        Args:
            ignore_default_columns: Whether fields of a las or laz file that only contain default values should be
                ignored. Defaults to `True`.
        """
        super().__init__()
        self._ignore_default_columns = ignore_default_columns

    def supported_file_formats(self) -> List[str]:
        """
        Returns:
            File formats supported by the point cloud file reader.
        """

        return ["las", "laz"]

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
        Reads point data from a point cloud file in las or laz format.

        Args:
            file_path: Path of the point cloud file to be read.
            columns: Name of the point cloud columns to be read. The x, y, and z columns are always read.

        Returns:
            Point cloud data.
        """
        las_data = laspy.read(file_path)
        data = numpy.array([las_data.x, las_data.y, las_data.z]).T
        point_cloud_df = pandas.DataFrame(data, columns=["x", "y", "z"])

        for column_name in las_data.header.point_format.standard_dimension_names:
            if column_name in ["X", "Y", "Z"] or columns is not None and column_name not in columns:
                continue
            column_values = numpy.array(las_data[column_name])
            default_value = LasReader._standard_field_defaults.get(column_name, 0)
            if (column_values != default_value).any() or not self._ignore_default_columns:
                point_cloud_df[column_name] = column_values

        for column_name in las_data.header.point_format.extra_dimension_names:
            if columns is not None and column_name not in columns:
                continue
            point_cloud_df[column_name] = las_data[column_name]

        return point_cloud_df

    @staticmethod
    def _read_max_resolutions(file_path: str) -> Tuple[float, float, float]:
        """
        Reads the maximum resolution for each coordinate dimension from the point cloud file.

        Args:
            file_path: Path of the point cloud file to be read.

        Returns:
            Maximum resolution of the x-, y-, and z-coordinates of the point cloud.
        """

        with laspy.open(file_path, "r") as f:
            scales = f.header.scales

        return scales
