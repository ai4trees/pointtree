""" Point cloud file reader for csv, las, laz, and txt files. """

__all__ = ["PointCloudReader"]

from typing import List, Optional

from ._csv_reader import CsvReader
from ._las_reader import LasReader
from ._point_cloud_io_data import PointCloudIoData


class PointCloudReader:
    """Point cloud file reader for csv, las, laz, and txt files."""

    def __init__(self):
        self._readers = {}
        for reader in [CsvReader(), LasReader()]:
            for file_format in reader.supported_file_formats():
                self._readers[file_format] = reader

    def supported_file_formats(self) -> List[str]:
        """
        Returns:
            File formats supported by the point cloud file reader.
        """
        return list(self._readers.keys())

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

        file_format = file_path.split(".")[-1]
        if file_format not in self.supported_file_formats():
            raise ValueError(f"The {file_format} format is not supported by the point cloud reader.")
        return self._readers[file_format].read(file_path, columns=columns)
