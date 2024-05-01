""" Point cloud file writer for csv, las, laz, and txt files. """

__all__ = ["PointCloudWriter"]

from typing import List, Optional

from ._csv_writer import CsvWriter
from ._las_writer import LasWriter
from ._point_cloud_io_data import PointCloudIoData


class PointCloudWriter:
    """Point cloud file writer for csv, las, laz, and txt files."""

    def __init__(self):
        self._writers = {}
        for writer in [CsvWriter(), LasWriter()]:
            for file_format in writer.supported_file_formats():
                self._writers[file_format] = writer

    def supported_file_formats(self) -> List[str]:
        """
        Returns:
            File formats supported by the point cloud file writer.
        """
        return list(self._writers.keys())

    def write(self, point_cloud: PointCloudIoData, file_path: str, columns: Optional[List[str]] = None) -> None:
        """
        Writes a point cloud to a file.

        Args:
            point_cloud: Point cloud to be written.
            file_path: Path of the output file.
            columns: Point cloud columns to be written. The x, y, and z columns are always written.

        Raises:
            ValueError: If the point cloud format is not supported by the writer or if `columns` contains a column name
                that is not existing in the point cloud.
        """

        file_format = file_path.split(".")[-1]
        if file_format not in self.supported_file_formats():
            raise ValueError(f"The {file_format} format is not supported by the point cloud writer.")
        return self._writers[file_format].write(point_cloud, file_path, columns=columns)
