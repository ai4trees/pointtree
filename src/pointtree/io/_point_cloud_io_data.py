""" Point cloud data structure used for I/O. """

__all__ = ["PointCloudIoData"]

from dataclasses import dataclass
from typing import Optional

import pandas


@dataclass
class PointCloudIoData:
    """Point cloud data structure used ."""

    data: pandas.DataFrame
    identifier: Optional[str] = None
    x_max_resolution: Optional[float] = None
    y_max_resolution: Optional[float] = None
    z_max_resolution: Optional[float] = None
