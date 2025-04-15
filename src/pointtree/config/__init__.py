"""Utilities for configuring the package setup."""

from ._config import *

__all__ = [name for name in globals().keys() if not name.startswith("_")]
