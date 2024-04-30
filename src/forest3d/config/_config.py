""" Utilities for configuring the package setup. """

__all__ = ["pytorch3d_is_available"]


def pytorch3d_is_available() -> bool:
    """
    Returns: `True` if PyTorch3D is installed and `False` otherwise.
    """

    pytorch3d_available = False
    try:
        import pytorch3d as _  # pylint: disable=import-outside-toplevel

        pytorch3d_available = True
    except (ModuleNotFoundError, TypeError):
        pass

    return pytorch3d_available
