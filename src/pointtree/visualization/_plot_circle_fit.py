from pathlib import Path
from typing import Union


from matplotlib import pyplot as plt
import numpy as np


def plot_circle_fit(xy: np.ndarray, output_path: Union[str, Path]) -> None:
    x_axis_min = xy[:, 0].min() - 1
    x_axis_max = xy[:, 0].max() - 1
    y_axis_min = xy[:, 1].min() - 1
    y_axis_max = xy[:, 1].max() - 1
    plt.xlim(x_axis_min, x_axis_max)
    plt.ylim(y_axis_min, y_axis_max)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.plot(xy[:, 0], xy[:, 1], ".")
    plt.savefig(output_path)
    plt.close()
