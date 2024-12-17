<img src="https://github.com/ai4trees/pointtree/blob/main/docs/assets/pointtree-logo-color.png?raw=true" alt="pointtree" width="300" height="100">

## A Python Package for Tree Instance Segmentation in 3D Point Clouds.

![pypi-image](https://badge.fury.io/py/pointtree.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/ai4trees/pointtree/actions/workflows/code-quality-main.yml/badge.svg)](https://github.com/ai4trees/pointtree/actions/workflows/code-quality-main.yml)
[![coverage](https://codecov.io/gh/ai4trees/pointtree/branch/main/graph/badge.svg)](https://codecov.io/github/ai4trees/pointtree?branch=main)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pointtree)

The package contains the official source code of the paper ["Burmeister, Josafat-Mattias, et al. "Tree Instance Segmentation in Urban 3D Point Clouds Using a Coarse-to-Fine Algorithm Based on Semantic Segmentation." ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences 10 (2024): 79-86.](https://isprs-annals.copernicus.org/articles/X-4-W5-2024/79/2024/isprs-annals-X-4-W5-2024-79-2024.pdf)

### Package Documentation

The documentation of our package is available [here](https://ai4trees.github.io/pointtree/stable).

### Project Setup

The setup of our package is described in the [documentation](https://ai4trees.github.io/pointtree/stable#get-started).

### Using the Package on Your Data

The `TreeXAlgorithm` can be used to segment tree instances in point clouds of forest areas. The algorithm assumes that
the input point clouds includes terrain and vegetation points, but no other types of objects (e.g., no man-made
structures). The following code can be used to apply the `TreeXAlgorithm` to a point cloud file:

```python
from pointtorch import read
import pandas as pd
import numpy as np

from pointtree.instance_segmentation import TreeXAlgorithm

# path of the input point cloud file
# pointtorch.read currently supports .csv, .las, and .laz files
file_path = "./demo.laz"
point_cloud = read(file_path)

# if you want to visualize intermediate results of the algorithm, you can to specify a directory in which to save the images
visualization_folder = "./visualizations"

# creating visualizations slows down the algorithm and is therefore only recommended for small datasets
# if you don't need the visualizations, you can just set the visualization_folder to None:
visualization_folder = None

# all parameters of the algorithm have default values
# to overwrite the default settings, you can pass additional keyword arguments to the constructor of the TreeXAlgorithm class
algorithm = TreeXAlgorithm(visualization_folder=visualization_folder)

# the only required input of the algorithm is a numpy array that contains the xyz-coordinates of all points
# to create visualizations of the intermediate results, you additionally need to specify a point_cloud_id, which is used in the file names of the created images
# the algorithm returns three values: an array with instance IDs for all input points (points that belong to the same tree have the same ID, points not belonging to any tree have the ID -1), the 2D trunk positions at breast height and the trunk diameters at breast height
instance_ids, trunk_positions, trunk_diameters = algorithm(point_cloud[["x", "y", "z"]].to_numpy(), point_cloud_id="Sauen")

# add the instance IDs as a new column to the point cloud 
point_cloud["instance_id"] = instance_ids

# save the segmented point cloud in a new file
point_cloud.to("./demo_segmented.laz", columns=["x", "y", "z", "instance_id"])
```

### How to Cite

If you use our code, please consider citing our paper:

```
@article{Burmeister_Tree_Instance_Segmentation_2024,
author = {Burmeister, Josafat-Mattias and Richter, Rico and Reder, Stefan and Mund, Jan-Peter and Döllner, Jürgen},
doi = {10.5194/isprs-annals-X-4-W5-2024-79-2024},
journal = {ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
pages = {79--86},
title = {{Tree Instance Segmentation in Urban 3D Point Clouds Using a Coarse-to-Fine Algorithm Based on Semantic Segmentation}},
volume = {X-4/W5-2024},
year = {2024}
}
```
