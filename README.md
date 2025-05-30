<img src="https://github.com/ai4trees/pointtree/blob/main/docs/assets/pointtree-logo-color.png?raw=true" alt="pointtree" width="300" height="100">

## A Python Package for Tree Instance Segmentation in 3D Point Clouds.

![pypi-image](https://badge.fury.io/py/pointtree.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/ai4trees/pointtree/actions/workflows/code-quality-main.yml/badge.svg)](https://github.com/ai4trees/pointtree/actions/workflows/code-quality-main.yml)
[![coverage](https://codecov.io/gh/ai4trees/pointtree/branch/main/graph/badge.svg)](https://codecov.io/github/ai4trees/pointtree?branch=main)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pointtree)

The package contains implementation of the following tree instance segmentation algorithms:

- TreeXAlgorithm
- CoarseToFineAlgorithm

It contains the official source code of the paper ["Burmeister, Josafat-Mattias, et al. "Tree Instance Segmentation in Urban 3D Point Clouds Using a Coarse-to-Fine Algorithm Based on Semantic Segmentation." ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences 10 (2024): 79-86.](https://isprs-annals.copernicus.org/articles/X-4-W5-2024/79/2024/isprs-annals-X-4-W5-2024-79-2024.pdf)

### Package Documentation

The documentation of our package is available [here](https://ai4trees.github.io/pointtree/stable).

### Project Setup

The setup of our package is described in the [documentation](https://ai4trees.github.io/pointtree/stable#get-started).

### How To Use the Package

The `TreeXAlgorithm` segments individual tree instances from point clouds of forest areas. It assumes that the input point cloud contains only terrain and vegetation points. If your data includes other objects (e.g., man-made structures), the algorithm can still be applied, but its accuracy may be reduced.

#### 1. Creating an Algorithm Instance

To get started, create an instance of the `TreeXAlgorithm` class. All parameters have default values, but you can override them by passing keyword arguments to the constructor. For a complete list of parameters and their descriptions, see the [documentation](https://ai4trees.github.io/pointtree/v0.1.0/pointtree.instance_segmentation.html#pointtree.instance_segmentation.TreeXAlgorithm).

```python
from pointtree.instance_segmentation import TreeXAlgorithm

# Optional: specify a folder for saving visualizations of intermediate results
# Note: generating visualizations slows down processing and is recommended only for small datasets
visualization_folder = "./visualizations"  # or set to None to disable

algorithm = TreeXAlgorithm(visualization_folder=visualization_folder)
```

#### 2. Using Presets

We provide presets tailored to typical point cloud characteristics from different laser scanning modalities: terrestrial (TLS), and UAV-borne (ULS). These presets simplify setup for common use cases.

```python
from pointtree.instance_segmentation import TreeXPresetTLS, TreeXPresetULS

preset = TreeXPresetTLS()  # or use TreeXPresetULS()
algorithm = TreeXAlgorithm(**preset)
```

#### 3. Running the Algorithm

The algorithm requires a numpy array of shape (n_points, 3) as input, containing the xyz-coordinates of the point cloud. If available, you can also pass reflection intensity values which may improve segmentation accuracy.

The algorithm returns a tuple of three numpy arrays:

- instance IDs: an array of instance labels (points that belong to the same tree have the same ID, points not belonging to any tree have the ID -1),
- trunk positions: 2D coordinates of the detected tree trunks at breast height
- trunk diameters: diameters of the detected trunks at breast height.

```python
from pointtorch import read

# Load your point cloud (supports .txt, .csv, .las, .laz, .ply)
file_path = "./demo.laz"
point_cloud = read(file_path)

# Run the algorithm
instance_ids, trunk_positions, trunk_diameters = algorithm(
    point_cloud[["x", "y", "z"]].to_numpy(),
    intensities=point_cloud["intensity"].to_numpy(),
    point_cloud_id="test-point-cloud",  # Optional: Used for naming visualization / intermediate outputs
    crs="EPSG:4326"  # Optional: Used for georeferencing intermediate outputs
)

# Add results to the point cloud and save to a new file
point_cloud["instance_id"] = instance_ids
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
