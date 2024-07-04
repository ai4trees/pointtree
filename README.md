# PointTree

## A Python Package for Deriving Information About Vegetation From 3D Point Clouds.

The package contains the official source code of the paper ["Burmeister, Josafat-Mattias, et al. "Tree Instance Segmentation in Urban 3D Point Clouds Using a Coarse-to-Fine Algorithm Based on Semantic Segmentation." ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences 10 (2024): 79-86.](https://isprs-annals.copernicus.org/articles/X-4-W5-2024/79/2024/isprs-annals-X-4-W5-2024-79-2024.pdf)

### Package Documentation

The documentation of our package is available [here](https://ai4trees.github.io/pointtree/).

### Project Setup

First, clone our repository:

```
git clone https://github.com/ai4trees/pointtree.git
```

Change to the pointtree directory:

```
cd pointtree
```

Install PyTorch using the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/). Then, install our package:

```
python -m pip install .
```

(Optional) To install the package including all dependencies needed for development, testing, and building the documentation:

```
python -m pip install .[dev,docs]
```

### Running the tests

To execute the tests, run:

```
pytest 
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
