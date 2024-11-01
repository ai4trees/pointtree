FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

RUN python -m pip install \
    torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html \
    torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu124.html \
    pointtree[dev,docs]
