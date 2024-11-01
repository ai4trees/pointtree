FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git gnupg

RUN sudo -v ; curl https://rclone.org/install.sh | sudo bash

RUN python -m pip install \
    torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html \
    torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu124.html \
    pointtree[dev,docs]
