FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git

RUN python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

COPY --from=builder /opt/conda/lib/python3.11/site-packages/pytorch3d /opt/conda/lib/python3.11/site-packages
COPY --from=builder /opt/conda/lib/python3.11/site-packages/pytorch3d-*.dist-info /opt/conda/lib/python3.11/site-packages

RUN apt-get update && apt-get install -y --no-install-recommends curl git gnupg make openssh-client unzip && \
    curl https://rclone.org/install.sh | bash

RUN python -m pip install \
    torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html \
    torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu124.html \
    pointtree[dev,docs]
