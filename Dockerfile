FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel AS Builder

###################### Create Virtual Environment ########################

ENV VIRTUAL_ENV=/workspace/venv
RUN python -m venv $VIRTUAL_ENV

# by adding the venv to the search path, we avoid activating it in each command
# see https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

################ PyTorch3D build #################################

# needed to install the CUDA version of PyTorch3D
# see https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
# see https://github.com/pytorch/extension-cpp/issues/71
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"

RUN python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

COPY --from=builder /workspace/venv/Lib/site-packages/pytorch3d $(python -m site --user-site)

RUN python -m pip install \
    torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html \
    torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu124.html \
    pointtree[dev,docs]
