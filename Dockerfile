FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ffmpeg \
    git \
    libegl1-mesa-dev \
    libglib2.0-0 \
    libgl1 \
    libgtk2.0-dev \
    unzip \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && conda update -n base -c defaults conda -y

WORKDIR /workspace/MGC-6D
COPY environment.yml requirements.txt ./

RUN conda env create -f environment.yml \
    && conda clean -afy

ENV PATH=${CONDA_DIR}/envs/mgc6d/bin:${PATH}
ENV CONDA_DEFAULT_ENV=mgc6d
ENV LD_LIBRARY_PATH=${CONDA_DIR}/envs/mgc6d/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

COPY . .

RUN PYTHON_BIN=python CUDA_HOME=/usr/local/cuda \
    bash scripts/install_revision_dependencies.sh

# RaySt3R remains external; pin it to the commit documented in README.md and
# set RAYST3R_ROOT at runtime. Model weights are intentionally not baked in.

ENV SHELL=/bin/bash
CMD ["bash"]
