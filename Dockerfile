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
    cmake \
    ffmpeg \
    gcc \
    g++ \
    git \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libboost-thread-dev \
    libegl1-mesa-dev \
    libglib2.0-0 \
    libgl1 \
    libgtk2.0-dev \
    make \
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

COPY . .

RUN conda install -y -c conda-forge eigen=3.4.0 \
    && cd sam2 && pip install -e . && cd .. \
    && cd bop_toolkit && pip install -e . && cd ..

# FoundationPose CUDA extensions are machine/toolchain-sensitive. Build them
# inside the container after mounting/downloading the required weights:
#   cd foundationpose && bash build_all_conda.sh
# RaySt3R remains external; set RAYST3R_ROOT at runtime.

ENV SHELL=/bin/bash
CMD ["bash"]
