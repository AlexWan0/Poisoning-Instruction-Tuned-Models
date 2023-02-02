# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

# setup for miniforge
ARG MINIFORGE_NAME=Miniforge3
ARG MINIFORGE_VERSION=4.12.0-2
ARG TINI_VERSION=v0.18.0
ARG TARGETPLATFORM

ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}

# shell setup
SHELL ["/bin/bash", "--login", "-c"]
WORKDIR /app/tk_instruct_jax/
CMD ["bash"]

# install basics
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    apt-utils \
    curl \
    git \
    vim \
    wget \
    tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install miniforge
RUN apt-get update > /dev/null && \
    apt-get install --no-install-recommends --yes \
        wget bzip2 ca-certificates \
        git > /dev/null && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    TARGETARCH="$(echo ${TARGETPLATFORM} | cut -d / -f 2)"; case ${TARGETARCH} in "ppc64le") TARGETARCH="ppc64el" ;; *) ;; esac ; \
    wget --no-hsts --quiet https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini-${TARGETARCH} -O /usr/local/bin/tini && \
    chmod +x /usr/local/bin/tini && \
    wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    conda clean -tipsy && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean -afy && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc
ENTRYPOINT ["tini", "--"]

# install python packages
COPY environment.yml .
RUN conda env create -f environment.yml
RUN rm -rf environment.yml
RUN conda init bash
RUN echo "conda activate tk_instruct_jax" >> ~/.bashrc
SHELL ["conda", "run", "--no-capture-output", "-n", "tk_instruct_jax", "/bin/bash", "-c"]
RUN pip install --upgrade pip && pip install --upgrade "jax[cuda]==0.3.16" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN conda install pytorch cudatoolkit=11.3 -c pytorch
SHELL ["/bin/bash", "--login", "-c"]
RUN source ~/.bashrc
