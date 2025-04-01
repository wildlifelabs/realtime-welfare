# FROM 11.6.1-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ARG MINIFORGE_NAME=Miniforge3
ARG MINIFORGE_VERSION=24.11.2-1
ARG TARGETPLATFORM
ARG CONDA_YAML=wt.yml
ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV CPATH=/usr/local/cuda/include
RUN apt-get update > /dev/null && \
    apt-get install --no-install-recommends --yes \
    wget bzip2 ca-certificates \
    git \
    tini \
    > /dev/null && \
    apt-get clean && \
    wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all --yes  && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Sydney/Australia
# Etc/UTC
RUN apt-get update
RUN apt-get -yqq upgrade
RUN apt-get install -yqq --fix-missing build-essential libpq-dev xorg libxi6 libxfixes3 libxcursor1 libxdamage1 libxext6 libxfont2
RUN apt-get install -yqq --fix-missing nodejs npm
RUN apt-get autoremove -y
# RUN timedatectl set-timezone Sydney/Australia

RUN mkdir /script
RUN mkdir /project
RUN mkdir /project/data

WORKDIR /script
COPY ./bin/$CONDA_YAML /script
RUN mamba env create -f $CONDA_YAML
RUN npm install --save-dev bash-language-server

ENV PYTHONPATH=/project/wildlife-datasets:${PYTHONPATH}
ENV PYTHONPATH=/project/wildlife-tools:${PYTHONPATH}
ENV PYTHONPATH=/project/welfareobs:${PYTHONPATH}

COPY ./bin/py.sh  /script
COPY ./bin/jupyter.sh /script
COPY ./bin/run_ipynb.sh /script
COPY ./bin/tensorboard.sh /script
ENV PATH=/script:${PATH}
