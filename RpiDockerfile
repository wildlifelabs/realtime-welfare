# https://hub.docker.com/r/nvidia/cuda/tags?name=12.6.3-cudnn-devel-ubuntu
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
ARG TARGETPLATFORM
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV TORCH_CUDA_ARCH_LIST="12.6"
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV CPATH=/usr/local/cuda/include
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Sydney/Australia
# Etc/UTC
RUN apt-get update > /dev/null
RUN apt-get install -yqq --fix-missing software-properties-common
RUN apt-get -yqq upgrade
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update > /dev/null
RUN apt-get install -yqq --fix-missing python3.13 libpython3.13-dev python3.13-dev
RUN apt-get install -yqq --fix-missing git wget bzip2 openssl build-essential libpq-dev xorg libxi6 libxfixes3 libxcursor1 libxdamage1 libxext6 libxfont2
RUN apt-get install -yqq --fix-missing nodejs
RUN apt-get autoremove -y
# RUN timedatectl set-timezone Sydney/Australia

RUN mkdir /script
RUN mkdir /project
RUN mkdir /project/data

WORKDIR /script
COPY ./bin/requirements.txt /script
COPY ./bin/requirements2.txt /script
COPY ./bin/get-pip.py /script
# RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.13 /script/get-pip.py
RUN pip3 install --upgrade pip
RUN pip3 install -r /script/requirements.txt
RUN pip3 install -r /script/requirements2.txt

ENV PYTHONPATH=/project/wildlife-datasets:${PYTHONPATH}
ENV PYTHONPATH=/project/wildlife-tools:${PYTHONPATH}
ENV PYTHONPATH=/project/welfareobs:${PYTHONPATH}

COPY ./bin/py.sh  /script
COPY ./bin/jupyter.sh /script
COPY ./bin/run_ipynb.sh /script
COPY ./bin/tensorboard.sh /script
ENV PATH=/script:${PATH}
