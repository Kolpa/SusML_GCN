FROM ubuntu:18.04
# Atilim Gunes Baydin, University of Oxford
# November 2018
LABEL maintainer="Atilim Gunes Baydin <gunes@robots.ox.ac.uk>"

RUN apt-get update && apt-get install -y git wget build-essential libopenmpi-dev cmake python3-pip
RUN mkdir /code

# ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
RUN pip3 install numpy ninja pyyaml cffi
# RUN conda install -c mingfeima mkldnn

RUN cd /code && git clone --recursive -b v1.5.0 https://github.com/pytorch/pytorch
ENV NO_CUDA=1
RUN cd /code/pytorch && python3 setup.py install