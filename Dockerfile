FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV LANG C.UTF-8
ENV TZ="Asia/Tokyo"
WORKDIR /root/
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
# ==================================================================
# tools
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        software-properties-common \
        wget \
        git \
        git-lfs \
        vim \
        nano \
        libssl-dev \
        curl \
        unzip \
        unrar \
        cmake \
        libglu1 libxi6 libgconf-2-4 libsdl1.2-dev 

# ==================================================================
# python
# ------------------------------------------------------------------

RUN apt-get update && \
    apt update -y && \
    apt upgrade -y && \
    APT_INSTALL="apt-get install -y --no-install-recommends" && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.10 \
        python3.10-dev \
        python3.10-distutils\
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/pip/get-pip.py && \
    python3.10 ~/get-pip.py && \
    ln -s /usr/bin/python3.10 /usr/local/bin/python


# ==================================================================
# pytorch
# ------------------------------------------------------------------

RUN python -m pip --no-cache-dir install --upgrade  torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118


# ==================================================================
# python packages
# ------------------------------------------------------------------
COPY requirements.txt /root/
RUN python -m pip --no-cache-dir install --upgrade -r  requirements.txt
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*