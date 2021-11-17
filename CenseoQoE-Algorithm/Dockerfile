FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
MAINTAINER shaoguowen
# setup timezone
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN mkdir -p /run/sshd && \
    apt-get update && \
    apt-get install -y openssh-server vim curl && \
    apt-get install  wget git cron zip nasm -y && \
    apt-get install libsm6 libxrender1 libxext-dev -y && \
    apt-get install nginx -y && \
	apt-get install -y --no-install-recommends \
        build-essential \
        ninja-build \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-tk \
        && \
    apt-get install ffmpeg -y && \
	rm -rf /usr/bin/python && \
    ln -s /usr/bin/python3.6 /usr/bin/python

# 设置环境变量
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# python依赖
COPY ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

USER root
RUN mkdir -p /my_projects/CenseoQoE-Algorithm
COPY . /my_projects/CenseoQoE-Algorithm
WORKDIR /my_projects/CenseoQoE-Algorithm