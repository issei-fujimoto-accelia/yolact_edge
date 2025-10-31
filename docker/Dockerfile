FROM nvcr.io/nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y  \
    git wget sudo build-essential \
    python3 python3-setuptools python3-pip python3-dev python3-tk \
    ffmpeg libsm6 libxext6 \
    libgtk2.0-dev pkg-config

RUN ln -svf /usr/bin/python3 /usr/bin/python
RUN python -m pip install --upgrade --force pip

# TensorRT
ARG version="8.0.5.39-1+cuda11.0"
RUN apt-get update && apt-get install -y libcudnn8=${version} libcudnn8-dev=${version} && apt-mark hold libcudnn8 libcudnn8-dev

ARG version="7.2.3-1+cuda11.0"
ARG repo="https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64"
## nvidia repo参照
## https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/
RUN \
wget ${repo}/libnvinfer7_${version}_amd64.deb &&\
sudo dpkg -i libnvinfer7_${version}_amd64.deb &&\

wget ${repo}/libnvonnxparsers7_${version}_amd64.deb &&\
sudo dpkg -i libnvonnxparsers7_${version}_amd64.deb &&\

wget ${repo}/libnvparsers7_${version}_amd64.deb &&\
sudo dpkg -i libnvparsers7_${version}_amd64.deb &&\

wget ${repo}/libnvinfer-plugin7_${version}_amd64.deb &&\
sudo dpkg -i libnvinfer-plugin7_${version}_amd64.deb &&\


wget ${repo}/libnvinfer-dev_${version}_amd64.deb &&\
sudo dpkg -i libnvinfer-dev_${version}_amd64.deb &&\

wget ${repo}/libnvonnxparsers-dev_${version}_amd64.deb &&\
sudo dpkg -i libnvonnxparsers-dev_${version}_amd64.deb &&\

wget ${repo}/libnvparsers-dev_${version}_amd64.deb &&\
sudo dpkg -i libnvparsers-dev_${version}_amd64.deb &&\

wget ${repo}/libnvinfer-plugin-dev_${version}_amd64.deb &&\
sudo dpkg -i libnvinfer-plugin-dev_${version}_amd64.deb &&\

wget ${repo}/python3-libnvinfer_${version}_amd64.deb &&\
sudo dpkg -i python3-libnvinfer_${version}_amd64.deb

# ARG version="7.2.3-1+cuda11.0"
# RUN apt-get update && \
#    apt-get install -y libnvinfer7=${version} libnvonnxparsers7=${version} libnvparsers7=${version} libnvinfer-plugin7=${version} libnvinfer-dev=${version} libnvonnxparsers-dev=${version} libnvparsers-dev=${version} libnvinfer-plugin-dev=${version} python3-libnvinfer=${version} && \
#    apt-mark hold libnvinfer7 libnvonnxparsers7 libnvparsers7 libnvinfer-plugin7 libnvinfer-dev libnvonnxparsers-dev libnvparsers-dev libnvinfer-plugin-dev python3-libnvinfer

# create a non-root user
ARG USER_ID=1000
ARG USER=appuser
RUN useradd -m --no-log-init --system --uid ${USER_ID} ${USER} -g sudo 
RUN usermod -aG video ${USER}
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ${USER}
WORKDIR /home/${USER}
ENV PATH="/home/${USER}/.local/bin:${PATH}"

# # Install dependencies
RUN pip install --upgrade pip &&\
pip install --user opencv-python --verbose
RUN pip install --user cython pillow matplotlib GitPython termcolor tensorboard
RUN pip install --user git+https://github.com/haotian-liu/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
RUN pip install --user torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# RUN pip install --upgrade pip &&\
# pip install wheel &&\
# pip install tensorrt --verbose
# RUN pip install tensorrt-cu11 tensorrt-lean-cu11 tensorrt-dispatch-cu11 --verbose

# torch2trt
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt
WORKDIR /home/${USER}/torch2trt
RUN pip install packaging --user
RUN python setup.py install --plugins --user

## realsense
RUN pip install pyrealsense2

WORKDIR /home/${USER}
RUN ln -s /yolact_edge
RUN ln -s /datasets
WORKDIR /home/${USER}/yolact_edge

ENV LANG C.UTF-8
