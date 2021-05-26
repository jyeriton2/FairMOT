# cuda version check
FROM nvidia/cuda:10.1-devel-ubuntu18.04

ENV DEBCONF_NOWARNINGS yes
ARG DEBIAN_FRONTEND=noninteractive

# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
# apt-cache policy <package> : apt-get <package> version name check
ENV TENSORFLOW_VERSION=2.3.0
ENV PYTORCH_VERSION=1.6.0
ENV TORCHVISION_VERSION=0.7.0
ENV CUDNN_VERSION=7.6.5.32-1+cuda10.1
ENV NCCL_VERSION=2.7.8-1+cuda10.1
ENV MXNET_VERSION=1.6.0.post0

# Python 3.7 is supported by Ubuntu Bionic out of the box
ARG python=3.7
ENV PYTHON_VERSION=${python}

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        g++-4.8 \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libcudnn7=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-distutils \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install TensorFlow, Keras, PyTorch and MXNet
RUN pip install future typing packaging
RUN pip install tensorflow==${TENSORFLOW_VERSION} \
                tensorflow-gpu==${TENSORFLOW_VERSION} \
                keras \
                h5py

# https://download.pytorch.org/whl/torch_stable.html -> table check
RUN PYTAGS=$(python -c "from packaging import tags; tag = list(tags.sys_tags())[0]; print(f'{tag.interpreter}-{tag.abi}')") && \
    pip install https://download.pytorch.org/whl/cu101/torch-${PYTORCH_VERSION}%2Bcu101-${PYTAGS}-linux_x86_64.whl \
        https://download.pytorch.org/whl/cu101/torchvision-${TORCHVISION_VERSION}%2Bcu101-${PYTAGS}-linux_x86_64.whl
RUN pip install mxnet-cu101==${MXNET_VERSION}

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

RUN export PATH=$PATH:/usr/local/openmpi/bin
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/openmpi/lib
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_NCCL_HOME=/usr/lib/x86_64-linux-gnu HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=1 \
         pip install --no-cache-dir horovod && \
    ldconfig

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Download examples
RUN apt-get install -y --no-install-recommends subversion && \
    svn checkout https://github.com/horovod/horovod/trunk/examples && \
    rm -rf /examples/.svn

WORKDIR "/examples"


RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get dist-upgrade

RUN apt-get install locales
RUN locale-gen ko_KR.UTF-8
RUN update-locale LANG=ko_KR.UTF-8

RUN apt-get install -y libffi-dev \
                       libxml2-dev \
                       libxslt1-dev \
                       zlib1g-dev

RUN apt-get install -y libssl-dev \
                       openssl \
                       zip \
                       openjdk-11-jdk

RUN apt-get update

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.7 get-pip.py --force-reinstall

RUN pip3 install cython
RUN apt-get update
RUN apt-get install -y python-opencv

RUN pip3 install opencv-python
RUN apt-get install -y ffmpeg
RUN pip3 install scikit-video

RUN apt-get install -y python3-numpy \
                       python-numpy

RUN pip3 install beautifulsoup4 \
                 lap \
                 motmetrics \
                 progress \
                 flake8 \
                 Flask \
                 flask_cors \
                 JPype1-py3 \
                 matplotlib \
                 pandas \
                 torch-optimizer==0.0.1a15

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
