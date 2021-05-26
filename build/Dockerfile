From pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
ENV DEBCONF_NOWARNINGS yes
ARG DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG ko_KR.UTF-8
ENV LANGUAGE ko_KR.UTF-8

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get dist-upgrade

RUN apt-get install locales
RUN locale-gen ko_KR.UTF-8
RUN update-locale LANG=ko_KR.UTF-8

RUN apt-get install -y g++ \
                       openjdk-11-jdk

RUN apt-get install -y vim
RUN apt-get update

RUN apt-get install -y build-essential \
                       libssl-dev \
                       openssl \
                       cmake \
                       zip \
                       wget

RUN apt-get install -y curl \
                       git

RUN apt-get update

RUN pip install cython
RUN apt-get update
RUN apt-get install -y python-opencv
RUN pip install opencv-python
RUN apt-get install -y ffmpeg
RUN pip install scikit-video

RUN apt-get install -y python3-numpy \
                       python-numpy

RUN pip install beautifulsoup4 \
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
