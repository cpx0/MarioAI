FROM nvcr.io/nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# ENV https_proxy "http://proxy..."
# ENV http_proxy "http://proxy..."

ARG TF_SERVING_VERSION=0.0.0
ARG NB_USER=cpx0

# TODO: User should be refactored instead of hard coded jovyan

USER root

ENV DEBIAN_FRONTEND noninteractive

ENV NB_USER $NB_USER

ENV NB_UID 1000
ENV HOME /home/$NB_USER
ENV NB_PREFIX /
ENV PATH $HOME/.local/bin:$PATH


# Use bash instead of sh
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -yq --no-install-recommends \
  apt-transport-https \
  build-essential \
  bzip2 \
  ca-certificates \
  curl \
  g++ \
  git \
  gnupg \
  graphviz \
  locales \
  lsb-release \
  openssh-client \
  sudo \
  unzip \
  vim \
  wget \
  zip \
  emacs \
  python3.8 \
  python3-pip \
  python3-dev \
  python3-setuptools \
  libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev \
  xz-utils tk-dev libffi-dev liblzma-dev python-openssl \
  libopencv-dev \
  cmake libopenmpi-dev zlib1g-dev \
  && apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Install Nodejs for jupyterlab-manager
RUN curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
RUN apt-get update && apt-get install -yq --no-install-recommends \
  nodejs \
  && apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
  locale-gen

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

# Create NB_USER user with UID=1000 and in the 'users' group
# but allow for non-initial launches of the notebook to have
# $HOME provided by the contents of a PV
RUN useradd -M -s /bin/bash -N -u $NB_UID $NB_USER && \
  chown -R ${NB_USER}:users /usr/local/bin && \
  mkdir -p $HOME && \
  chown -R ${NB_USER}:users ${HOME}

# Install Tini - used as entrypoint for container
RUN cd /tmp && \
  wget --quiet https://github.com/krallin/tini/releases/download/v0.18.0/tini && \
  echo "12d20136605531b09a2c2dac02ccee85e1b874eb322ef6baf7561cd93f93c855 *tini" | sha256sum -c - && \
  mv tini /usr/local/bin/tini && \
  chmod +x /usr/local/bin/tini

# NOTE: Beyond this point be careful of breaking out
# or otherwise adding new layers with RUN, chown, etc.
# The image size can grow significantly.

# python to 3.8
RUN apt-get update && apt-get install -yq --no-install-recommends \
  python-is-python3 \
  && apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# GLFW
RUN apt-get update && apt-get install -yq --no-install-recommends \
  libglew2.1  libgl1-mesa-glx libosmesa6 \
  && apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Install base python3 packages
COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install --no-cache-dir -r requirements.txt

# GitLFS
RUN apt-get update && apt-get install -yq --no-install-recommends \
  git-lfs \
  && apt-get clean && \
  rm -rf /var/lib/apt/lists/*

## Set ENV for dm_control
ENV MUJOCO_GL osmesa
ENV PYOPENGL_PLATFORM osmesa


# Configure container startup
EXPOSE 2980
USER cpx0
ENTRYPOINT ["tini", "--"]
CMD ["sh","-c", "jupyter lab --notebook-dir=/home/${NB_USER} --ip=0.0.0.0 --no-browser --allow-root --port=2980 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]
