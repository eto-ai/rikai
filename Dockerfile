# ------------------------- 1. builder -------------------------
FROM ubuntu:20.04 AS builder
LABEL stage=builder
LABEL maintainer="rikai developers<rikai-dev@eto.ai>"
LABEL description="Demo Image"

ARG NB_USER="eto"
ARG NB_UID="1000"
ARG SCALA_VERSION="2.12"
ARG RIKAI_VERSION="0.0.19"

# ---------------------------
# Install system dependencies
# ---------------------------

# `apt list --installed` to list all installed packages
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -qy --no-install-recommends \
    autoconf \
    automake \
    build-essential \
    ca-certificates \
    curl \
    fontconfig \
    git \
    jq \
    libsnappy-dev \
    libtool \
    locales \
    lsof \
    openjdk-11-jdk \
    sudo \
    unzip \
    vim \
    wget \
    && \
    fc-cache -f && \
    rm -rf /var/lib/apt/lists/*

# -------------------------
# System env and user setup
# -------------------------
ENV SHELL=/bin/bash \
    NB_UID=$NB_UID \
    NB_USER=$NB_USER \
    USER=$NB_USER \
    HOME=/home/$NB_USER \
    NPM_DIR=/opt/npm \
    CONDA_DIR=/opt/conda

ARG REPO_DIR=${HOME}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER} && \
    echo "${NB_USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    mkdir -p $HOME/.ssh

ENV NPM_CONFIG_GLOBALCONFIG=${NPM_DIR}/npmrc \
    NB_PYTHON_PREFIX=$CONDA_DIR \
    KERNEL_PYTHON_PREFIX=$CONDA_DIR \
    PATH=$CONDA_DIR/bin:$NPM_DIR/bin:$PATH \
    REPO_DIR=${REPO_DIR}

# -----------------------
# Conda environment setup
# -----------------------
COPY environment.yml $HOME/environment.yml

RUN mkdir -p "$CONDA_DIR" && \
    wget -q "https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O miniconda.sh && \
    bash miniconda.sh -f -b -p "$CONDA_DIR" && \
    echo "export PATH=\$PATH" > /etc/profile.d/conda.sh && \
    rm miniconda.sh && \
    conda update --all --yes && \
    conda config --system --add channels conda-forge && \
    conda config --system --add channels pytorch && \
    conda config --system --set auto_update_conda false && \
    conda config --system --set show_channel_urls true && \
    echo 'update_dependencies: false' >> ${CONDA_DIR}/.condarc && \
    conda env update -n root -f $HOME/environment.yml && \
    conda clean -tipy && \
    echo '' > ${CONDA_DIR}/conda-meta/history && \
    chown -R $NB_USER:$NB_USER ${CONDA_DIR} && \
    mkdir -p "$CONDA_DIR/locks" && \
    chmod 777 "$CONDA_DIR/locks" && \
    conda list && \
    mkdir -p $NPM_DIR && \
    chown -R $NB_USER:$NB_USER $NPM_DIR ${REPO_DIR} && \
    conda list '^(conda|tini)$' --export | grep -v '^#' | cut -d= -f1,2 | tr '=' ' ' >> $CONDA_DIR/conda-meta/pinned

# --------
# Spark
# --------
ARG spark_version="3.1.2"
ARG hadoop_version="3.2"
ARG spark_checksum="2385CB772F21B014CE2ABD6B8F5E815721580D6E8BC42A26D70BBCDDA8D303D886A6F12B36D40F6971B5547B70FAE62B5A96146F0421CB93D4E51491308EF5D5"
ARG py4j_version="0.10.9"

ENV APACHE_SPARK_VERSION="${spark_version}" \
    HADOOP_VERSION="${hadoop_version}"

# Spark installation
WORKDIR /tmp

# Using the preferred mirror to download Spark
# hadolint ignore=SC2046

RUN wget -q $(wget -qO- https://www.apache.org/dyn/closer.lua/spark/spark-${APACHE_SPARK_VERSION}/spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz\?as_json | \
    python -c "import sys, json; content=json.load(sys.stdin); print(content['preferred']+content['path_info'])") && \
    echo "${spark_checksum} *spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" | sha512sum -c - && \
    tar xzf "spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" -C /usr/local --owner root --group root --no-same-owner && \
    rm "spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz"

WORKDIR /usr/local
RUN ln -s "spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}" spark

# Configure Spark
ENV SPARK_HOME=/usr/local/spark
ENV PYTHONPATH="${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-${py4j_version}-src.zip" \
    SPARK_OPTS="--driver-java-options=-Xms1024M --driver-java-options=-Xmx4096M --driver-java-options=-Dlog4j.logLevel=info" \
    PATH=$PATH:$SPARK_HOME/bin

# copy spark-defaults.conf
COPY ./conf/spark-defaults.conf $SPARK_HOME/conf/spark-defaults.conf

# -------------
# Install Rikai
# -------------
ENV SCALA_VERSION=${SCALA_VERSION}
ENV RIKAI_VERSION=${RIKAI_VERSION}

USER $NB_UID
RUN pip install rikai[all]==${RIKAI_VERSION}
USER root

# -----------------
# Install notebooks
# -----------------
RUN mkdir -p $HOME/rikai/notebooks && \
    chmod 777 $HOME/rikai/notebooks && \
    chown -R $NB_USER:$NB_USER $HOME/rikai/notebooks

COPY ./notebooks/*.ipynb $HOME/rikai/notebooks/
COPY ./notebooks/wordcount.txt $HOME/rikai/notebooks/

# ----------
# User setup
# ----------
USER $NB_UID
WORKDIR ${REPO_DIR}
ENV PATH=${HOME}/.local/bin:${REPO_DIR}/.local/bin:${PATH}

# -----------------------------
# clean up clean up
# we all had fun today
# now it's time to clean up and
# put everything away
# -----------------------------
RUN conda clean -tipy && \
    rm -rf \
    $HOME/environment.yml \
    $HOME/.cache \
    $CONDA_DIR/pkgs/

#RUN python -c "import torch;torch.cuda.get_device_name(0)"

# ------------------------- 3. Remove intermediates -------------------------
FROM ubuntu:20.04 AS output
LABEL maintainer="rikai developers<rikai-dev@eto.ai>"
LABEL description="Demo Image"

ARG NB_USER="eto"
ARG NB_UID="1000"

# necessary tools
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -qy --no-install-recommends \
    autoconf \
    automake \
    build-essential \
    ca-certificates \
    curl \
    fontconfig \
    git \
    jq \
    libsnappy-dev \
    libtool \
    locales \
    lsof \
    openjdk-11-jdk \
    python-dev \
    sudo \
    unzip \
    vim \
    wget \
    && \
    fc-cache -f && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# create new user with sudo permission
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER} && \
    echo "${NB_USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    mkdir -p $HOME/.ssh && \
    rm -rf /opt/conda /opt/npm

ENV SHELL=/bin/bash \
    NB_USER=$NB_USER \
    NB_UID=$NB_UID \
    USER=$NB_USER \
    HOME=/home/$NB_USER \
    NPM_DIR=/opt/npm \
    CONDA_DIR=/opt/conda

ENV NPM_CONFIG_GLOBALCONFIG=${NPM_DIR}/npmrc \
    NB_PYTHON_PREFIX=$CONDA_DIR \
    KERNEL_PYTHON_PREFIX=$CONDA_DIR \
    PATH=$CONDA_DIR/bin:$NPM_DIR/bin:${HOME}/.local/bin:$PATH

COPY --from=builder --chown=eto:eto /opt/npm /opt/npm
COPY --from=builder --chown=eto:eto /opt/conda /opt/conda
COPY --from=builder --chown=eto:eto /home/$NB_USER /home/$NB_USER


# Spark

COPY --from=builder $SPARK_HOME $SPARK_HOME

USER $NB_UID
ENV SPARK_HOME=/usr/local/spark
ENV PYTHONPATH="${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-${py4j_version}-src.zip" \
    SPARK_OPTS="--driver-java-options=-Xms1024M --driver-java-options=-Xmx4096M --driver-java-options=-Dlog4j.logLevel=info" \
    PATH=$PATH:$SPARK_HOME/bin

# ------------------
# Launch Jupyter Lab
# ------------------

WORKDIR ${HOME}/rikai/notebooks
EXPOSE 8888
ENTRYPOINT ["tini", "-g", "--"]
CMD ["jupyter", "lab", "--ip", "0.0.0.0"]
