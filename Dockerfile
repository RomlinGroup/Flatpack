FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    FORCE_COLOR=1 \
    NODE_VERSION=22 \
    NODE_PATH=/home/flatpackuser/.nvm/v22/lib/node_modules \
    NVM_DIR=/home/flatpackuser/.nvm \
    PATH=/home/flatpackuser/.nvm/versions/node/v22/bin:/home/flatpackuser/.local/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    TERM=xterm-256color

LABEL authors="flatpack"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        file \
        git \
        jq \
        libbz2-dev \
        libcurl4-openssl-dev \
        libffi-dev \
        libgdbm-dev \
        liblzma-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        procps \
        python3-dev \
        python3-full \
        python3-pip \
        sox \
        wget \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/* && \
    wget https://www.python.org/ftp/python/3.12.8/Python-3.12.8.tgz && \
    tar xvf Python-3.12.8.tgz && \
    cd Python-3.12.8 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.12.8 Python-3.12.8.tgz && \
    ln -sf /usr/local/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3.12 /usr/bin/pip3 && \
    useradd -m -s /bin/bash -u 1001 flatpackuser && \
    chown -R flatpackuser:flatpackuser /home/flatpackuser

USER flatpackuser

WORKDIR /home/flatpackuser

RUN mkdir -p ${NVM_DIR} && \
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash && \
    . ${NVM_DIR}/nvm.sh && \
    nvm install ${NODE_VERSION} && \
    nvm use ${NODE_VERSION} && \
    nvm alias default ${NODE_VERSION} && \
    chmod 700 ${NVM_DIR} && \
    pip3 install --upgrade pip setuptools wheel && \
    pip3 install flatpack

EXPOSE 3000 8000