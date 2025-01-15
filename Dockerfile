FROM ubuntu:25.04

ENV FORCE_COLOR=1 \
    NODE_VERSION=22 \
    NODE_PATH=/home/flatpackuser/.nvm/v22/lib/node_modules \
    NVM_DIR=/home/flatpackuser/.nvm \
    PATH=/home/flatpackuser/.nvm/versions/node/v22/bin:/home/flatpackuser/.local/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    TERM=xterm-256color

LABEL authors="flatpack"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential && \
    rm -rf /var/lib/apt/lists/* && \
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
    pip install --no-cache-dir flatpack

EXPOSE 3000 8000