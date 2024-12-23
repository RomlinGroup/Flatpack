# Ubuntu 24.04 LTS (Noble Numbat)
# End Of Legacy Support: April 2036
# https://ubuntu.com/about/release-cycle

FROM ubuntu:24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

ENV NVM_DIR=/home/flatpackuser/.nvm \
    NODE_VERSION=22 \
    NODE_PATH=/home/flatpackuser/.nvm/v22/lib/node_modules \
    PATH=/home/flatpackuser/.nvm/versions/node/v22/bin:/home/flatpackuser/.local/bin:${PATH}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    file \
    git \
    libcurl4-openssl-dev \
    pipx \
    python3-dev \
    python3-full \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash -u 1001 flatpackuser && \
    chown -R flatpackuser:flatpackuser /home/flatpackuser

USER flatpackuser
WORKDIR /home/flatpackuser

RUN mkdir -p ${NVM_DIR} && \
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash && \
    . ${NVM_DIR}/nvm.sh && \
    nvm install ${NODE_VERSION} && \
    nvm use ${NODE_VERSION} && \
    nvm alias default ${NODE_VERSION} && \
    chmod 700 ${NVM_DIR}

RUN pipx install flatpack

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV NVM_DIR=/home/flatpackuser/.nvm \
    NODE_VERSION=22 \
    NODE_PATH=/home/flatpackuser/.nvm/v22/lib/node_modules \
    PATH=/home/flatpackuser/.nvm/versions/node/v22/bin:/home/flatpackuser/.local/bin:${PATH}

LABEL authors="flatpack"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    git \
    jq \
    procps \
    python3 \
    sox \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash -u 1001 flatpackuser && \
    chown -R flatpackuser:flatpackuser /home/flatpackuser

COPY --from=builder --chown=flatpackuser /home/flatpackuser/.local /home/flatpackuser/.local
COPY --from=builder --chown=flatpackuser /home/flatpackuser/.nvm /home/flatpackuser/.nvm

RUN mkdir -p /home/flatpackuser/.cache /home/flatpackuser/.config && \
    chown -R flatpackuser:flatpackuser /home/flatpackuser/.cache /home/flatpackuser/.config

USER flatpackuser
WORKDIR /home/flatpackuser

RUN echo '#!/bin/bash\nsource ${HOME}/.nvm/nvm.sh\nexec "$@"' > /home/flatpackuser/docker-entrypoint.sh && \
    chmod +x /home/flatpackuser/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/home/flatpackuser/docker-entrypoint.sh"]
CMD ["bash"]