# Ubuntu 24.04 LTS (Noble Numbat)
# End Of Legacy Support: April 2036
# https://ubuntu.com/about/release-cycle

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ENV FORCE_COLOR=1 \
    NODE_PATH=/home/flatpackuser/.nvm/v22/lib/node_modules \
    NODE_VERSION=22 \
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
    libcurl4-openssl-dev \
    pipx \
    procps \
    python3-dev \
    python3-full \
    python3-pip \
    sox \
    wget \
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

RUN mkdir -p /home/flatpackuser/.cache /home/flatpackuser/.config && \
    echo '#!/bin/bash\nsource ${HOME}/.nvm/nvm.sh\nexec "$@"' > /home/flatpackuser/docker-entrypoint.sh && \
    chmod +x /home/flatpackuser/docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/home/flatpackuser/docker-entrypoint.sh"]
CMD ["bash"]