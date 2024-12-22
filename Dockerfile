# Ubuntu 24.04 LTS (Noble Numbat)
# End Of Legacy Support: April 2036
# https://ubuntu.com/about/release-cycle

FROM ubuntu:24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

ENV NVM_DIR=/home/flatpackuser/.nvm \
    NODE_VERSION=22 \
    NODE_PATH=/home/flatpackuser/.nvm/v22/lib/node_modules \
    PATH=/home/flatpackuser/.nvm/versions/node/v22/bin:/home/flatpackuser/.local/bin:${PATH}

LABEL authors="flatpack"

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
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
    unzip \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash -u 1001 flatpackuser
USER flatpackuser
WORKDIR /home/flatpackuser

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash && \
    . ${NVM_DIR}/nvm.sh && \
    nvm install ${NODE_VERSION} && \
    nvm use ${NODE_VERSION} && \
    nvm alias default ${NODE_VERSION}

RUN pipx install flatpack

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV NVM_DIR=/home/flatpackuser/.nvm \
    NODE_VERSION=22 \
    NODE_PATH=/home/flatpackuser/.nvm/v22/lib/node_modules \
    PATH=/home/flatpackuser/.nvm/versions/node/v22/bin:/home/flatpackuser/.local/bin:${PATH}

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    curl \
    libcurl4-openssl-dev \
    procps \
    sox && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash -u 1001 flatpackuser

COPY --from=builder /home/flatpackuser /home/flatpackuser
COPY --from=builder ${NVM_DIR} ${NVM_DIR}

USER flatpackuser
WORKDIR /home/flatpackuser

RUN echo '#!/bin/bash\n\
source ${HOME}/.nvm/nvm.sh\n\
exec "$@"' > /home/flatpackuser/docker-entrypoint.sh && \
    chmod +x /home/flatpackuser/docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/home/flatpackuser/docker-entrypoint.sh"]
CMD ["bash"]