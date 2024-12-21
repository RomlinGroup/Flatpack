# Ubuntu 24.04 LTS
# EOL: April 2036
# https://ubuntu.com/about/release-cycle
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ENV NVM_DIR=/home/flatpackuser/.nvm
ENV NODE_PATH=$NVM_DIR/v$NODE_VERSION/lib/node_modules
ENV NODE_VERSION=22
ENV PATH=$NVM_DIR/versions/node/v$NODE_VERSION/bin:/home/flatpackuser/.local/bin:$PATH
ENV VIRTUAL_ENV=/home/flatpackuser/.venv

LABEL authors="flatpack"

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    apparmor \
    apparmor-utils \
    build-essential \
    cmake \
    curl \
    git \
    procps \
    python3-dev \
    python3-full \
    python3-pip \
    python3-venv \
    pipx \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash -u 1001 flatpackuser && \
    chown -R flatpackuser:flatpackuser /home/flatpackuser

USER flatpackuser
WORKDIR /home/flatpackuser

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash && \
    . $NVM_DIR/nvm.sh && \
    nvm install $NODE_VERSION && \
    nvm use $NODE_VERSION && \
    nvm alias default $NODE_VERSION

RUN pipx install flatpack

ENV PATH=/home/flatpackuser/.local/bin:$PATH

RUN echo '#!/bin/bash\n\
source $HOME/.nvm/nvm.sh\n\
exec "$@"' > /home/flatpackuser/docker-entrypoint.sh && \
chmod +x /home/flatpackuser/docker-entrypoint.sh

ENTRYPOINT ["/home/flatpackuser/docker-entrypoint.sh"]
CMD ["bash"]