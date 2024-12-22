# Ubuntu 24.04 LTS (Noble Numbat)
# End Of Legacy Support: April 2036
# https://ubuntu.com/about/release-cycle

FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive
ENV NVM_DIR=/home/flatpackuser/.nvm
ENV NODE_VERSION=22
ENV NODE_PATH=$NVM_DIR/v$NODE_VERSION/lib/node_modules
ENV PATH=$NVM_DIR/versions/node/v$NODE_VERSION/bin:/home/flatpackuser/.local/bin:$PATH
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
    procps \
    python3-dev \
    python3-full \
    python3-pip \
    pipx \
    sox \
    unzip \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash -u 1001 flatpackuser && \
    chown -R flatpackuser:flatpackuser /home/flatpackuser && \
    chmod 755 /home/flatpackuser

USER flatpackuser
WORKDIR /home/flatpackuser

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash && \
    . $NVM_DIR/nvm.sh && \
    nvm install $NODE_VERSION && \
    nvm use $NODE_VERSION && \
    nvm alias default $NODE_VERSION && \
    chmod 700 $NVM_DIR

RUN pipx install flatpack

RUN echo '#!/bin/bash\n\
source $HOME/.nvm/nvm.sh\n\
exec "$@"' > /home/flatpackuser/docker-entrypoint.sh && \
    chmod +x /home/flatpackuser/docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/home/flatpackuser/docker-entrypoint.sh"]
CMD ["bash"]