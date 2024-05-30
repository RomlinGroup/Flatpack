FROM ubuntu:22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    dnsutils \
    git \
    iproute2 \
    iptables \
    iputils-ping \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv venv

RUN . venv/bin/activate && pip install flatpack

COPY custom.sh .
COPY network.sh .

RUN chmod +x custom.sh network.sh

CMD ["sh", "-c", ". venv/bin/activate && ./custom.sh"]
