FROM nvidia/cuda:12.1.1-base-ubuntu22.04

LABEL version="1.0"

RUN useradd -ms /bin/bash flatpack

WORKDIR /home/flatpack

USER root

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y build-essential git python3 && \
    rm -rf /var/lib/apt/lists/*

USER flatpack

RUN echo "Welcome, you are now logged in as flatpack"

CMD tail -f /dev/null