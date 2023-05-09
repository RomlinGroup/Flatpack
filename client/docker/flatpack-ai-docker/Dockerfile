FROM nvidia/cuda:12.1.1-base-ubuntu22.04
LABEL version="0.7"
RUN mkdir /home/flatpack-ai
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y build-essential git python3
CMD tail -f /dev/null