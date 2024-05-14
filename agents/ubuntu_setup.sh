#!/bin/bash

# Add the deadsnakes PPA, update package list, and install necessary packages
sudo add-apt-repository ppa:deadsnakes/ppa -y && \
sudo apt-get update && \
sudo apt-get install -y build-essential python3.11 python3.11-dev python3.11-venv && \
curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.11 && \
pip3.11 install flatpack --no-cache-dir