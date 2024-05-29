#!/bin/bash

source venv/bin/activate

flatpack version

echo "👋 Welcome, brave explorer!"

./network.sh

while :; do
  sleep 3600
done
