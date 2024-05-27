#!/bin/bash
set -e
set -u

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <flatpack_name>"
  exit 1
fi

FLATPACK_NAME=$(echo $1 | tr '[:upper:]' '[:lower:]' | sed 's/ /-/g')

if [[ ! "$FLATPACK_NAME" =~ ^[a-z0-9-]+$ ]]; then
  echo "Error: Invalid name format. Only lowercase letters, numbers, and hyphens are allowed."
  exit 1
fi

cd ../..

if [ ! -d "warehouse/template" ]; then
  echo "Error: warehouse/template directory not found."
  exit 1
fi

mkdir -p "warehouse/$FLATPACK_NAME"

cp -r warehouse/template/* "warehouse/$FLATPACK_NAME/"

sed -i '' "s/# template/# $FLATPACK_NAME/g" "warehouse/$FLATPACK_NAME/README.md"
sed -i '' "s/{{model_name}}/$FLATPACK_NAME/g" "warehouse/$FLATPACK_NAME/flatpack.toml"
sed -i '' "s/export DEFAULT_REPO_NAME=template/export DEFAULT_REPO_NAME=$FLATPACK_NAME/g" "warehouse/$FLATPACK_NAME/build.sh"
sed -i '' "s/export FLATPACK_NAME=template/export FLATPACK_NAME=$FLATPACK_NAME/g" "warehouse/$FLATPACK_NAME/build.sh"

echo "Contents of warehouse/template copied to warehouse/$FLATPACK_NAME."
