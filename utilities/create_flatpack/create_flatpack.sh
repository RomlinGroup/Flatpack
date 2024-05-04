#!/bin/bash
set -e
set -u

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 MY_FLATPACK_NAME"
  exit 1
fi

# Name of the new folder (converted to lowercase and spaces replaced with hyphens)
FLATPACK_NAME=$(echo $1 | tr '[:upper:]' '[:lower:]' | sed 's/ /-/g')

# Navigate to the root directory of your project
cd ../..

# Check if the warehouse/template directory exists
if [ ! -d "warehouse/template" ]; then
  echo "Error: warehouse/template directory not found."
  exit 1
fi

# Create the new directory if it doesn't exist
mkdir -p "warehouse/$FLATPACK_NAME"

# Copy the contents of the warehouse/template directory to the new directory
cp -r warehouse/template/* "warehouse/$FLATPACK_NAME/"

sed -i '' "s/# template/# $FLATPACK_NAME/g" "warehouse/$FLATPACK_NAME/README.md"
sed -i '' "s/{{model_name}}/$FLATPACK_NAME/g" "warehouse/$FLATPACK_NAME/flatpack.toml"
sed -i '' "s/export DEFAULT_REPO_NAME=template/export DEFAULT_REPO_NAME=$FLATPACK_NAME/g" "warehouse/$FLATPACK_NAME/build.sh"
sed -i '' "s/export FLATPACK_NAME=template/export FLATPACK_NAME=$FLATPACK_NAME/g" "warehouse/$FLATPACK_NAME/build.sh"

echo "Contents of warehouse/template copied to warehouse/$FLATPACK_NAME."
