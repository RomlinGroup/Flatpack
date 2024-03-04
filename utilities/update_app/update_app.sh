#!/bin/bash
set -e
set -u

BASE_DIR="$(dirname $(dirname $(dirname $(realpath $0))))"
SOURCE_DIR="${BASE_DIR}/warehouse/template/app"

if [[ ! -d ${SOURCE_DIR} ]]; then
  echo "Error: Source directory ${SOURCE_DIR} does not exist."
  exit 1
fi

for dir in ${BASE_DIR}/warehouse/*/; do
  if [[ "${dir}" != "${BASE_DIR}/warehouse/template/" ]]; then
    echo "Copying ${SOURCE_DIR} to ${dir}"
    cp -r "${SOURCE_DIR}" "${dir}"
  fi
done

echo "Update completed."
