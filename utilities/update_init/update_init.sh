#!/bin/bash
set -e
set -u

BASE_DIR="$(dirname $(dirname $(dirname $(realpath $0))))"
SOURCE_FILE="${BASE_DIR}/warehouse/template/init.sh"

if [[ ! -f ${SOURCE_FILE} ]]; then
  echo "Error: Source file ${SOURCE_FILE} does not exist."
  exit 1
fi

for dir in ${BASE_DIR}/warehouse/*/; do
  if [[ "${dir}" != "${BASE_DIR}/warehouse/template/" ]]; then
    echo "Copying ${SOURCE_FILE} to ${dir}"
    cp "${SOURCE_FILE}" "${dir}"
  fi
done

echo "Update completed."
