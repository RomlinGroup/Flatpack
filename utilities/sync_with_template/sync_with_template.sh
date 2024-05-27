#!/bin/bash
set -e
set -u

BASE_DIR="$(dirname $(dirname $(dirname $(realpath $0))))"
DEVICE_SOURCE_FILE="${BASE_DIR}/warehouse/template/device.sh"
INIT_SOURCE_FILE="${BASE_DIR}/warehouse/template/init.sh"

if [[ ! -f ${DEVICE_SOURCE_FILE} ]]; then
  echo "Error: Source file ${DEVICE_SOURCE_FILE} does not exist."
  exit 1
fi

if [[ ! -f ${INIT_SOURCE_FILE} ]]; then
  echo "Error: Source file ${INIT_SOURCE_FILE} does not exist."
  exit 1
fi

for dir in ${BASE_DIR}/warehouse/*/; do
  if [[ "${dir}" != "${BASE_DIR}/warehouse/template/" ]]; then
    echo "Copying ${DEVICE_SOURCE_FILE} to ${dir}"
    cp "${DEVICE_SOURCE_FILE}" "${dir}"
    echo "Copying ${INIT_SOURCE_FILE} to ${dir}"
    cp "${INIT_SOURCE_FILE}" "${dir}"
  fi
done

echo "Update completed."
