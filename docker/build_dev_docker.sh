#!/bin/bash

: ${IMAGE_NAME:="trajflow-dev-${USER}"}
: ${IMAGE_TAG:=latest}
: ${IMAGE_FULL_NAME:="${IMAGE_NAME}:${IMAGE_TAG}"}

DEV_DOCKER_DIR=$(dirname "$(realpath -s "$0")")
BUILD_CONTEXT=$(realpath "${DEV_DOCKER_DIR}/..")

echo "Building image   : ${IMAGE_FULL_NAME}"
echo "Build context    : ${BUILD_CONTEXT}"

docker build \
    -t "${IMAGE_FULL_NAME}" \
    --network=host \
    -f "${DEV_DOCKER_DIR}/trajflow_23_12.Dockerfile" \
    --target dev \
    "${BUILD_CONTEXT}/"
