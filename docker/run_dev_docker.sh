#!/bin/bash

# Env vars.
: ${IMAGE_NAME:="trajflow-dev-${USER}"}
# Default values
DEFAULT_IMAGE_TAG="latest"
DEFAULT_USER_NAME="ubuntu"
DEFAULT_CONTAINER_NAME="trajflow"
DEFAULT_HOST_DATA_DIR="/data/"
DEFAULT_CONTAINER_DATA_DIR="/data/"

# Set variables with defaults
: ${IMAGE_TAG:=${DEFAULT_IMAGE_TAG}}
: ${USER_NAME:=${DEFAULT_USER_NAME}}
: ${CONTAINER_NAME:=${DEFAULT_CONTAINER_NAME}}
: ${HOST_DATA_DIR:=${DEFAULT_HOST_DATA_DIR}}
: ${CONTAINER_DATA_DIR:=${DEFAULT_CONTAINER_DATA_DIR}}

echo "Container name    : ${CONTAINER_NAME}"
echo "Host data dir     : ${HOST_DATA_DIR}"
echo "Container data dir: ${CONTAINER_DATA_DIR}"

CONTAINER_ID=`docker ps -aqf "name=^/${CONTAINER_NAME}$"`
if [ -z "${CONTAINER_ID}" ]; then
    echo "Creating new ${CONTAINER_NAME} container."
    docker run -it -d                                    \
        --gpus 'all,"capabilities=compute,utility,graphics,display,video"'      \
        --network=host                                   \
        --ipc=host                                       \
        --cap-add=SYS_PTRACE                             \
        -v /dev/shm:/dev/shm                             \
        -v ${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}:rw     \
        -v /tmp/.X11-unix:/tmp/.X11-unix                 \
        -v /etc/localtime:/etc/localtime:ro              \
        -e DISPLAY=unix${DISPLAY}                        \
        -e NEW_UID=$(id -u)                              \
        -e NEW_GID=$(id -g)                              \
        --ulimit memlock=-1                              \
        --ulimit stack=67108864                          \
        --name=${CONTAINER_NAME}                         \
        ${IMAGE_NAME}:${IMAGE_TAG}

    CONTAINER_ID=`docker ps -aqf "name=^/${CONTAINER_NAME}$"`
else
    echo "Found ${CONTAINER_NAME} container: ${CONTAINER_ID}."
    # Check if the container is already running and start if necessary.
    if [ -z `docker ps -qf "name=^/${CONTAINER_NAME}$"` ]; then
        echo "Starting ${CONTAINER_NAME} container..."
        docker start ${CONTAINER_ID}
    fi
fi

xhost +local:${HOSTNAME}
docker exec --user ${USER_NAME} -it ${CONTAINER_ID} bash
xhost -local:${HOSTNAME}
