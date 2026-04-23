FROM nvcr.io/nvidia/pytorch:23.12-py3 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

RUN set -eux; \
    apt-get update; \
    echo 'tzdata tzdata/Areas select America' | debconf-set-selections; \
    echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections; \
    apt-get install -y --no-install-recommends \
        htop \
        mc \
        tmux \
        python3-tk \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Build context must be the repo root so setup/requirements.txt is available.
COPY setup/requirements.txt /tmp/trajflow-requirements.txt
RUN pip install --no-cache-dir -r /tmp/trajflow-requirements.txt


FROM base AS dev

ARG USER_NAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN set -eux; \
    if getent passwd "${USER_NAME}" > /dev/null; then \
        usermod -aG video "${USER_NAME}"; \
    else \
        if ! getent group "${USER_GID}" > /dev/null; then \
            groupadd --gid "${USER_GID}" "${USER_NAME}"; \
        fi; \
        useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home --no-log-init --shell /bin/bash "${USER_NAME}"; \
        usermod -aG video "${USER_NAME}"; \
    fi

RUN set -ex; \
    apt-get update && apt-get install -y --no-install-recommends \
        sudo; \
    echo "$USER_NAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USER_NAME; \
    chmod 0440 /etc/sudoers.d/$USER_NAME;

WORKDIR /workspace/
