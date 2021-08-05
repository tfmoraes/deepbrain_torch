# /usr/bin/env bash

export DOCKER_UID=$(id -u)
export DOCKER_GID=$(id -g)

docker build --build-arg UID=$DOCKER_UID --build-arg GID=$DOCKER_GID -t unet_brain_tfmoraes .
