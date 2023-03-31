#!/bin/bash
docker run --gpus all --rm -it --net host --ipc host\
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
    ur3e-hande-real:noetic bash
