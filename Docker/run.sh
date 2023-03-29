#!/bin/bash
docker run --gpus all --rm -it --net host --ipc host\
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
    -v $HOME/FT_300s_ur3e:/root/FT_300s_ur3e \
    ur3e-hande-real:noetic bash
