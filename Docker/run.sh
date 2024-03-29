#!/bin/bash
docker run --gpus all --rm -it --net host --ipc host --privileged\
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
    -v $HOME/ur3e_hande_real:/root/ur3e_hande_real \
    -v $HOME/learning_data:/root/learning_data \
    ur3e-hande-real:noetic bash
