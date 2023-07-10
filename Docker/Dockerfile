FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
# FROM osrf/ros:noetic-desktop-full

WORKDIR /root/

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV DEBIAN_FRONTEND=noninteractive

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


########################
### INSTALL PACKAGES ###
########################
RUN apt-get update && apt-get install -y \
    sudo \
    vim \
    wget \
    unzip \
    git \
    iputils-ping \
	net-tools \
    curl \
	init \
    build-essential \
    gcc \
    g++ \
    gnupg \
    make \
    libtool \
    texinfo \
    dpkg-dev \
    pkg-config \
    libgl1-mesa-glx \
    libcurl4-openssl-dev \
    libssl-dev 


######################
### INSTALL Noetic ###
######################
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
RUN apt-get update && apt-get install -y ros-noetic-desktop
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash"


#####################
### INSTALL CMAKE ###
#####################
# Reference: https://apt.kitware.com/
RUN git clone https://gitlab.kitware.com/cmake/cmake.git \
    && cd cmake \
    && git checkout tags/v3.16.3 \
    && ./bootstrap --parallel=8 \
    && make -j8 \
    && make install \
    && cd .. \
    && rm -rf cmake


###########################################
### INSTALL Universal_Robots_ROS_Driver ###
###########################################
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-rospkg \
    python3-catkin-pkg \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-rosdep \
    python3-catkin-tools \
    python3-setuptools \
    python3-tk \
    python3-pip \
    ros-noetic-moveit-commander \
    ros-noetic-universal-robots \
    build-essential

RUN /bin/bash -c "source /opt/ros/noetic/setup.bash"
RUN mkdir -p /root/catkin_ws/src
RUN cd /root/catkin_ws && \
    git clone https://github.com/UniversalRobots/Universal_Robots_ROS_Driver.git src/Universal_Robots_ROS_Driver && \
    git clone https://github.com/ros-planning/moveit_calibration.git src/moveit_calibration


##################  
### EXTRA DEPS ###
##################
RUN pip3 install catkin_pkg \
    && pip3 install "pybind11[global]" \
    && pip3 install pyyaml \
    && pip3 install empy \
    && wget -q --show-progress --progress=bar:force https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz \
    && tar -xzf eigen-3.3.7.tar.gz \
    && cd eigen-3.3.7 \
    && mkdir build && cd build \
    && cmake .. \
    && make install \
    && cd ../.. \
    && rm eigen-3.3.7.tar.gz


#########################
### INSTALL REALSENSE ###
#########################
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt-get update \
    && apt-get install -y software-properties-common

RUN apt-get update \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE \
    && add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u \
    && apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 6F3EFCDE \
    && apt-get update \
    # Install the librealsense, Development packages & and other needed packages
    && apt-get install -y \
    librealsense2-dkms \
    librealsense2-utils \
    librealsense2-dev  \
    librealsense2-dbg \
    v4l-utils \
    # Upgrade the local packages 
    && apt-get update && apt-get --only-upgrade install -y librealsense2-utils librealsense2-dkms ros-noetic-realsense2-camera ros-noetic-ddynamic-reconfigure
RUN cd /root/catkin_ws && \
    git clone -b ros1-legacy https://github.com/IntelRealSense/realsense-ros.git src/realsense-ros && \
    git clone https://github.com/ros-drivers/rgbd_launch.git src/rgbd_launch


######################
### INSTALL ur pkg ###
######################
# RUN cd /root/catkin_ws \
#     && git clone -b noetic-real https://github.com/shumpe-m/ur3.git src/ros_ur3 \
#     && git clone -b main https://github.com/shumpe-m/ur3e_hande_real.git src/ur3e_hande_real
RUN cd /root/catkin_ws \
    && git clone -b noetic-real https://github.com/shumpe-m/ur3.git src/ros_ur3
COPY . /root/catkin_ws/src/ur3e_hande_real/

RUN rm -rf /etc/ros/rosdep/sources.list.d/20-default.list
RUN cd /root/catkin_ws \
    && rosinstall /root/catkin_ws/src /opt/ros/noetic src/ros_ur3/dependencies.rosinstall \
    && apt-get update -qq \
    && rosdep init && rosdep update && rosdep install --from-paths src --ignore-src --rosdistro=noetic -y \
    && /bin/bash -c "source /opt/ros/noetic/setup.bash; cd /root/catkin_ws; catkin_make" \
    && echo "source /root/catkin_ws/devel/setup.bash" >> ~/.bashrc \
    && echo "export ROS_PACKAGE_PATH=\${ROS_PACKAGE_PATH}:/root/catkin_ws" >> ~/.bashrc \
    && echo "export ROS_WORKSPACE=/root/catkin_ws" >> ~/.bashrc \
    && echo "chmod -R u+x /root/catkin_ws/src/ur3e_hande_real/ur_control_scripts/scripts" >> ~/.bashrc


#####################
### INSTALL torch ###
#####################

RUN pip3 install \
    torch==1.10.0+cu113 \
    torchvision==0.11.1+cu113 \
    torchaudio==0.10.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html 
RUN pip3 install \
    numpy \
    opencv-python \
    matplotlib \
    tensorboard \
    scikit-learn \
    seaborn \
    tqdm \
    loguru

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "cd /root/catkin_ws" >> ~/.bashrc