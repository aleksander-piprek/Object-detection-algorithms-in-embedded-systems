#!/bin/bash


### This script sets up the environment for building and running the project.
### It is specific for x86-64 architecture on WSL2 with Ubuntu 24.04.


# Make me executable!
# chmod +x setup.sh


# Install dependencies (some may be missing or already installed)
sudo apt-get update
sudo apt-get upgrade -y

sudo apt-get install -y cmake g++ build-essential clang-tidy gcc-12 g++-12
sudo apt-get install -y libgtk2.0-dev pkg-config libgtk-3-0 libgail-common libatk-adaptor libgtk-3-common gtk2-engines-murrine gtk2-engines-pixbuf \
                        libcanberra-gtk-module libcanberra-gtk3-module ffmpeg libavcodec-dev libavformat-dev libswscale-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev                  

###
### CUDA 12.9
###
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda-repo-ubuntu2404-12-9-local_12.9.1-575.57.08-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-9-local_12.9.1-575.57.08-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9

###
### CUDNN 9.10.2
###
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cudnn9-cuda-12

# If you don't have propriety NVIDIA drivers installed, you can install them with:
# sudo ubuntu-drivers autoinstall
# or 
# sudo apt-get -y install cuda-drivers

###
### OpenCV 5.x
###
# Currently this cmake command is set up for CUDA 12.9 and CUDNN 9.10.2 working on WSL2 Ubuntu 24.04.
# If you are using a different version of CUDA or CUDNN, you may need to adjust the paths and versions accordingly.

mkdir -p lib/opencv/build && cd lib/opencv/build
cmake ../../opencv \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=../install \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DWITH_QT=OFF \
  -DWITH_GTK=ON \
  -DWITH_OPENGL=OFF \
  -DWITH_VTK=OFF \
  -DWITH_CUDA=ON \
  -DWITH_CUDNN=ON \
  -DWITH_GSTREAMER=ON \
  -DWITH_FFMPEG=ON \
  -DCUDNN_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
  -DCUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so \
  -DCUDA_ARCH_BIN="8.6" \
  -DCUDA_ARCH_PTX="" \
  -DOPENCV_DNN_CUDA=ON \
  -DOPENCV_ENABLE_NONFREE=ON \
  -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
  -DBUILD_opencv_python3=OFF \
  -DOPENCV_GENERATE_PKGCONFIG=ON \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12
make -j$(nproc)
make install

###
### Google Test
###
cd ../../..
mkdir -p lib/gtest/build && cd lib/gtest/build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=../install 

make -j$(nproc)
make install