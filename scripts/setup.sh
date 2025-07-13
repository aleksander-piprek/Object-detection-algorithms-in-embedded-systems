#!/bin/bash


### This script sets up the environment for building and running the project.


# Make me executable!
# chmod +x setup.sh



# Install dependencies (some may be missing or already installed)
sudo apt-get update
sudo apt-get upgrade -y

sudo apt-get install -y cmake g++ build-essential clang-tidy
sudo apt-get install -y libgtk2.0-dev pkg-config libgtk-3-0 libgail-common libatk-adaptor libgtk-3-common gtk2-engines-murrine gtk2-engines-pixbuf \
                        libcanberra-gtk-module libcanberra-gtk3-module 

# CUDA 12.9
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda-repo-ubuntu2404-12-9-local_12.9.1-575.57.08-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-9-local_12.9.1-575.57.08-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9
sudo apt-get -y install cuda-drivers

# CUDNN 9.10.2
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cudnn9-cuda-12

# Install OpenCV dependencies
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
  -DCUDNN_INCLUDE_DIR=/usr/include/cuda/ \
  -DCUDNN_LIBRARY=/usr/lib/cuda/lib64/libcudnn.so \
  -DCUDA_ARCH_BIN="8.7" \
  -DCUDA_ARCH_PTX="" \
  -DOPENCV_DNN_CUDA=ON \
  -DOPENCV_ENABLE_NONFREE=ON \
  -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
  -DBUILD_opencv_python3=OFF \
  -DOPENCV_GENERATE_PKGCONFIG=ON
make -j$(nproc)
make install


# Install GoogleTest dependencies
cd ../../..
mkdir -p lib/gtest/build && cd lib/gtest/build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=../install 

make -j$(nproc)
make install