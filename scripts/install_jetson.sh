#!/bin/bash

# chmod u+x install_jetson.sh

set -e

### Dependencies

sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y cmake g++ build-essential clang-tidy gcc-12 g++-12 \
    libgtk2.0-dev pkg-config libgtk-3-0 libgail-common libatk-adaptor libgtk-3-common \
    gtk2-engines-murrine gtk2-engines-pixbuf libcanberra-gtk-module \
    libcanberra-gtk3-module ffmpeg libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

### CUDA 

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda-tegra-repo-ubuntu2204-12-6-local_12.6.0-1_arm64.deb
sudo dpkg -i cuda-tegra-repo-ubuntu2204-12-6-local_12.6.0-1_arm64.deb
sudo cp /var/cuda-tegra-repo-ubuntu2204-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6 cuda-compat-12-6

### cuDNN

wget https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-tegra-repo-ubuntu2204-9.3.0_1.0-1_arm64.deb
sudo dpkg -i cudnn-local-tegra-repo-ubuntu2204-9.3.0_1.0-1_arm64.deb
sudo cp /var/cudnn-local-tegra-repo-ubuntu2204-9.3.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn cudnn-cuda-12

### TensorRT



echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc


### OpenCV

mkdir -p lib/opencv/build
cd lib/opencv/build

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
    -DCUDNN_INCLUDE_DIR=/usr/include/aarch64-linux-gnu \
    -DCUDNN_LIBRARY=/usr/lib/aarch64-linux-gnu/libcudnn.so \
    -DCUDA_ARCH_BIN=8.7 \
    -DCUDA_ARCH_PTX="" \
    -DOPENCV_DNN_CUDA=ON \
    -DOPENCV_ENABLE_NONFREE=ON \
    -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -DBUILD_opencv_python3=OFF \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DWITH_V4L=ON \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-12
        
make -j${NUM_CORES}
make install    

### Verify
# nvcc --version
# which nvcc trtexec
# dpkg -l | grep -E 'cuda|cudnn|tensorrt|nvinfer'