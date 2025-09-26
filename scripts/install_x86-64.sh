#!/bin/bash

# chmod u+x install_x86-64.sh

set -e

### Dependencies

sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y cmake g++ build-essential clang-tidy gcc-12 g++-12 \
    libgtk2.0-dev pkg-config libgtk-3-0 libgail-common libatk-adaptor libgtk-3-common \
    gtk2-engines-murrine gtk2-engines-pixbuf libcanberra-gtk-module \
    libcanberra-gtk3-module ffmpeg libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgtk-3-dev libopenjp2-7-dev \
    zlib1g-dev git ninja-build libprotobuf-dev protobuf-compiler libeigen3-dev python3.10-venv

### CUDA 12.6

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.0-560.28.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-6-local_12.6.0-560.28.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

### cuDNN 9.3

wget https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-repo-ubuntu2204-9.3.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.3.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.3.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn cudnn9-cuda-12 cudnn-cuda-12

### TensorRT 10.3

wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.3.0/local_repo/nv-tensorrt-local-repo-ubuntu2204-10.3.0-cuda-12.5_1.0-1_amd64.deb
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2204-10.3.0-cuda-12.5_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-10.3.0-cuda-12.5/nv-tensorrt-local-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install tensorrt

### ONNX Runtime 1.22.0

cd lib/onnxruntime
./build.sh --config Release \
    --use_cuda \
    --cuda_home /usr/local/cuda-12.6 \
    --cudnn_home /usr \
    --use_tensorrt \
    --tensorrt_home /usr \
    --build_shared_lib \
    --parallel 8 \
    --cmake_extra_defines CMAKE_CXX_COMPILER=/usr/bin/g++-12 \
                          CMAKE_C_COMPILER=/usr/bin/gcc-12 \
                          CUDA_VERSION=12.6 \
                          CUDNN_VERSION=9.3.0 \
                          onnxruntime_BUILD_UNIT_TESTS=OFF \
                          onnxruntime_USE_PREINSTALLED_EIGEN=ON 

cd build/Linux/Release
make install
sudo cp -r ./install/* /usr/local/
cd ~/Object-

echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
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
    -DCUDNN_INCLUDE_DIR=/usr/include/ \
    -DCUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so \
    -DCUDA_ARCH_BIN=8.6 \
    -DCUDA_ARCH_PTX="" \
    -DOPENCV_DNN_CUDA=ON \
    -DOPENCV_ENABLE_NONFREE=ON \
    -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -DBUILD_opencv_python3=OFF \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-12

make -j16
make install

### Verify
# nvcc --version
# which nvcc trtexec
# dpkg -l | grep -E 'cuda|cudnn|tensorrt|nvinfer'