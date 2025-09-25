#!/bin/bash

# chmod u+x uninstall.sh

set -e

### Uninstall CUDA 

echo "Uninstalling CUDA..." 
sudo dpkg --remove --force-all cuda-repo-ubuntu2404-12-6-local
sudo apt purge --autoremove cuda* nvidia* -y 
sudo apt autoremove -y 
sudo rm -rf /usr/local/cuda* 
sudo rm -rf /etc/apt/sources.list.d/cuda* 
sudo rm -rf /var/cuda-repo* 
sudo rm -f /etc/apt/preferences.d/cuda-repository-pin* 
sudo apt-get update 

### Uninstall cuDNN

echo "Uninstalling cuDNN..." 
sudo apt purge --autoremove cudnn* libcudnn* -y 
sudo apt autoremove -y 
sudo rm -rf /usr/lib/x86_64-linux-gnu/libcudnn* 
sudo rm -rf /usr/include/x86_64-linux-gnu/cudnn* 
sudo rm -rf /var/cudnn-local-repo* 
sudo apt-get update 

### Uninstall TensorRT

echo "Uninstalling TensorRT..." 
sudo dpkg --remove --force-all nv-tensorrt-local-repo-ubuntu2204-10.3.0-cuda-12.5_1.0-1
sudo apt purge --autoremove tensorrt libnvinfer* libnvparsers* libnvonnx* -y 
sudo apt autoremove -y 
sudo rm -rf /usr/src/tensorrt 
sudo rm -rf /usr/lib/x86_64-linux-gnu/libnvinfer* 
sudo rm -rf /usr/include/x86_64-linux-gnu/Nv* 
sudo rm -rf /var/nv-tensorrt-local-repo* 
sudo rm -f /etc/apt/preferences.d/tensorrt-pin 
sudo apt-get update 

sed -i '/cuda/d' ~/.bashrc
sed -i '/tensorrt/d' ~/.bashrc
sed -i '/LD_LIBRARY_PATH/d' ~/.bashrc
source ~/.bashrc

### Verify
# dpkg -l | grep -E 'cuda|cudnn|tensorrt|nvinfer'
# ls /usr/local/cuda* /usr/src/tensorrt /usr/lib/aarch64-linux-gnu/libnvinfer* /usr/lib/aarch64-linux-gnu/libcudnn*
# which nvcc trtexec