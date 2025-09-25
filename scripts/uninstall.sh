#!/bin/bash

# chmod u+x uninstall.sh

set -e

### Uninstall CUDA 

echo "Uninstalling CUDA..." | tee -a uninstall_log.txt
sudo dpkg --remove --force-all cuda-repo-ubuntu2404-12-6-local
sudo apt purge --autoremove cuda* nvidia* -y | tee -a uninstall_log.txt
sudo apt autoremove -y | tee -a uninstall_log.txt
sudo rm -rf /usr/local/cuda* | tee -a uninstall_log.txt
sudo rm -rf /etc/apt/sources.list.d/cuda* | tee -a uninstall_log.txt
sudo rm -rf /var/cuda-repo* | tee -a uninstall_log.txt
sudo rm -f /etc/apt/preferences.d/cuda-repository-pin* | tee -a uninstall_log.txt
sudo apt-get update | tee -a uninstall_log.txt

### Uninstall cuDNN

echo "Uninstalling cuDNN..." | tee -a uninstall_log.txt
sudo apt purge --autoremove cudnn* libcudnn* -y | tee -a uninstall_log.txt
sudo apt autoremove -y | tee -a uninstall_log.txt
sudo rm -rf /usr/lib/x86_64-linux-gnu/libcudnn* | tee -a uninstall_log.txt
sudo rm -rf /usr/include/x86_64-linux-gnu/cudnn* | tee -a uninstall_log.txt
sudo rm -rf /var/cudnn-local-repo* | tee -a uninstall_log.txt
sudo apt-get update | tee -a uninstall_log.txt

### Uninstall TensorRT

echo "Uninstalling TensorRT..." | tee -a uninstall_log.txt
sudo dpkg --remove --force-all nv-tensorrt-local-repo-ubuntu2204-10.3.0-cuda-12.5
sudo apt purge --autoremove tensorrt libnvinfer* libnvparsers* libnvonnx* -y | tee -a uninstall_log.txt
sudo apt autoremove -y | tee -a uninstall_log.txt
sudo rm -rf /usr/src/tensorrt | tee -a uninstall_log.txt
sudo rm -rf /usr/lib/x86_64-linux-gnu/libnvinfer* | tee -a uninstall_log.txt
sudo rm -rf /usr/include/x86_64-linux-gnu/Nv* | tee -a uninstall_log.txt
sudo rm -rf /var/nv-tensorrt-local-repo* | tee -a uninstall_log.txt
sudo rm -f /etc/apt/preferences.d/tensorrt-pin | tee -a uninstall_log.txt
sudo apt-get update | tee -a uninstall_log.txt

sed -i '/cuda/d' ~/.bashrc
sed -i '/tensorrt/d' ~/.bashrc
sed -i '/LD_LIBRARY_PATH/d' ~/.bashrc
source ~/.bashrc

### Verify
# dpkg -l | grep -E 'cuda|cudnn|tensorrt|nvinfer'
# ls /usr/local/cuda* /usr/src/tensorrt /usr/lib/aarch64-linux-gnu/libnvinfer* /usr/lib/aarch64-linux-gnu/libcudnn*
# which nvcc trtexec