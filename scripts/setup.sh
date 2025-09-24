#!/bin/bash

# chmod u+x setup.sh

set -e

###=== Constants ===###

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_CORES=$(nproc)
LOG_FILE="$SCRIPT_DIR/setup_$(date +'%Y-%m-%d_%H-%M-%S').log"

SUPPORTED_TARGETS=("x86_64-wsl2")
TARGET=""
UBUNTU_VERSION="ubuntu2404"
CUDA_VERSION="12.6.2"
CUDNN_VERSION="9.3.0"
TENSORRT_VERSION="10.3.0"
CUDA_ARCH_BIN="8.6"

###=== Colors ===###

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' 

###=== Miscellaneous ===###

help() 
{
  echo -e "${YELLOW}Usage: $0 --target <target> [options]"
  echo -e "Options:"
  echo -e "  --target <target>      Set the hardware target (${SUPPORTED_TARGETS[*]})"
  echo -e "  --all                  Install dependencies and build everything"
  echo -e "  --deps                 Install system dependencies"
  echo -e "  --cuda_cudnn           Install CUDA and CUDNN"
  echo -e "  --opencv               Build OpenCV"
  echo -e "  --gtest                Build GoogleTest"
  echo -e "  --help                 Show this help message${NC}"
  exit 1
}

log_and_run() 
{
  "$@" >>"$LOG_FILE" 2>&1
}

check_target_validity() 
{
  for t in "${SUPPORTED_TARGETS[@]}"; do
    [[ "$t" == "$TARGET" ]] && return
  done
  echo -e "${RED}Invalid or missing target: '$TARGET'. Supported targets are: ${SUPPORTED_TARGETS[*]}${NC}"
  exit 1
}

parse_args() 
{
  if [[ $# -eq 0 ]]; then
    help
  fi

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --target)
        TARGET="$2"
        shift 2
        ;;
      --all)
        ACTIONS+=("install_deps" "install_cuda_cudnn" "build_opencv" "build_gtest")
        shift
        ;;
      --deps)
        ACTIONS+=("install_deps")
        shift
        ;;
      --cuda_cudnn)
        ACTIONS+=("install_cuda_cudnn")
        shift
        ;;
      --opencv)
        ACTIONS+=("build_opencv")
        shift
        ;;
      --gtest)
        ACTIONS+=("build_gtest")
        shift
        ;;
      --help)
        help
        ;;
      *)
        echo -e "${RED}Unknown argument: $1${NC}"
        help
        ;;
    esac
  done

  if [[ -z "$TARGET" ]]; then
    echo -e "${RED}Error: --target must be specified${NC}"
    help
  fi

  check_target_validity
}

###=== Dependency Installation ===###

install_deps() 
{
  echo -e "${GREEN}Installing system packages...${NC}"
  log_and_run sudo apt-get update
  log_and_run sudo apt-get upgrade -y
  log_and_run sudo apt-get install -y cmake g++ build-essential clang-tidy gcc-12 g++-12 \
    libgtk2.0-dev pkg-config libgtk-3-0 libgail-common libatk-adaptor libgtk-3-common \
    gtk2-engines-murrine gtk2-engines-pixbuf libcanberra-gtk-module \
    libcanberra-gtk3-module ffmpeg libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
}

install_cuda()
{
  echo -e "${GREEN}Installing CUDA ${CUDA_VERSION}...${NC}"
  log_and_run wget -q https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_VERSION}/x86_64/cuda-ubuntu2404.pin
  log_and_run sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
  log_and_run wget -q https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/cuda-repo-${UBUNTU_VERSION}-${CUDA_VERSION//./-}-local_${CUDA_VERSION//./.}-575.57.08-1_amd64.deb
  log_and_run sudo dpkg -i cuda-repo-${UBUNTU_VERSION}-${CUDA_VERSION//./-}-local_${CUDA_VERSION//./.}-575.57.08-1_amd64.deb
  log_and_run sudo cp /var/cuda-repo-${UBUNTU_VERSION}-${CUDA_VERSION//./-}-local/cuda-*-keyring.gpg /usr/share/keyrings/
  log_and_run sudo apt-get update
  log_and_run sudo apt-get -y install cuda-toolkit-${CUDA_VERSION//./-}
}

install_cudnn()
{
  echo -e "${GREEN}Installing cuDNN ${CUDNN_VERSION}...${NC}"
  log_and_run wget -q https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_VERSION}/x86_64/cuda-keyring_1.1-1_all.deb
  log_and_run sudo dpkg -i cuda-keyring_1.1-1_all.deb
  log_and_run sudo apt-get update
  log_and_run sudo apt-get -y install cudnn9-cuda-12
}

install_tensorrt()
{
wget https://developer.download.nvidia.com/compute/machine-learning/tensorrt/secure/10.3.0/local_repos/nv-tensorrt-local-repo-ubuntu2404-10.3.0-cuda-12.5_1.0-1_amd64.deb
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2404-10.3.0-cuda-12.5_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-10.3.0-cuda-12.5/nv-tensorrt-local-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install tensorrt=10.3.0.1-1+cuda12.5 -y
sudo apt-get install libnvinfer10=10.3.0.1-1+cuda12.5 libnvinfer-dev=10.3.0.1-1+cuda12.2 libnvinfer-plugin10=10.3.0.1-1+cuda12.2 libnvparsers10=10.3.0.1-1+cuda12.2 libnvonnxparsers10=10.3.0.1-1+cuda12.2 -y
}

###=== OpenCV Build ===###

build_opencv() 
{
  echo -e "${GREEN}Building OpenCV...${NC}"
  log_and_run mkdir -p lib/opencv/build
  cd lib/opencv/build

  log_and_run cmake ../../opencv \
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
    -DCUDA_ARCH_BIN=${CUDA_ARCH_BIN} \
    -DCUDA_ARCH_PTX="" \
    -DOPENCV_DNN_CUDA=ON \
    -DOPENCV_ENABLE_NONFREE=ON \
    -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -DBUILD_opencv_python3=OFF \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-12

  log_and_run make -j${NUM_CORES}
  log_and_run make install
  cd "$SCRIPT_DIR"
}

###=== GTest Build ===###

build_gtest() 
{
  echo -e "${GREEN}Building Google Test...${NC}"
  log_and_run mkdir -p lib/gtest/build
  cd lib/gtest/build

  log_and_run cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install
  log_and_run make -j${NUM_CORES}
  log_and_run make install
  cd "$SCRIPT_DIR"
}

###=== Main ===###

ACTIONS=()
parse_args "$@"

echo -e "${GREEN}Setup started for target: $TARGET${NC}"

for action in "${ACTIONS[@]}"; do
  $action
done

echo -e "${GREEN}Setup complete ${NC}"
echo -e "${NC}Log saved to: $LOG_FILE${NC}"