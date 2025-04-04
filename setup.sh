#!/bin/bash

sudo apt-get update
sudo apt-get upgrade -y

sudo apt-get install -y cmake g++ build-essential libgtk2.0-dev pkg-config 

cd lib/opencv
mkdir build
cd build
cmake ..
make -j$(nproc)
sudo make install