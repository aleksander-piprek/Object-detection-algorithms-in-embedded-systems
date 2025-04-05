#!/bin/bash

sudo apt-get update
sudo apt-get upgrade -y

sudo apt-get install -y cmake g++ build-essential 
sudo apt-get install -y libgtk2.0-dev pkg-config libgtk-3-0 libgail-common libatk-adaptor libgtk-3-common gtk2-engines-murrine gtk2-engines-pixbuf

mkdir -p lib/opencv/build && cd lib/opencv/build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=../install \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DWITH_QT=OFF \
  -DWITH_GTK=ON \
  -DWITH_OPENGL=OFF \
  -DWITH_VTK=OFF

make -j$(nproc)
make install