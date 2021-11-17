#!/bin/bash
export LD_LIBRARY_PATH=$PWD/third_party/ffmpeg_lib/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PWD/third_party/openssl/lib/:$LD_LIBRARY_PATH
CC=gcc
CXX=g++
CMAKE=cmake
rm -rf build
mkdir build
cd build
$CMAKE \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX \
  -DCMAKE_BUILD_TYPE=Release \
  ..

make clean && make -j$(nproc)
