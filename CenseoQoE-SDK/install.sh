#!/bin/bash

CENSEOQOE_SDK_PATH=$PWD
INSTALL_PACKAGES_PATH="$CENSEOQOE_SDK_PATH/install_packages/$1"

mkdir -p ${CENSEOQOE_SDK_PATH}/third_party/ffmpeg_lib/lib
mkdir -p ${CENSEOQOE_SDK_PATH}/third_party/ffmpeg_lib/bin

cd third_party
## 安装libtorch，只需要解压就好
unzip -o ${INSTALL_PACKAGES_PATH}/libtorch-1.7.1-cpu.zip -d ./

## 安装libzip
tar -zxvf ${INSTALL_PACKAGES_PATH}/libzip-1.7.3.tar.gz
cd libzip-1.7.3
if [[ -d build ]]; then
  rm -rf build
fi
mkdir build && cd build
cmake -D CMAKE_INSTALL_PREFIX="${CENSEOQOE_SDK_PATH}/third_party/libzip" ..
make clean && make -j$(nproc)
make install
cd ../..
rm -rf libzip-1.7.3

## 安装 openssl
tar -zxvf ${INSTALL_PACKAGES_PATH}/openssl-3.0.0-beta1.tar.gz
cd openssl-3.0.0-beta1
if [[ -d build ]]; then
  rm -rf build
fi
./config shared --prefix="${CENSEOQOE_SDK_PATH}/third_party/openssl" --openssldir="${CENSEOQOE_SDK_PATH}/third_party/openssl/ssl"
make clean && make -j$(nproc)
make install
cd ..
rm -rf openssl-3.0.0-beta1

### 安装libcurl
tar -zxvf ${INSTALL_PACKAGES_PATH}/curl-7.77.0.tar.gz
cd curl-7.77.0
if [[ -d build ]]; then
  rm -rf build
fi
./configure --prefix="${CENSEOQOE_SDK_PATH}/third_party/libcurl" \
  --with-ssl="${CENSEOQOE_SDK_PATH}/third_party/openssl"
make clean && make -j$(nproc)
make install
cd ..
rm -rf curl-7.77.0

## 安装x264
unzip -o ${INSTALL_PACKAGES_PATH}/x264-master.zip -d ./
cd x264-master
if [[ -d build ]]; then
  rm -rf build
fi
./configure --prefix="${CENSEOQOE_SDK_PATH}/third_party/ffmpeg_lib" \
  --bindir="${CENSEOQOE_SDK_PATH}/third_party/ffmpeg_lib/bin" \
  --enable-static --enable-shared --enable-pic
make clean && make -j$(nproc)
make install
cd ..
rm -rf x264

## 安装ffmpeg
unzip -o ${INSTALL_PACKAGES_PATH}/FFmpeg-n4.0.4.zip -d ./
cd FFmpeg-n4.0.4
export LD_LIBRARY_PATH=${CENSEOQOE_SDK_PATH}/third_party/ffmpeg_lib/lib/:$LD_LIBRARY_PATH
export PATH=${CENSEOQOE_SDK_PATH}/third_party/ffmpeg_lib/bin:$PATH
export PKG_CONFIG_PATH=${CENSEOQOE_SDK_PATH}/third_party/ffmpeg_lib/lib/pkgconfig:$PKG_CONFIG_PATH
if [[ -d build ]]; then
  rm -rf build
fi
./configure \
  --prefix=${CENSEOQOE_SDK_PATH}/third_party/ffmpeg_lib \
  --extra-cflags=-I${CENSEOQOE_SDK_PATH}/third_party/ffmpeg_lib/include \
  --extra-ldflags=-L${CENSEOQOE_SDK_PATH}/third_party/ffmpeg_lib/lib \
  --extra-libs=-lpthread \
  --extra-libs=-lm \
  --bindir=${CENSEOQOE_SDK_PATH}/third_party/ffmpeg_lib/bin \
  --enable-gpl --enable-zlib --enable-libx264 \
  --enable-shared --enable-pic
make clean && make -j$(nproc)
make install
cd ..
rm -rf FFmpeg-n4.0.4

## 安装opencv
tar -zxvf ${INSTALL_PACKAGES_PATH}/opencv-3.4.15.tar.gz
cd opencv-3.4.15
if [[ -d build ]]; then
  rm -rf build
fi
mkdir build && cd build
cmake \
  -D CMAKE_INSTALL_PREFIX=${CENSEOQOE_SDK_PATH}/third_party/opencv_lib \
  -D BUILD_EXAMPLES=OFF \
  -D BUILD_TESTS=OFF \
  -D WITH_CUDA=OFF \
  -D BUILD_DOCS=OFF \
  -D BUILD_PNG=ON \
  -D BUILD_JPEG=ON \
  -D BUILD_TIFF=ON \
  -D OPENCV_EXTRA_EXE_LINKER_FLAGS="-Wl,-rpath,${CENSEOQOE_SDK_PATH}/third_party/ffmpeg_lib/lib" \
  ..
make clean && make -j$(nproc)
make install
cd ../..
rm -rf opencv-3.4.15
