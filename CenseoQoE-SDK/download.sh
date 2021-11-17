#!/bin/bash
CENSEOQOE_SDK_PATH=$PWD
INSTALL_PACKAGES_PATH="$CENSEOQOE_SDK_PATH/install_packages/$1"
mkdir -p $INSTALL_PACKAGES_PATH
echo $INSTALL_PACKAGES_PATH

if [ $1 == "Linux" ]; then
  echo "<<<Download Linux ThirdParty Packages>>>"
  wget https://curl.se/download/curl-7.77.0.tar.gz -O $INSTALL_PACKAGES_PATH/curl-7.77.0.tar.gz
  wget https://libzip.org/download/libzip-1.7.3.tar.gz -O $INSTALL_PACKAGES_PATH/libzip-1.7.3.tar.gz
  wget https://www.openssl.org/source/openssl-3.0.0-beta1.tar.gz -O $INSTALL_PACKAGES_PATH/openssl-3.0.0-beta1.tar.gz
  wget https://code.videolan.org/videolan/x264/-/archive/master/x264-master.zip -O $INSTALL_PACKAGES_PATH/x264-master.zip
  wget https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n4.0.4.zip -O $INSTALL_PACKAGES_PATH/FFmpeg-n4.0.4.zip
  wget https://github.com/opencv/opencv/archive/refs/tags/3.4.15.tar.gz -O $INSTALL_PACKAGES_PATH/opencv-3.4.15.tar.gz
  wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.7.1%2Bcpu.zip -O $INSTALL_PACKAGES_PATH/libtorch-1.7.1-cpu.zip
else
  echo "<<<Download MacOS ThirdParty Packages>>>"
  wget https://curl.se/download/curl-7.77.0.tar.gz -O $INSTALL_PACKAGES_PATH/curl-7.77.0.tar.gz
  wget https://libzip.org/download/libzip-1.7.3.tar.gz -O $INSTALL_PACKAGES_PATH/libzip-1.7.3.tar.gz
  wget https://www.openssl.org/source/openssl-3.0.0-beta1.tar.gz -O $INSTALL_PACKAGES_PATH/openssl-3.0.0-beta1.tar.gz
  wget https://code.videolan.org/videolan/x264/-/archive/master/x264-master.zip -O $INSTALL_PACKAGES_PATH/x264-master.zip
  wget https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n4.0.4.zip -O $INSTALL_PACKAGES_PATH/FFmpeg-n4.0.4.zip
  wget https://github.com/opencv/opencv/archive/refs/tags/3.4.15.tar.gz -O $INSTALL_PACKAGES_PATH/opencv-3.4.15.tar.gz
  wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.7.1.zip -O $INSTALL_PACKAGES_PATH/libtorch-1.7.1-cpu.zip
fi
