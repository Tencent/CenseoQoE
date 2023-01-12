## CenseoQoE-SDK: Image/Video Quality Assessment SDK
### Project Description
As shown in [Model Zoo](../CenseoQoE-Algorithm/README.md), we released image quality assessment models covering cloud games, UGC video, PGC video and other different business scenarios, and CenseoQoE-SDK integrates these models, which does not require users to manually download the model and write inference code. After compiling, you can choose the corresponding model to evaluate the picture/video according to your own needs. It can be used as a picture quality evaluation tool, or as an SDK embedded in your own project to achieve out-of-the-box use.

For cross-platform support, the CenseoQoE-SDK is implemented in C++ and the model inference part is implemented in libtorch (In this section, you can change the inference framework to suit your needs).

### Instructions
#### 1. Installation dependencies
* System: The tests could be compiled and run on both Linux and MacOS.
* cmake>=3.10
* gcc>=5+ (c++14)
* Additional installation packages (openssl, libcurl, libzip, x264, ffmpeg, opencv, libtorch) are needed for model downloading, decompressing, input preprocessing, inference, etc. 
  Please run `bash download.sh MacOS` for MacOS and `bash download.sh Linux` for Linux. The following table is the official download address of the third-party library:

    |第三方库|Linux|MacOS|
    |----|----|----|
    |libcurl|[curl-7.77.0.tar.gz](https://curl.se/download/curl-7.77.0.tar.gz)|[curl-7.77.0.tar.gz](https://curl.se/download/curl-7.77.0.tar.gz)|
    |libzip|[libzip-1.7.3.tar.gz](https://libzip.org/download/libzip-1.7.3.tar.gz)|[libzip-1.7.3.tar.gz](https://libzip.org/download/libzip-1.7.3.tar.gz)|
    |openssl|[openssl-3.0.0-beta1.tar.gz](https://www.openssl.org/source/openssl-3.0.0-beta1.tar.gz)|[openssl-3.0.0-beta1.tar.gz](https://www.openssl.org/source/openssl-3.0.0-beta1.tar.gz)|
    |x264|[x264-master.zip](https://code.videolan.org/videolan/x264/-/archive/master/x264-master.zip)|[x264-master.zip](https://code.videolan.org/videolan/x264/-/archive/master/x264-master.zip)|
    |ffmpeg|[FFmpeg-n4.0.4.zip](https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n4.0.4.zip)|[FFmpeg-n4.0.4.zip](https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n4.0.4.zip)|
    |opencv|[opencv-3.4.15.tar.gz](https://github.com/opencv/opencv/archive/refs/tags/3.4.15.tar.gz)|[opencv-3.4.15.tar.gz](https://github.com/opencv/opencv/archive/refs/tags/3.4.15.tar.gz)|
    |libtorch|[libtorch-1.7.1-cpu.zip](https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.7.1%2Bcpu.zip)|[libtorch-1.7.1-cpu.zip](https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.7.1.zip)|
    
  Downloaded files will be placed in the same folder:`install_packages/Linux` or `install_packages/MacOS`.

* To install dependency packages, you can run`bash install.sh MacOS` or `bash install.sh Linux` depending on your system.

**Note that if these third-party libraries have already been installed, you can ignore the above steps. Please make sure you can find these third-party libraries when compiling (CMakeLists.txt)**

#### 2. Compiling & Running
* Run `bash build.sh` to compile the project.
* Run: 
    ```
    ./build/CenseoQoE \
    --fr {If it is NR, remove this line} \
    --dis_file {Distorted picture/video paths} \
    --ref_file {Reference picture/video path (no need to write in NR mode)} \
    --subsample {Only for video，1 means every frame is predicted} \
    --model_scene {Model scenario, selected from [UGC, PGC, GAMES]} \
    --model_name {Model name, selected from [general, compressed]} \
    --result_dir {Result save path} 
    ```
For example: `./build/CenseoQoE --dis_file ./assets/pgc-high.mp4 --model_scene PGC --model_name compressed --subsample 1 --result_dir ./results`

**For more details about the support for model_name and model_scene, please see './assets/model_urls.json '. If model_url is available, you can choose it.**

### TODO
- [ ] Add GPU running for libtorch.
- [ ] Add other inference frameworks such as tensorflow, etc.
 
### More Problems
#### 1. The gcc/g++ version is too early.
If your version of gcc is less than 4.8, you need to upgrade it. The centos systems can be managed using scl, ubuntu can be managed using update-alternatives, which does not affect previous versions of gcc.
For centos, the reference command is as follows：
```
sudo yum install centos-release-scl
sudo yum install devtoolset-7-gcc*
scl enable devtoolset-7 bash
which gcc
gcc --version
```
For other systems, you should do the same or upgrade gcc/g++.

#### 2. The cmake version is too low.
For centos, cmake installation can refer to：
```
wget https://github.com/Kitware/CMake/releases/download/v3.11.0/cmake-3.11.0.tar.gz
tar -zxf cmake-3.11.0.tar.gz
cd cmake-3.11.0
./bootstrap --prefix=/usr --datadir=share/cmake --docdir=doc/cmake && make -j8
sudo make install
```
Please search in Google for other systems.
    
#### 3. Failed to install x264.
The possible reason is that nasm is not installed in the system. To solve the problem, you need to download nasm, decompress it, and then install it.
```
tar -zxvf nasm-2.14.02.tar.gz
cd nasm-2.14.02
./configure make && make install
cd ..
```

#### 4. Can't find x264.so, libssl.so or other .so file when running after a successful compilation.
Can't find `x264.so`：`export LD_LIBRARY_PATH=$PWD/third_party/ffmpeg_lib/lib/:$LD_LIBRARY_PATH`
Can't find `libssl.so`：`export LD_LIBRARY_PATH=$PWD/third_party/openssl/lib/:$LD_LIBRARY_PATH`

Other .so documents can be resolved by referring to the above.
