## CenseoQoE-SDK: 图像/视频画质评价工具/SDK
### 项目说明
如[Model Zoo](../CenseoQoE-Algorithm/README.md) 所示，我们开源了涵盖云游戏、UGC视频和PGC视频等不同业务场景下的画质评价模型，
CenseoQoE-SDK则把这些模型集成起来，无需用户手动下载模型和写推理的代码，编译完之后可根据自己的需求选择相应的模型对图片/视频做画质评价，
既可以当作一个画质评价工具使用，也可以当成SDK嵌入到自己的项目中使用，实现开箱即用。
为了支持跨平台运行，CenseoQoE-SDK使用C++实现，模型推理部分使用libtorch实现（这部分可根据自己的需求更换推理框架）。

### 使用说明
#### 1. 安装依赖
* 系统: 目前测试在Linux和MacOS上都可以编译运行
* cmake>=3.10
* gcc>=5+ (c++14)
* 需要有下载其他的安装包(openssl、libcurl、libzip、x264、ffmpeg、opencv、libtorch）用于模型的下载、解压、输入预处理、推理等。如果是MacOS系统，
运行`bash download.sh MacOS`, 如果是Linux系统，运行`bash download.sh Linux`。下表为第三方库的官方下载地址：

    |第三方库|Linux|MacOS|
    |----|----|----|
    |libcurl|[curl-7.77.0.tar.gz](https://curl.se/download/curl-7.77.0.tar.gz)|[curl-7.77.0.tar.gz](https://curl.se/download/curl-7.77.0.tar.gz)|
    |libzip|[libzip-1.7.3.tar.gz](https://libzip.org/download/libzip-1.7.3.tar.gz)|[libzip-1.7.3.tar.gz](https://libzip.org/download/libzip-1.7.3.tar.gz)|
    |openssl|[openssl-3.0.0-beta1.tar.gz](https://www.openssl.org/source/openssl-3.0.0-beta1.tar.gz)|[openssl-3.0.0-beta1.tar.gz](https://www.openssl.org/source/openssl-3.0.0-beta1.tar.gz)|
    |x264|[x264-master.zip](https://code.videolan.org/videolan/x264/-/archive/master/x264-master.zip)|[x264-master.zip](https://code.videolan.org/videolan/x264/-/archive/master/x264-master.zip)|
    |ffmpeg|[FFmpeg-n4.0.4.zip](https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n4.0.4.zip)|[FFmpeg-n4.0.4.zip](https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n4.0.4.zip)|
    |opencv|[opencv-3.4.15.tar.gz](https://github.com/opencv/opencv/archive/refs/tags/3.4.15.tar.gz)|[opencv-3.4.15.tar.gz](https://github.com/opencv/opencv/archive/refs/tags/3.4.15.tar.gz)|
    |libtorch|[libtorch-1.7.1-cpu.zip](https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.7.1%2Bcpu.zip)|[libtorch-1.7.1-cpu.zip](https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.7.1.zip)|
    
    下载的文件会放在同一个文件夹下：`install_packages/Linux`或`install_packages/MacOS`。

* 安装依赖包，根据自己系统`bash install.sh MacOS`或`bash install.sh Linux`即可。

**注意，如果已经安装过这些第三方库，则忽略上面的步骤。但要确保在编译时（CMakeLists.txt）能找到这些第三方库**

#### 2. 编译运行
* 运行 `bash build.sh`编译项目。
* 运行: 
    ```
    ./build/CenseoQoE \
    --fr {如果是NR模式则去掉这一行} \
    --dis_file {失真图片/视频路径} \
    --ref_file {参考图片/视频路径（NR模式下不用写)} \
    --subsample {对视频有效，1表示每一帧都预测} \
    --model_scene {模型场景，从[UGC, PGC, GAMES]选择} \
    --model_name {模型名称, 从[general, compressed]选择} \
    --result_dir {结果保存路径} 
    ```
示例: `./build/CenseoQoE --dis_file ./assets/pgc-high.mp4 --model_scene PGC --model_name compressed --subsample 1 --result_dir ./results`

**model_name和model_scene的支持具体请看`./assets/model_urls.json`, 如果有model_url, 说明可选。**

### TODO
- [ ] 增加libtorch的GPU运行
- [ ] 增加其他推理框架的模型推理，比如tensorflow等
 
### 可能出现的问题
#### 1. gcc/g++版本过低
如果你的gcc版本小于4.8，需要升级版本。centos系统可以使用scl来管理，ubuntu可以使用update-alternatives管理，不会影响到之前的gcc版本。
centos 参考命令如下：
```
sudo yum install centos-release-scl
sudo yum install devtoolset-7-gcc*
scl enable devtoolset-7 bash
which gcc
gcc --version
```
其他系统应该也有类似的做法。或者直接升级gcc/g++。

#### 2. cmake 版本太低
centos 安装cmake可以参考：
```
wget https://github.com/Kitware/CMake/releases/download/v3.11.0/cmake-3.11.0.tar.gz
tar -zxf cmake-3.11.0.tar.gz
cd cmake-3.11.0
./bootstrap --prefix=/usr --datadir=share/cmake --docdir=doc/cmake && make -j8
sudo make install
```
其他系统请自行google。
    
#### 3. 安装x264不成功
可能原因是系统没有安装nasm，解决方法是安装nasm，下载nasm，解压后安装即可
```
tar -zxvf nasm-2.14.02.tar.gz
cd nasm-2.14.02
./configure make && make install
cd ..
```

#### 4. 编译成功后运行找不到 x264.so、libssl.so或其他的so文件
找不到`x264.so`：`export LD_LIBRARY_PATH=$PWD/third_party/ffmpeg_lib/lib/:$LD_LIBRARY_PATH`
找不到`libssl.so`：`export LD_LIBRARY_PATH=$PWD/third_party/openssl/lib/:$LD_LIBRARY_PATH`

其他so文件可参考以上解决。
