##  CenseoQoE-Algorithm: 图像/视频画质评价算法模型

CenseoQoE-Algorithm 提供无参考和有参考的图像/视频画质评价算法模型训练Codebase，基于这个Codebase，不需要复杂的模型或tricks，
就可以在多个质量评价数据集上获得 **Strong Baseline** 。CenseoQoE-Algorithm 包含我们自研的改进、优化之后的模型，后续也会不断地集成业界开源的优秀的算法模型。
除此之外，CenseoQoE-Algorithm 开源多个在不同业务场景下训练得到的模型，用户可以直接应用这些预训练模型做画质评价或是进行微调、优化等。

### Model Zoo
Model Zoo是我们用CenseoQoE针对不同的业务场景用不同的数据集训练出来的模型，支持不同格式的模型，包括pytorch、onnx、tensorflow、torchscript等，
Config文件是模型输入、推理、输出的参数设置。pytorch可作为预训练模型进行恢复训练，具体见[train_code](./train_code) 。其他格式需要搭配Config文件
进行使用，具体见 [test_code](./test_code)。

|业务场景|说明|是否有参考|数据集|PLCC|SROCC|Pytorch|Onnx|Tensorflow|TorchScript|Config文件|
|----|----|----|----|----|----|----|----|----|----|----|
|UGC视频|通用|否|自研数据集|0.902|0.880|[模型下载](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/UGC/nr/general/model.pth) / [config路径](./train_code/configs/baseline/OWN_UGC/ugc_video_general_vqa_res18_basemodel_nr.py) |[下载](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/UGC/nr/general/model.onnx)|[下载](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/UGC/nr/general/saved_model.pb)|[下载](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/UGC/nr/general/model.pt)|[下载](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/UGC/nr/general/config.json)|
|PGC视频|针对编解码压缩|否|自研数据集|0.961|0.959|[模型下载](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/PGC/nr/compressed/model.pth) / [config路径](./train_code/configs/baseline/OWN_PGC/pgc_video_compressed_vqa_res18_basemodel_nr.py) |[下载](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/PGC/nr/compressed/model.onnx)|[下载](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/PGC/nr/compressed/saved_model.pb)|[下载](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/PGC/nr/compressed/model.pt)|[下载](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/PGC/nr/compressed/config.json)|
|云游戏|针对编解码压缩|否|自研数据集|0.971|0.968|[模型下载](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/GAMES/nr/compressed/model.pth) / [config路径](./train_code/configs/baseline/OWN_CLOUD_GAME_VIDEO/cloud_game_video_compressed_vqa_res18_basemodel_nr.py) |[下载](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/GAMES/nr/compressed/model.onnx)|[下载](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/GAMES/nr/compressed/saved_model.pb)|[下载](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/GAMES/nr/compressed/model.pt)|[下载](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/GAMES/nr/compressed/config.json)|
|直播视频|coming soon~|-|-|-|-|-|-|-|-|-|
注：
* PLCC和SROCC的结果在自研数据集的测试集（随机20%）计算得到。
* 后续会公布更多针对不同场景的模型。
* 当前公布出来的是baseline的模型，后续会开源性能更优的模型。

### 环境安装
#### 普通安装
* python环境: python3。建议使用anaconda创建一个新的python=3.6的环境，避免影响到本地其他python环境。
* 安装python包：`pip install -r requirement.txt`。
* 其他（可选）
    * 如果你需要将pytorch模型转为tensorflow 模型，请根据[onnx-tensorflow](https://github.com/onnx/onnx-tensorflow) 安装onnx-tensorflow。
即：
        ``` 
        git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow
        git checkout tf-1.x
        pip install -e .
        ```
    * 如果你选择使用 tensorflow的模型进行推理（[test_code](./test_code))，需要安装tensorflow。GPU版本：`pip install tensorflow-gpu==1.15`, CPU版本：`pip install tensorflow==1.15.0`


#### Docker安装
* 构建镜像，运行`docker build -t censeo-algo .`
* 运行进入容器示例:
    ```
    docker run -it --gpus=all \
    --name dev-censeoqoe-algo \
    censeoqoe-algo \
    /bin/bash
    ```
* 更多docker用法请自行google

### 训练
详见 [train_code](./train_code) 里的介绍。
    
### 测试
详见 [test_code](./test_code) 里的介绍。
