##  CenseoQoE-Algorithm: Image/Video Quality Assessment Algorithm

CenseoQoE-Algorithm provides training codebase for unreferenced and referenced image/video image quality assessment algorithm. Based on this codebase, a **Strong Baseline** can be achieved on multiple quality evaluation datasets without any complex model or tricks.
CenseoQoE-Algorithm includes the improved and optimized model developed by us, and will continue to integrate the excellent open source algorithm model in the industry.

In addition, CenseoQoE-Algorithm releases several models trained in different business scenarios. Users can directly apply these pre-trained models for image quality assessment, fine tuning, optimization, etc.



### Model Zoo
Model Zoo is a model trained by CenseoQoE with different datasets for different business scenarios. It supports models in different formats, including pytorch, onnx, tensorflow, torchscript, etc. The config file is the parameter setting for model input, inference, and output. Meanwhile, pytorch can be used as a pre-training model for recovery training, see details in [train_code](./train_code). Other formats must be used with the config file. For details, see [test_code](./test_code)。

|Business Scenarios|Description|Any Reference|Dataset|PLCC|SROCC|Pytorch|Onnx|Tensorflow|TorchScript|Config File|
|----|----|----|----|----|----|----|----|----|----|----|
|UGC Video|General Purpose|No|Self-developed Dataset|0.902|0.880|[Model download](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/UGC/nr/general/model.pth) / [config path](./train_code/configs/baseline/OWN_UGC/ugc_video_general_vqa_res18_basemodel_nr.py) |[download](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/UGC/nr/general/model.onnx)|[download](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/UGC/nr/general/saved_model.pb)|[download](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/UGC/nr/general/model.pt)|[download](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/UGC/nr/general/config.json)|
|PGC Video|For Codec Compression|No|Self-developed Dataset|0.961|0.959|[Model download](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/PGC/nr/compressed/model.pth) / [config path](./train_code/configs/baseline/OWN_PGC/pgc_video_compressed_vqa_res18_basemodel_nr.py) |[download](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/PGC/nr/compressed/model.onnx)|[download](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/PGC/nr/compressed/saved_model.pb)|[download](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/PGC/nr/compressed/model.pt)|[download](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/PGC/nr/compressed/config.json)|
|Cloud Game|For Codec Compression|No|Self-developed Dataset|0.971|0.968|[Model download](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/GAMES/nr/compressed/model.pth) / [config path](./train_code/configs/baseline/OWN_CLOUD_GAME_VIDEO/cloud_game_video_compressed_vqa_res18_basemodel_nr.py) |[download](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/GAMES/nr/compressed/model.onnx)|[download](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/GAMES/nr/compressed/saved_model.pb)|[download](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/GAMES/nr/compressed/model.pt)|[download](https://censeoqoe-1258344700.cos.ap-guangzhou.myqcloud.com/CenseoQoE/checkpoints/GAMES/nr/compressed/config.json)|
|Live Video|Coming Soon~|-|-|-|-|-|-|-|-|-|
Note:
* The results of PLCC and SROCC were calculated in the test set (random 20%) of the self-developed dataset.
* More models for different scenarios will be released in the future.
* The baseline model is currently being released, and models with better performance will be released in the future.

### Installations
#### General Installation
* Python Environment Setting: python3。It is recommended to use anaconda to create a new python=3.6 environment. It can avoid affecting other local python environments.
* Python Package Installation：`pip install -r requirement.txt`。
* Others (optional)
    * If you need to convert the pytorch model to a tensorflow model, please refer to [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow) to install onnx-tensorflow。
For example：
        ``` 
        git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow
        git checkout tf-1.x
        pip install -e .
        ```
    * If you choose to use the tensorflow model for inference（[test_code](./test_code)), you need to install tensorflow. For GPU version：`pip install tensorflow-gpu==1.15`, for CPU version：`pip install tensorflow==1.15.0`


#### Docker Installation
* Build the image and run `docker build -t censeo-algo .`
* Run the container:
    ```
    docker run -it --gpus=all \
    --name dev-censeoqoe-algo \
    censeoqoe-algo \
    /bin/bash
    ```
*  Please search in Google for more docker usage.

### Trainning
See [train_code](./train_code) for more details.
    
### Testing
See [test_code](./test_code) for more details.
