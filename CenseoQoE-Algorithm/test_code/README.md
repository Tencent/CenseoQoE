##  CenseoQoE-Algorithm: 模型推理
这里的推理代码统一使用tensorflow的saved model的形式(需要先安装tensorflow==1.15)，可以在本地快速地调用调试，也可以方便地部署工业常用的
tf serving方式。但我们不仅提供tensorflow的模型， 还有对应其他格式的模型，比如`onnx`和`torchscript`等, 具体见[Model Zoo](../README.md)，
用户可以根据自己的情况选择合适的模型并部署。
具体的使用为：
* 在[Model Zoo](../README.md)下载想要运行的tensorflow模型，保存到某一个文件夹下`$SAVED_MODEL_PATH`。
**注意要把对应的Config文件一块下载，并放到同一个文件夹下**
    ```
    python run.py \
    --saved_model $SAVED_MODEL_PATH \
    --dst_file_path $DST_FILE_PATH \
    --ref_file_path $REF_FILE_PATH \
    --save_name $SAVE_NAME
    
    其中 $SAVED_MODEL_PATH是包含saved_model文件夹路径;
    $DST_FILE_PATH是需要预测的文件路径，可以是视频或是图片，也可以是文件夹，如果是文件夹，默认将文件夹下的所有符合要求的文件都预测;
    $REF_FILE_PATH是对应的参考文件路径，无参模型不用提供;
    $SAVE_NAME 是结果保存的json文件名，非必须，如果不指定将生成带时间戳的json文件，并在最后将文件名打印出来。
    ```
* 结果保存在`results`文件夹下，名字为`$SAVE_NAME.json`
