# -*- coding: utf-8 -*-
# @Time    : 2021/5/13 3:58 下午
# @Author  : shaoguowen
# @Email   : shaoguowen@tencent.com
# @FileName: pytorch2onnx_jit_pb.py
# @Software: PyCharm

"""
将训练好的模型转为onnx、pb、tfserving使用的save_model、jit等格式，方便后续做模型部署，实现接口的统一
"""

import sys

sys.path.append(".")
import os
import argparse
import datetime
import shutil
import torch.onnx
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import torch
from mmcv import Config

import models

parser = argparse.ArgumentParser(description='model transform')
parser.add_argument('config', help='train config file path')
args = parser.parse_args()

config = Config.fromfile(args.config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = getattr(models, config.model.model_name)(config.model)

assert os.path.exists(config.trainer.resume_from)  # 因为是推理预测阶段，所以要保证resume_from的模型路径存在
print('==> Start Building model')
print('resume from:%s' % config.trainer.resume_from)
net_dict = model.state_dict()
pretrained_dict = torch.load(config.trainer.resume_from, map_location=device)
tmp1 = set(list(net_dict.keys()))
pretrained_dict_tmp = {k: v for k, v in pretrained_dict.items() if
                       k in net_dict and v.size() == net_dict[k].size()}
tmp3 = set(list(pretrained_dict_tmp.keys()))
print(tmp1 - tmp3)  # 为空才说明是完全匹配加载
net_dict.update(pretrained_dict_tmp)
model.load_state_dict(net_dict)
print('==> Successfully Building model')
model.forward = model.forward_onnx
model.to(device=device)
model.eval()


def torch_to_jit(torch_jit_path):
    """
    导出 torch jit, 用于c++ libtorch 推理
    :param torch_jit_path:
    :return:
    """
    input_c = 6 if config.model.fr else 3
    img_input = torch.randn(1, input_c, *config.data.sub_img_dim).to(device)
    dummy_inputs = (img_input,)
    traced_script_module = torch.jit.trace(model, dummy_inputs)
    traced_script_module.save(torch_jit_path)


def torch_to_onnx(onnx_save_path):
    """
    pytorch 模型导出为 onnx
    :param onnx_save_path:
    :return:
    """
    input_c = 6 if config.model.fr else 3
    img_input = torch.randn(1, input_c, *config.data.sub_img_dim).to(device)
    dummy_inputs = (img_input,)
    input_names = ['test_input']
    output_names = ['test_output']
    torch.onnx.export(model, args=dummy_inputs, input_names=input_names, output_names=output_names, f=onnx_save_path,
                      dynamic_axes={'test_input': [0]})


def onnx_to_pb(onnx_save_path, pb_save_path):
    """
    onnx 模型转为pb 格式
    :param onnx_save_path:
    :param pb_save_path:
    :return:
    """
    onnx_model = onnx.load(onnx_save_path)
    tf_rep = prepare(onnx_model, strict=False)
    tf_rep.export_graph(pb_save_path)


def create_graph(pb_file):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def pb_to_tfserving(pb_file, export_path, pb_io_name=None, input_node_name='input', output_node_name='output',
                    signature_name='serving_default'):
    """
    pb模型转为 tfserving 的save model格式
    :param pb_file:
    :param export_path:
    :param pb_io_name:
    :param input_node_name:
    :param output_node_name:
    :param signature_name:
    :return:
    """
    if pb_io_name is None:
        pb_io_name = []
    create_graph(pb_file)
    input_name = '%s:0' % pb_io_name[0]
    output_name = '%s:0' % pb_io_name[1]
    with tf.Session() as sess:
        in_tensor = sess.graph.get_tensor_by_name(input_name)
        out_tensor = sess.graph.get_tensor_by_name(output_name)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)  # export_path导出路径
        inputs = {input_node_name: tf.saved_model.utils.build_tensor_info(in_tensor)}
        outputs = {output_node_name: tf.saved_model.utils.build_tensor_info(out_tensor)}
        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs, outputs, method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        builder.add_meta_graph_and_variables(
            sess=sess, tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={signature_name: signature}, clear_devices=True)  # signature_name为签名，可自定义
        builder.save()


if __name__ == '__main__':
    date_str = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    model_save_dir = config.trainer.resume_from + "-" + date_str
    os.makedirs(model_save_dir, exist_ok=True)

    onnx_save_path = os.path.join(model_save_dir, "model.onnx")
    pb_save_path = os.path.join(model_save_dir, "model.pb")
    jit_save_path = os.path.join(model_save_dir, "model.pt")
    tfserving_path = os.path.join(model_save_dir, "saved_model")

    if os.path.exists(tfserving_path):
        shutil.rmtree(tfserving_path)
    os.makedirs(tfserving_path, exist_ok=True)
    torch_to_jit(jit_save_path)
    torch_to_onnx(onnx_save_path)
    onnx_to_pb(onnx_save_path, pb_save_path)
    pb_to_tfserving(pb_save_path, tfserving_path, pb_io_name=['test_input', 'test_output'])

    print(onnx_save_path)
    print(pb_save_path)
    print(jit_save_path)
    print(tfserving_path)
