# -*- coding: utf-8 -*-
# @Time    : 2021/5/12 2:48 下午
# @Author  : shaoguowen
# @Email   : shaoguowen@tencent.com
# @FileName: inference.py
# @Software: PyCharm

import os
import argparse
import json
import datetime
import torch
from mmcv import Config
import numpy as np
import cv2
import torchvision.transforms as transforms

from utils import utils, logger
import models

parser = argparse.ArgumentParser(description='model inference test')
parser.add_argument('config', help='train config file path')
parser.add_argument('--dst_path', required=True, help='path of dis image/video')
parser.add_argument('--ref_path', required=False, help='path of ref image/video if in FR model')
parser.add_argument('--save_name', required=False, help='must be json file')
args = parser.parse_args()

config = Config.fromfile(args.config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = getattr(models, config.model.model_name)(config.model)

assert os.path.exists(config.trainer.resume_from)  # 因为是推理预测阶段，所以要保证resume_from的模型路径存在
logger_test = logger.get_logger(os.path.dirname(config.trainer.resume_from), "test.log")
logger_test.info('==> Start Building model')
logger_test.info('resume from:%s' % config.trainer.resume_from)
net_dict = model.state_dict()
pretrained_dict = torch.load(config.trainer.resume_from)
tmp1 = set(list(net_dict.keys()))
pretrained_dict_tmp = {k: v for k, v in pretrained_dict.items() if
                       k in net_dict and v.size() == net_dict[k].size()}
tmp3 = set(list(pretrained_dict_tmp.keys()))
logger_test.info(tmp1 - tmp3)  # 为空才说明是完全匹配加载
net_dict.update(pretrained_dict_tmp)
model.load_state_dict(net_dict)
logger_test.info('==> Successfully Building model')

model.to(device=device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])


def predict_fr(dst_path, ref_path=None):
    """
    predict_fr
    :param dst_path:
    :param ref_path:
    :return:
    """
    assert ref_path is not None
    if utils.is_img(dst_path):
        img_dst = cv2.imread(dst_path)
        img_dst = utils.transform_image(img_dst, transform, config.data)
        if img_dst is None:
            return -1
        img_dst = img_dst.unsqueeze(0)
        img_ref = cv2.imread(ref_path)
        img_ref = utils.transform_image(img_ref, transform, config.data)
        if img_ref is None:
            return -1
        img_ref = img_ref.unsqueeze(0)
        preds = model(torch.cat([img_dst, img_ref], 1).unsqueeze(0).float().to(device))
        pred = preds.mean().cpu().item()
        return pred
    elif utils.is_video(dst_path):
        vcap_dst = utils.read_video_gen(dst_path, 0, -1, 10, 4)
        vcap_ref = utils.read_video_gen(ref_path, 0, -1, 10, 4)
        mos_preds = []
        while True:
            imgs_dst = next(vcap_dst, None)
            imgs_ref = next(vcap_ref, None)
            if imgs_dst is None or len(imgs_dst) == 0 or imgs_ref is None or len(imgs_ref) == 0:
                break
            imgs_dst_ = []
            for img in imgs_dst:
                img_ = utils.transform_image(img, transform, config.data)
                if img_ is None:
                    return -1
                imgs_dst_.append(img_)
            imgs_dst = torch.stack(imgs_dst_)

            imgs_ref_ = []
            for img in imgs_ref:
                img_ = utils.transform_image(img, transform, config.data)
                if img_ is None:
                    return -1
                imgs_ref_.append(img_)
            imgs_ref = torch.stack(imgs_ref_)
            imgs_input = torch.cat([imgs_dst, imgs_ref], 1).unsqueeze(0).to(device)
            preds = model(imgs_input).cpu().data.numpy()
            mos_preds.append(preds.reshape(-1))
        return np.concatenate(mos_preds).mean().tolist()


def predict_nr(dst_path):
    """
    predict_nr
    :param dst_path:
    :return:
    """
    if utils.is_img(dst_path):
        img_dst = cv2.imread(dst_path)
        img_dst = utils.transform_image(img_dst, transform, config.data)
        if img_dst is None:
            return -1
        img_dst = img_dst.unsqueeze(0).unsqueeze(0).to(device)
        preds = model(img_dst)
        pred = preds.mean().cpu().item()
        return pred
    elif utils.is_video(dst_path):
        vcap_dst = utils.read_video_gen(dst_path, 0, -1, 10, 4)
        mos_preds = []
        while True:
            imgs_dst = next(vcap_dst, None)
            if imgs_dst is None or len(imgs_dst) == 0:
                break
            imgs_dst_ = []
            for img in imgs_dst:
                img_ = utils.transform_image(img, transform, config.data)
                if img_ is None:
                    return -1
                imgs_dst_.append(img_)
            imgs_dst = torch.stack(imgs_dst_).unsqueeze(0).to(device)
            preds = model(imgs_dst).cpu().data.numpy()
            mos_preds.append(preds.reshape(-1))
        return np.concatenate(mos_preds).mean().tolist()


def predict(dst_path, ref_path=None):
    """
    预测
    :param dst_path:
    :param ref_path:
    :return: 如果是返回-1 则说明是预测出错
    """
    if config.data.is_fr:
        return predict_fr(dst_path, ref_path)
    else:
        return predict_nr(dst_path)


if __name__ == '__main__':
    results = {}
    # 如果是要预测是目录，则默认将目录下所有符合条件的文件都预测
    if os.path.isdir(args.dst_path):
        for i, vname in enumerate(os.listdir(args.dst_path)):
            vpath_dst = os.path.join(args.dst_path, vname)
            if config.data.is_fr:
                results[vname] = predict(vpath_dst, os.path.join(args.ref_path, vname))
            else:
                results[vname] = predict(vpath_dst)
            print(i, vname, results[vname])
    else:
        if config.data.is_fr:
            results[args.dst_path] = predict(args.dst_path, args.ref_path)
        else:
            results[args.dst_path] = predict(args.dst_path)
        print(args.dst_path, results[args.dst_path])
    # 如果save_name 不指定，则自动生成一个
    if args.save_name is None:
        date_str = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
        save_name = date_str + ".json"
    else:
        save_name = args.save_name
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", save_name), "w") as fw:
        print("saving result at :{}".format(os.path.join("results", save_name)))
        json.dump(results, fw)
