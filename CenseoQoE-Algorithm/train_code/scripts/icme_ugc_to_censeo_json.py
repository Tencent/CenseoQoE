# -*- coding: utf-8 -*-
# @Time    : 2021/5/13 4:30 下午
# @Author  : shaoguowen
# @Email   : shaoguowen@tencent.com
# @FileName: icme_ugc_to_censeo_json.py
# @Software: PyCharm

"""
提供一个样例: 将icme ugc数据格式转为censeo training所需要的格式
"""
import sys

sys.path.append(".")
import json
import os
from multiprocessing import Pool
import cv2
from utils import utils


def video2imgs(v_dir, vname, save_vdir):
    """
    将视频帧保存成png图片保存
    :param v_dir:
    :param vname:
    :param save_vdir:
    :return:
    """
    vpath = os.path.join(v_dir, vname)
    vgen = utils.read_video_gen(vpath, 0, -1, 10, 1)
    save_vsubdir = os.path.join(save_vdir, vname)
    os.makedirs(save_vsubdir, exist_ok=True)
    i = 1
    for frames in vgen:
        print("save image at:{}".format(os.path.join(save_vsubdir, "%05d.png" % i)))
        cv2.imwrite(os.path.join(save_vsubdir, "%05d.png" % i), frames[0])
        i += 1


def process(json_mos_path, json_dmos_path, json_train_path, json_val_path):
    """
    将外部数据格式转为censeo指定输入的json格式
    :param json_mos_path:
    :param json_dmos_path:
    :param json_train_path:
    :param json_val_path:
    :return:
    """
    with open(json_mos_path, 'r') as fin:
        data_mos = json.load(fin)
    with open(json_dmos_path, 'r') as fin:
        data_dmos = json.load(fin)
    results_train = {}
    results_val = {}
    train_dis_vnames = data_dmos["train"]["dis"]
    train_ref_vnames = data_dmos["train"]["ref"]
    test_dis_vnames = data_dmos["test"]["dis"]
    test_ref_vnames = data_dmos["test"]["ref"]
    for i, vname in enumerate(train_dis_vnames):
        vname_mp4 = vname.split(".")[0] + ".mp4"
        vname_ref_mp4 = train_ref_vnames[i].split(".")[0] + ".mp4"
        mos_i = data_mos["train"]["dis"].index(vname)
        info_r = {'ref_video': vname_ref_mp4, 'ref_dir': '',
                  'dis_dir': '', 'mos': float(data_mos["train"]["mos"][mos_i]),
                  'mos_dv': float(data_dmos["train"]["mos"][i]), 'std': None, "mos_list": None}
        results_train[vname_mp4] = info_r

    for i, vname in enumerate(test_dis_vnames):
        vname_mp4 = vname.split(".")[0] + ".mp4"
        vname_ref_mp4 = test_ref_vnames[i].split(".")[0] + ".mp4"
        mos_i = data_mos["test"]["dis"].index(vname)
        info_r = {'ref_video': vname_ref_mp4, 'ref_dir': '',
                  'dis_dir': '', 'mos': float(data_mos["test"]["mos"][mos_i]),
                  'mos_dv': float(data_dmos["test"]["mos"][i]), 'std': None, "mos_list": None}
        results_val[vname_mp4] = info_r

    print('total train video:%d' % len(results_train))
    with open(json_train_path, 'w') as fin:
        json.dump(results_train, fin)
    print('total val video:%d' % len(results_val))
    with open(json_val_path, 'w') as fin:
        json.dump(results_val, fin)
    results_train.update(results_val)
    return results_train


if __name__ == '__main__':
    VDIR = "/media/shaoguowen/shaoguo/IQA/ICME-UGC-Videos/videos"
    JSON_MOS_PATH = '/media/shaoguowen/shaoguo/IQA/ICME-UGC-Videos/ugcset_mos.json'
    JSON_DMOS_PATH = '/media/shaoguowen/shaoguo/IQA/ICME-UGC-Videos/ugcset_dmos.json'
    JSON_TRAIN_PATH = './data/ICME_UGC/icme_ugcset_processed_train.json'
    JSON_VAL_PATH = './data/ICME_UGC/icme_ugcset_processed_test.json'
    os.makedirs(os.path.dirname(JSON_TRAIN_PATH), exist_ok=True)
    results = process(JSON_MOS_PATH, JSON_DMOS_PATH, JSON_TRAIN_PATH, JSON_VAL_PATH)

    SAVE_VDIR = "/data/IVQA/public_data/ICME_UGC_frames"
    os.makedirs(SAVE_VDIR, exist_ok=True)
    pool = Pool(8)
    for vname in results:
        pool.apply_async(video2imgs, args=(VDIR, vname, SAVE_VDIR))
    pool.close()
    pool.join()
