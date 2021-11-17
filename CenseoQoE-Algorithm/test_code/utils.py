# -*- coding: utf-8 -*-
# @Time    : 2021/4/15 3:40 下午
# @Author  : shaoguowen
# @Email   : shaoguowen@tencent.com
# @FileName: utils.py
# @Software: PyCharm

import os
import cv2
import numpy as np


def transform(img, config):
    """
    图片转换
    :param img:
    :param config:
    :return:
    """
    if img is None:
        return None
    sub_img_h, sub_img_w = config.sub_img_dim
    h, w, _ = img.shape
    if config.read_mode == 'random_crop':
        h_i = (h - sub_img_h) // 2
        w_i = (w - sub_img_w) // 2
        img_sub = img[h_i:h_i + sub_img_h, w_i:w_i + sub_img_w, :].copy()
    elif config.read_mode == 'resize':
        img_sub = cv2.resize(img, dsize=(sub_img_w, sub_img_h))
    elif config.read_mode == 'resize_random_crop':
        flag = False
        resize_h, resize_w = config.resize_dim
        if (w - h) * (resize_w - resize_h) < 0:
            sub_img_w, sub_img_h = sub_img_h, sub_img_w
            flag = True
            resize_w, resize_h = resize_h, resize_w
        scale = max(resize_h / h, resize_w / w)
        sub_img_h_, sub_img_w_ = int(h * scale), int(w * scale)
        img_sub = cv2.resize(img, (sub_img_w_, sub_img_h_), cv2.INTER_CUBIC)
        w_i = (sub_img_w_ - sub_img_w) // 2
        h_i = (sub_img_h_ - sub_img_h) // 2
        img_sub = img_sub[h_i:h_i + sub_img_h, w_i:w_i + sub_img_w, :]
        if flag:
            img_sub = np.rot90(img_sub)
    elif config.read_mode == 'resize_padding':
        if (sub_img_w - sub_img_h) * (w - h) < 0:
            img = np.rot90(img)
            h, w = img.shape[:2]
        h_scale = sub_img_h / h
        w_scale = sub_img_w / w
        scale = min(h_scale, w_scale)
        img = cv2.resize(img, (int(scale * w), int(scale * h)), cv2.INTER_CUBIC)
        h_new, w_new, _ = img.shape
        top_pad = (sub_img_h - h_new) // 2
        left_pad = (sub_img_w - w_new) // 2
        bottom_pad = sub_img_h - h_new - top_pad
        right_pad = sub_img_w - w_new - left_pad
        img_sub = np.pad(img, [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)], mode='constant',
                         constant_values=0)
    else:
        print("%s is not supported yet" % config.read_mode)
        img_sub = None
    if img_sub is None:
        return img_sub
    else:
        img_sub = img_sub
        img_sub = img_sub / 255.0
        img_sub = np.transpose(img_sub, (2, 0, 1))
        return img_sub


def read_video_gen(video_path, start_time, end_time, subsample=1, batch_size=1):
    """
    yield读取视频帧
    :param video_path:视频路径
    :param start_time:开始时间
    :param end_time:结束时间
    :param subsample:采样频率（每隔多少帧采样一次）
    :param batch_size:返回视频帧的batch size
    :return:
    """
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if subsample < 1:
        # 如果subsample 小于1, 则默认没隔一秒取一帧
        subsample = int(fps)
    start_frame = start_time * fps
    end_frame = -1 if end_time == -1 else int(end_time * fps)
    frames = []
    i = 0
    while True:
        if len(frames) == batch_size:
            frames = []
        success, frame = video_capture.read()
        if not success:
            break
        if i >= start_frame and i % subsample == 0:
            frames.append(frame)
        if len(frames) == batch_size:
            yield frames
        i += 1
        if end_frame != -1 and i >= end_frame:
            break

    video_capture.release()
    if frames:
        yield frames
    else:
        yield None


def is_img(filename):
    '''
    判断是不是图片
    :param filename:
    :return:
    '''
    return isinstance(filename, str) and filename.split('.')[-1].lower() in ["png", "jpg", "jpeg", "gif"]


def is_video(filename):
    '''
    判断是不是视频
    :param filename:
    :return:
    '''
    return isinstance(filename, str) and filename.split('.')[-1].lower() in ["mp4", "mkv", "h264"]


def sigmoid(x):
    """
    sigmoid 函数
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def file_exists(file_path):
    """
    判断文件是否存在
    :param file_path:
    :return:
    """
    return isinstance(file_path, str) and os.path.exists(file_path)


class Dict2Obj(object):
    """
    dict to object
    """

    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [Dict2Obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, Dict2Obj(b) if isinstance(b, dict) else b)
