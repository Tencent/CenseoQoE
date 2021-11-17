# -*- coding: utf-8 -*-
# @Time    : 2021/5/13 11:03 上午
# @Author  : shaoguowen
# @Email   : shaoguowen@tencent.com
# @FileName: utils.py
# @Software: PyCharm
import numpy as np
import cv2
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
import torch


class UnNormalizer(object):
    """
    反归一化
    """

    def __init__(self, mean=None, std=None):
        if mean is None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std is None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def plot_corr(y_pred, y_gt, srocc=0., plcc=0., dataset="censeo_ivqa_data"):
    """
    画SROCC和PLCC的相关系数
    :param y_pred:
    :param y_gt:
    :param srocc:
    :param plcc:
    :param dataset:
    :return:
    """
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    ax.scatter(y_pred, y_gt)
    ax.set_xlabel("y_pred")
    ax.set_xlim(0.2, 1)
    ax.set_ylabel("y_gt")
    ax.set_ylim(0.2, 1)
    ax.plot([0, 1], [0, 1])
    ax.set_title('%s: srocc:%0.2f, plcc:%0.2f' % (dataset, srocc, plcc))
    # Force a draw so we can grab the pixel buffer
    canvas.draw()
    # grab the pixel buffer and dump it into a numpy array
    image = np.array(canvas.renderer.buffer_rgba())[:, :, :3]
    image = np.transpose(image, (2, 0, 1))
    return image


def plot_image(vis, imgs, imgs_pred, imgs_gt, mode='train'):
    """
    画图
    :param vis: visualize
    :param imgs:
    :param imgs_pred:
    :param imgs_gt:
    :param mode:
    :return:
    """
    unorm = UnNormalizer(mean=[0., 0., 0.], std=[1, 1, 1])  # 这里的参数要根据在dataset类里的transform来设置
    imgs_pred = imgs_pred.view(-1).cpu().data.numpy()
    imgs_gt = imgs_gt.view(-1).cpu().data.numpy()
    with torch.no_grad():
        for i in range(imgs.size(0)):
            img = unorm(imgs.data[i].cpu()).numpy().copy()
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            img = np.transpose(img, [1, 2, 0])
            r, g, b = cv2.split(img)
            img = cv2.merge([b, g, r])
            img = np.transpose(img, [2, 0, 1])
            vis.img('%s_%d' % (mode, i), img, title="%s, pred:%.2f, gt:%.2f" % (mode, imgs_pred[i], imgs_gt[i]))


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
    return isinstance(filename, str) and filename.split('.')[-1].lower() in ["mp4", "yuv", "mkv", "h264"]


def read_video_gen(video_path, start_time, end_time, subsample, batch_size):
    """
    yield 批量读取视频帧
    :param video_path:
    :param start_time:
    :param end_time:
    :param subsample: 每个多少帧读取一次，subsample=1时代表每一帧都读
    :param batch_size:
    :return:
    """
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    start_frame = start_time * fps
    end_frame = -1 if end_time == -1 else end_time * fps
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


def transform_image(img, transform, config):
    """
    读取图片并进行转换
    :param img_path:
    :return:
    """
    if img is None:
        return None
    sub_img_h, sub_img_w = config.sub_img_dim
    h, w, _ = img.shape
    if config.read_mode == 'resize':
        img_sub = cv2.resize(img, dsize=(sub_img_w, sub_img_h))
    elif config.read_mode == 'random_crop':
        h_i = (h - sub_img_h) // 2
        w_i = (w - sub_img_w) // 2
        img_sub = img[h_i:h_i + sub_img_h, w_i:w_i + sub_img_w, :].copy()
    elif config.read_mode == 'resize_random_crop':
        resize_h, resize_w = config.resize_dim
        flag = False
        if (w - h) * (resize_w - resize_h) < 0:
            flag = True
            resize_w, resize_h = resize_h, resize_w
            sub_img_w, sub_img_h = sub_img_h, sub_img_w
        w_scale = resize_w / w
        h_scale = resize_h / h
        scale = max(h_scale, w_scale)
        sub_img_h_, sub_img_w_ = int(h * scale), int(w * scale)
        img_sub = cv2.resize(img, (sub_img_w_, sub_img_h_), cv2.INTER_CUBIC)
        h_i = (sub_img_h_ - sub_img_h) // 2
        w_i = (sub_img_w_ - sub_img_w) // 2
        img_sub = img_sub[h_i:h_i + sub_img_h, w_i:w_i + sub_img_w, :]
        if flag:
            img_sub = np.rot90(img_sub)
    elif config.read_mode == 'resize_padding':
        if (w - h) * (sub_img_w - sub_img_h) < 0:
            img = np.rot90(img)
            h, w = img.shape[:2]
        w_scale = sub_img_w / w
        h_scale = sub_img_h / h
        scale = min(h_scale, w_scale)
        img = cv2.resize(img, (int(scale * w), int(scale * h)), cv2.INTER_CUBIC)
        h_new, w_new, _ = img.shape
        top_pad = (sub_img_h - h_new) // 2
        bottom_pad = sub_img_h - h_new - top_pad
        left_pad = (sub_img_w - w_new) // 2
        right_pad = sub_img_w - w_new - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        img_sub = np.pad(img, padding, mode='constant', constant_values=0)
    else:
        print("%s is not supported yet" % config.read_mode)
        img_sub = None
    if img_sub is None:
        return None
    else:
        return transform(img_sub.copy()).float()


def get_lr(optimizer):
    """
    从optimizer获取lr
    :param optimizer:
    :return:
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
