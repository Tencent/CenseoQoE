# -*- coding: utf-8 -*-
# @Time    : 2021/5/12 2:44 下午
# @Author  : shaoguowen
# @Email   : shaoguowen@tencent.com
# @FileName: base_dataset.py
# @Software: PyCharm

import os
import random
import math
import json
import torch
from torch.utils import data
import numpy as np
import torchvision.transforms as transforms
import cv2


class ImageVideoGeneralDataset(data.Dataset):
    '''
    建立一个通用的图像、视频通用数据集loader
    适用图像/视频
    适用有参/无参

    '''

    def __init__(self, config, logger, mode='train'):
        super(ImageVideoGeneralDataset, self).__init__()
        self.config = config
        self.mode = mode  # 数据模式: train/val/test
        self.logger = logger  # 用于记录

        # transforms 来做数据的扩增/转换，这里只简单使用ToTensor，可根据自己的需求增加
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        # 获取数据的信息
        assert isinstance(self.config.data_infos, list)  # 设置data_infos是list的形式是为了可能需要多个数据集混合训练
        self.dis_paths = []
        self.ref_paths = []
        self.moses = []

        for i, data_info_path in enumerate(self.config.data_infos):
            self.logger.info("%s->处理数据: %s >>>>>>>>>>>" % (mode, data_info_path))
            with open(data_info_path, 'r') as fin:
                data_info = json.load(fin)
            # 注意要根据参考图片/视频来做数据的划分
            ref_names = sorted(list(set([data_info[key]['ref_video'] for key in data_info])))
            random.seed(config.split_seed)
            random.shuffle(ref_names)
            self.logger.info('总的场景数量:%d' % len(ref_names))
            test_num = math.ceil(self.config.test_p[i] * len(ref_names))
            train_num = math.ceil(self.config.train_p[i] * len(ref_names))
            if self.mode == "train":
                ref_names_select = ref_names[test_num:test_num + train_num]
            else:
                ref_names_select = ref_names[:test_num]
            self.logger.info('%s 场景数量:%d' % (mode, len(ref_names_select)))

            for dis_name in data_info:
                if data_info[dis_name]['ref_video'] in ref_names_select:
                    dis_p = os.path.join(self.config.data_roots[i], data_info[dis_name]['dis_dir'], dis_name)
                    ref_p = os.path.join(self.config.data_roots[i], data_info[dis_name]['ref_dir'],
                                         data_info[dis_name]['ref_video'])
                    # 确保文件存在
                    if not os.path.exists(dis_p) or (self.config.is_fr and not os.path.exists(ref_p)):
                        continue
                    self.dis_paths.append(dis_p)
                    self.ref_paths.append(ref_p)
                    self.moses.append(data_info[dis_name][self.config.mos_type] * self.config.mos_scales[i])
            self.logger.info('%s 样本数量:%d' % (mode, len(self.dis_paths)))
            self.logger.info("Max mos: %03f" % (np.max(self.moses) if self.moses else 0))

    def read_image(self, img_path, hseed=0, wseed=0):
        """
        读取图片并做相应的转换
        :param img_path:
        :param hseed:
        :param wseed:
        :return:
        """
        sub_img_h, sub_img_w = self.config.sub_img_dim
        img = cv2.imread(img_path)
        random.seed(hseed + wseed)
        if random.random() > 0.5 and self.mode == 'train':
            img = cv2.flip(img, 1)
        h, w = img.shape[:2]
        if self.config.read_mode == 'random_crop':
            if self.mode == 'train':
                random.seed(hseed)
                h_i = random.randint(0, h - sub_img_h)
                random.seed(wseed)
                w_i = random.randint(0, w - sub_img_w)
                img_sub = img[h_i:h_i + sub_img_h, w_i:w_i + sub_img_w, :]
            else:
                h_i = (h - sub_img_h) // 2
                w_i = (w - sub_img_w) // 2
                img_sub = img[h_i:h_i + sub_img_h, w_i:w_i + sub_img_w, :]
            return img_sub
        elif self.config.read_mode == 'resize_random_crop':
            resize_h, resize_w = self.config.resize_dim
            flag = False
            if (w - h) * (resize_w - resize_h) < 0:
                flag = True
                resize_w, resize_h = resize_h, resize_w
                sub_img_w, sub_img_h = sub_img_h, sub_img_w

            # 注意只能等比例缩放
            w_scale = resize_w / w
            h_scale = resize_h / h
            scale = max(h_scale, w_scale)
            img = cv2.resize(img, (int(scale * w), int(scale * h)), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]
            if self.mode == 'train':
                random.seed(hseed)
                h_i = random.randint(0, h - sub_img_h)
                random.seed(wseed)
                w_i = random.randint(0, w - sub_img_w)
                img_sub = img[h_i:h_i + sub_img_h, w_i:w_i + sub_img_w, :]
            else:
                h_i = (h - sub_img_h) // 2
                w_i = (w - sub_img_w) // 2
                img_sub = img[h_i:h_i + sub_img_h, w_i:w_i + sub_img_w, :]
            if flag:
                img_sub = np.rot90(img_sub)
            return img_sub
        elif self.config.read_mode == 'resize_padding':
            if (sub_img_w - sub_img_h) * (w - h) < 0:
                img = np.rot90(img)
                h, w = img.shape[:2]
            scale = min(sub_img_w / w, sub_img_h / h)
            img = cv2.resize(img, (int(scale * w), int(scale * h)), cv2.INTER_CUBIC)
            h_new, w_new, _ = img.shape
            left_pad = (sub_img_w - w_new) // 2
            right_pad = sub_img_w - w_new - left_pad
            top_pad = (sub_img_h - h_new) // 2
            bottom_pad = sub_img_h - h_new - top_pad
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            img = np.pad(img, padding, mode='constant', constant_values=0)
            return img
        else:
            self.logger.error("%s is not supported yet, do nothing" % self.config.read_mode)
            return img

    def __len__(self):
        return len(self.dis_paths)

    def __getitem__(self, i):
        hseed = np.random.randint(0, 1e8)
        wseed = np.random.randint(0, 1e8)
        dis_path = self.dis_paths[i]
        ref_path = self.ref_paths[i]
        # 如果是dis_path是目录路径，则说明是对视频来加载数据
        if os.path.isdir(dis_path):
            clip_len = self.config.train_clip_len if self.mode == "train" else self.config.test_clip_len
            # 确保在extract视频帧的时候保存成png的格式，请勿转成jpg/jpeg的格式，会带来压缩损失
            frames_len = len([img_name for img_name in os.listdir(dis_path) if img_name.lower().endswith(".png")])
            if frames_len < clip_len:
                # self.logger.warn(dis_path + ": do not have enough frames!")
                return self.__getitem__(random.randint(0, self.__len__() - 1))
            clip_len = min(frames_len, clip_len)
            if self.mode != "train":
                fstart = (frames_len - clip_len) // 2
            else:
                fstart = random.randint(0, frames_len - clip_len)
            imgs_dis = []
            imgs_ref = []
            for j in range(clip_len):
                # 所有的帧序号应该有对应的数字记录(00001.png, 00002.png ...)
                img_dis_path = os.path.join(dis_path, "%05d.png" % (fstart + j + 1))
                img_ref_path = os.path.join(ref_path, "%05d.png" % (fstart + j + 1))
                assert os.path.exists(img_dis_path)
                imgs_dis.append(self.read_image(img_dis_path, hseed, wseed))
                if self.config.is_fr:
                    assert os.path.exists(img_ref_path)
                    imgs_ref.append(self.read_image(img_ref_path, hseed, wseed))

            imgs_dis = [self.transform(img.copy()) for img in imgs_dis]
            img_mos = self.moses[i]
            if self.config.is_fr:
                imgs_ref = [self.transform(img.copy()) for img in imgs_ref]
                return torch.cat((torch.stack(imgs_dis).float(), torch.stack(imgs_ref).float()),
                                 1), torch.FloatTensor([img_mos])
            else:
                return torch.stack(imgs_dis).float(), torch.FloatTensor([img_mos])
        else:
            # 图片
            assert os.path.exists(dis_path)
            img_dis = self.read_image(dis_path, hseed, wseed)
            img_dis = self.transform(img_dis.copy())
            img_mos = self.moses[i]
            if self.config.is_fr:
                assert os.path.exists(ref_path)
                img_ref = self.read_image(ref_path, hseed, wseed)
                img_ref = self.transform(img_ref.copy())
                return torch.cat((torch.stack([img_dis]).float(), torch.stack([img_ref]).float()),
                                 1), torch.FloatTensor([img_mos])
            else:
                return torch.stack([img_dis]).float(), torch.FloatTensor([img_mos])
