# -*- coding: utf-8 -*-
# @Time    : 2021/5/12 2:06 下午
# @Author  : shaoguowen
# @Email   : shaoguowen@tencent.com
# @FileName: censeo_ivqa_model.py
# @Software: PyCharm

import torch
from torch import nn
from . import heads
from . import backbones


class CenseoIVQAModel(nn.Module):
    """
    censeo的一个strong baseline model
    """

    def __init__(self, config):
        super(CenseoIVQAModel, self).__init__()
        self.config = config
        input_channels = 6 if config.fr else 3
        self.backbone = getattr(backbones, self.config.backbone.model_name)(input_channels=input_channels,
                                                                            **self.config.backbone.kwargs)
        self.head = getattr(heads, self.config.head.model_name)(self.backbone.ouput_dims,
                                                                **self.config.head.kwargs)

    def forward(self, x):
        if self.config.fr:
            x_dst, x_ref = x.chunk(2, 2)
            x = torch.cat([x_dst, x_ref - x_dst], 2)
        n, t, c, h, w = x.size()
        x = x.view(n * t, c, h, w)
        feats = self.backbone(x)
        out = self.head(feats)
        if self.config.use_sigmoid:
            out = torch.sigmoid(out)
        out = out.view(n, t, -1)
        return out

    def forward_onnx(self, x):
        """
        专门给模型转换时使用
        :return:
        """
        if self.config.fr:
            x_dst, x_ref = x.chunk(2, 1)
            x = torch.cat([x_dst, x_ref - x_dst], 1)
        feats = self.backbone(x)
        out = self.head(feats)
        if self.config.use_sigmoid:
            out = torch.sigmoid(out)
        return out
