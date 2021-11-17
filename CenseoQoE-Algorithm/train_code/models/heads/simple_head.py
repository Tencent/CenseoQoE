# -*- coding: utf-8 -*-
# @Time    : 2021/5/12 11:55 上午
# @Author  : shaoguowen
# @Email   : shaoguowen@tencent.com
# @FileName: simple_head.py
# @Software: PyCharm

import torch
from torch import nn


class SimpleHead(torch.nn.Module):
    """
    一个简单的模型预测头部
    """

    def __init__(self, feats_dims, **kwargs):
        super(SimpleHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        out_num = kwargs.get("out_num", 1)
        self.fc_out = nn.Sequential(nn.Linear(feats_dims[-1], 1024),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(1024, out_num))

    def forward(self, x):
        x = x[-1]
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc_out(x)
        return x
