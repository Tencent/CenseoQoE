# -*- coding: utf-8 -*-
# @Time    : 2021/5/13 11:25 上午
# @Author  : shaoguowen
# @Email   : shaoguowen@tencent.com
# @FileName: l1_loss.py
# @Software: PyCharm

import torch
import torch.nn.functional as F


class L1Loss(torch.nn.Module):
    """
    L1 loss
    """

    def __init__(self, **kwargs):
        super(L1Loss, self).__init__()
        self.loss_w = kwargs.get("loss_weight", 1)

    def forward(self, preds, gts):
        return F.l1_loss(preds.view(-1), gts.view(-1)) * self.loss_w
