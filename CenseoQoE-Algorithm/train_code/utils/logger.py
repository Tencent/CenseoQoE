# -*- coding: utf-8 -*-
# @Time    : 2020/6/15 5:27 下午
# @Author  : shaoguowen
# @Email   : shaoguowen@tencent.com
# @FileName: logger.py
# @Software: PyCharm

import logging
import os
import sys


def get_logger(exp_dir, log_name='train.log'):
    """
    设置logger
    :param exp_dir:
    :param log_name:
    :return:
    """
    os.makedirs(exp_dir, exist_ok=True)
    logger = logging.getLogger()
    formatter = logging.Formatter(
        "[%(levelname)-7s][%(asctime)s][%(filename)s:\
        %(lineno)d] %(message)s", "%d %b %Y %H:%M:%S")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(exp_dir, log_name), encoding='utf8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)
    return logger
