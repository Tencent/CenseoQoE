# -*- coding: utf-8 -*-
# @Time    : 2021/5/12 2:48 下午
# @Author  : shaoguowen
# @Email   : shaoguowen@tencent.com
# @FileName: train.py
# @Software: PyCharm

import argparse
from mmcv import Config
import trainers

# 解析传入的参数
parser = argparse.ArgumentParser(description='Train IVQA model')
parser.add_argument('config', help='train config file path')
args = parser.parse_args()
config = Config.fromfile(args.config)

if __name__ == '__main__':
    trainer = getattr(trainers, config.trainer.trainer_name)(config, args.config)
    trainer.run()
