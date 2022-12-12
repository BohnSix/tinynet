# 2021.01.09-Changed for main script for testing TinyNet on ImageNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import argparse
import os
import csv
import glob
import time
import logging
import paddle
import paddle.nn as nn
import torch.nn.parallel
from collections import OrderedDict

from timm.models import create_model, load_checkpoint, is_model
from timm.data import Dataset, create_loader, resolve_data_config
from timm.utils import accuracy, AverageMeter, setup_default_logging



"""
python ./test.py ../tiny-imagenet-200 --model_name=tinynet_a
"""

# torch.backends.cudnn.benchmark = True


def validate(args):
    # create model
    from tinynet import tinynet

    args.r = 0.86
    args.w = 1.0
    args.d = 1.2
    ckpt_path = './models/tinynet_a.pth'


    model = tinynet(
        r=args.r,
        w=args.w,
        d=args.d, )

    print(model)


class A:
    def __init__(self):
        self.r, self.w, self.d = 1.0, 1.0, 1.0


def main():
    # setup_default_logging()
    args = A()
    print(args)
    validate(args)


if __name__ == '__main__':
    main()
