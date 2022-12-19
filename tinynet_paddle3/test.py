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
# import torch.nn.parallel
from collections import OrderedDict

#from timm.models import create_model, load_checkpoint, is_model
from timm.data import Dataset, create_loader, resolve_data_config
from timm.utils import accuracy, AverageMeter, setup_default_logging
from efficientnet import EfficientNetB0
model = EfficientNetB0()

"""
python ./test.py ../tiny-imagenet-200 --model_name=tinynet_a
"""

# torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model_name', default='tinynet-a',
                    help='model architecture (default: tinynet-a)')



# def validate(args):
#
#     from tinynet import tinynet
#
#     args.r = 0.86
#     args.w = 1.0
#     args.d = 1.2
#     ckpt_path = './ckpt_path/tinynet_c.pth'
#
#
#     model = tinynet(
#         r=args.r,
#         w=args.w,
#         d=args.d, )
#
#     print(model)
#
#     state_dict = paddle.load(ckpt_path)
#     model.load_state_dict(state_dict, strict=False)

layer_state_dict = model.state_dict()

# for k in layer_state_dict:
#     print(k)
pd_pd = paddle.load("tinynet_c.pdparams")

model.set_state_dict(pd_pd)


params = sum([param.numel() for param in model.parameters()])
logging.info('Model %s created, #params: %d' % (args.model_name, params))

data_config = resolve_data_config(vars(args), model=model)

model = model.cuda()
criterion = nn.CrossEntropyLoss().cuda()

dataset = Dataset(args.data)
data_loader = create_loader(
    dataset,
    is_training=False,
    input_size=data_config['input_size'],
    batch_size=128,
    use_prefetcher=False,
    interpolation=data_config['interpolation'],
    mean=data_config['mean'],
    std=data_config['std'],
    num_workers=4,
    crop_pct=data_config['crop_pct'],
    pin_memory=False)

losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()

model.eval()
with paddle.no_grad():
    for i, (input, target) in enumerate(data_loader):
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        if i % 100 == 0:
            logging.info(
                'Test: [{0:>4d}/{1}]  Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})'.format(
                    i, len(data_loader), loss=losses))

    logging.info(' * Acc@1 {:.3f} Acc@5 {:.3f}'.format(top1.avg, top5.avg))


def main():
    # setup_default_logging()
    # args = A()
    # print(args)
    # validate(args)

    setup_default_logging()
    args = parser.parse_args()
    EfficientNetB0(args)


if __name__ == '__main__':
    main()
