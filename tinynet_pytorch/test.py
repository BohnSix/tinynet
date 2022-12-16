# 2021.01.09-Changed for main script for testing TinyNet on ImageNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import argparse
import os
import csv
import glob
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict

from timm.models import create_model, load_checkpoint, is_model
from timm.data import Dataset, create_loader, resolve_data_config
from timm.utils import accuracy, AverageMeter, setup_default_logging

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('--model_name', default='tinynet_c',
                    help='model architecture (default: tinynet-c)')


def validate(args):
    # create model
    from tinynet import tinynet

    args.r=0.825
    args.w=0.54
    args.d=0.85
    ckpt_path = './models/tinynet_c.pth'

    model = tinynet(
        r=args.r,
        w=args.w,
        d=args.d,)


    state_dict = torch.load(ckpt_path)
    # print(state_dict.keys())
    model.load_state_dict(state_dict, strict=True)

    print(model)

    # params = sum([param.numel() for param in model.parameters()])
    # logging.info('Model %s created, #params: %d' % (args.model_name, params))
    #
    # data_config = resolve_data_config(vars(args), model=model)
    #
    # model = model.cuda()
    # criterion = nn.CrossEntropyLoss().cuda()
    #
    # dataset = Dataset(args.data)
    # data_loader = create_loader(
    #     dataset,
    #     is_training=False,
    #     input_size=data_config['input_size'],
    #     batch_size=128,
    #     use_prefetcher=False,
    #     interpolation=data_config['interpolation'],
    #     mean=data_config['mean'],
    #     std=data_config['std'],
    #     num_workers=4,
    #     crop_pct=data_config['crop_pct'],
    #     pin_memory=False)
    #
    # losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()
    #
    # model.eval()
    # with torch.no_grad():
    #     for i, (input, target) in enumerate(data_loader):
    #         print(input.shape)
    #         input = input.cuda()
    #         target = target.cuda()
    #
    #         output = model(input)
    #         loss = criterion(output, target)
    #
    #         acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
    #         losses.update(loss.item(), input.size(0))
    #         top1.update(acc1.item(), input.size(0))
    #         top5.update(acc5.item(), input.size(0))
    #
    #         if i % 100 == 0:
    #             logging.info(
    #                 'Test: [{0:>4d}/{1}]  Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})'.format(
    #                     i, len(data_loader), loss=losses))
    #
    # logging.info(' * Acc@1 {:.3f} Acc@5 {:.3f}'.format(top1.avg, top5.avg))


def main():
    setup_default_logging()
    args = parser.parse_args()
    validate(args)


if __name__ == '__main__':
    main()
