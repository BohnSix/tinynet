import argparse
import torch

from tinynet import tinynet

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')

args = parser.parse_args()
ckpt_path = './models/tinynet_c.pth'

args.r = 1.
args.w = 1.
args.d = 1.
model = tinynet(
    r=args.r,
    w=args.w,
    d=args.d, )

state_dict = torch.load(ckpt_path)
# print(state_dict.keys())
model.load_state_dict(state_dict, strict=True)

print(model)

