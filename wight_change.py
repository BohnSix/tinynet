import torch
import paddle
import os
from collections import OrderedDict
from PaddleClas.model_weight_torch2pd import model as pd_model
import numpy as np

def export_weight_names(net):
    print(net.state_dict().keys())
    with open('paddle.txt', 'w') as f:
        for key in net.state_dict().keys():
            f.write(key + '\n')

res2net_paddle_implement=pd_model
export_weight_names(res2net_paddle_implement)  # 将自己paddle模型的keys存为txt
paddle_list = open('paddle.txt') # paddle的keys
state_dict = torch.load(r"D:\PycharmProjects\Efficient-AI-Backbones\tinynet_paddle2\ckpt_path\tinynet_c.pth")

paddle_state_dict = OrderedDict()
paddle_list = paddle_list.readlines()
torch_list = state_dict.keys()
for p in paddle_list:
    p = p.strip()
    t = p
    if "mean" in p:
        t = p.replace("_mean", "running_mean")
    if "variance" in p:
        t = p.replace("_variance", "running_var")
    if t in torch_list:
        if 'fc' not in p:
            paddle_state_dict[p] = state_dict[t].detach().cpu().numpy()
        else:
            paddle_state_dict[p] = state_dict[t].detach().cpu().numpy().T
    else:
        print(p)

f = open('tinynet_c.pdparams', 'wb')
import pickle
pickle.dump(paddle_state_dict, f)
f.close()
