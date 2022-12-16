import numpy as np
import torch
import paddle
from weight_dict import d

def torch2paddle():
    torch_path = r"/tinynet_paddle2/ckpt_path/tinynet_c.pth"
    paddle_path = "tinynet_c.pdparams"
    torch_state_dict = torch.load(torch_path)
    fc_names = ["classifier"]
    paddle_state_dict = {}
    for k in torch_state_dict:
        # print(k, end=" ------ ")
        if "num_batches_tracked" in k:
            continue
        v = torch_state_dict[k].detach().cpu().numpy()
        flag = [i in k for i in fc_names]
        if any(flag) and "weight" in k: # ignore bias
            new_shape = [1, 0] + list(range(2, v.ndim))
            print(f"name: {k}, ori shape: {v.shape}, new shape: {v.transpose(new_shape).shape}")
            v = v.transpose(new_shape)
        if k not in d.keys():
            print(k)
        else:
            paddle_state_dict[d[k]] = v
    paddle.save(paddle_state_dict, paddle_path)

if __name__ == "__main__":
    torch2paddle()