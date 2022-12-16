from collections import OrderedDict
import torch

if __name__ == '__main__':
    model_path = r"D:\PycharmProjects\Efficient-AI-Backbones\tinynet_paddle2\ckpt_path\tinynet_c.pthh"
    model = resnet18(False)
    torch_state_dict = torch.load(model_path)
    paddle_state_dict = model.state_dict()
    new_state_dict = OrderedDict()
    # torch_key = [(key,val) for key,val in torch_state_dict.items()]
    # paddle_key = [(key,val) for key,val in model.state_dict().items()]
    torch_keys = [key for key, val in torch_state_dict.items()]
    paddle_keys = [key for key, val in paddle_state_dict.items()]

    assert len(torch_keys) == len(paddle_keys)

    for paddle_key in paddle_keys:
        # bn层命名区别
        if '_mean' in paddle_key:
            torch_key = paddle_key.replace('_mean', 'running_mean')
        elif '_variance' in paddle_key:
            torch_key = paddle_key.replace('_variance', 'running_var')
        else:
            torch_key = paddle_key
        paddle_val = torch_state_dict[torch_key].detach().numpy().astype('float32')
        # fc层参数转置
        if 'fc' in paddle_key and 'weight' in paddle_key:
            paddle_val = paddle_val.T
        new_state_dict[paddle_key] = paddle_val

    model.set_state_dict(new_state_dict)
    paddle.save(model.state_dict(), "res18.pdparams")