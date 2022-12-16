import paddle

from ppcls.arch.backbone.model_zoo.efficientnet import EfficientNet


def EfficientNetMy(padding_type='SAME',
                   override_params=None,
                   use_se=True,
                   pretrained=False,
                   use_ssld=False,
                   **kwargs):
    model = EfficientNet(
        name='my',
        padding_type=padding_type,
        override_params=override_params,
        use_se=use_se,
        **kwargs)
    return model

model = EfficientNetMy()
layer_state_dict = model.state_dict()

# for k in layer_state_dict:
#     print(k)
pd_pd = paddle.load("tinynet_c.pdparams")

model.set_state_dict(pd_pd)