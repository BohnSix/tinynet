import paddle
from efficientnet import EfficientNet

def EfficientNetmy(padding_type='SAME',
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
    # _load_pretrained(pretrained, model, MODEL_URLS["EfficientNetB0"])
    return model

model = EfficientNetmy()
layer_state_dict = model.state_dict()
for k in layer_state_dict:
    print(k)