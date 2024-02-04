from torch import nn
from torchvision.models.vision_transformer import VisionTransformer, vit_b_16

from clrnet.models.registry import BACKBONES


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


@BACKBONES.register_module
class VisionTransformerWrapper(nn.Module):
    def __init__(self, cfg=None):
        super(VisionTransformerWrapper, self).__init__()
        self.cfg = cfg
        self.model = vit_b_16()

    def forward(self, x):
        self.model.encoder.layers.encoder_layer_0.register_forward_hook(get_activation('encoder_layer_0'))
        self.model.encoder.layers.encoder_layer_1.register_forward_hook(get_activation('encoder_layer_1'))
        self.model.encoder.layers.encoder_layer_2.register_forward_hook(get_activation('encoder_layer_2'))
        self.model.encoder.layers.encoder_layer_3.register_forward_hook(get_activation('encoder_layer_3'))
        output = self.model(x)
        return [
            activation['encoder_layer_0'][:, None, :, :],
            activation['encoder_layer_1'][:, None, :, :],
            activation['encoder_layer_2'][:, None, :, :],
            activation['encoder_layer_3'][:, None, :, :],
        ]
