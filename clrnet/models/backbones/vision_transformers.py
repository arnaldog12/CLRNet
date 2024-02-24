from torch import nn
from torchvision.models.vision_transformer import (
    VisionTransformer,
    ViT_B_16_Weights,
    vit_b_16,
)

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
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    def forward(self, x):
        self.model.encoder.layers.encoder_layer_0.register_forward_hook(
            get_activation("encoder_layer_0")
        )
        self.model.encoder.layers.encoder_layer_1.register_forward_hook(
            get_activation("encoder_layer_1")
        )
        self.model.encoder.layers.encoder_layer_2.register_forward_hook(
            get_activation("encoder_layer_2")
        )
        self.model.encoder.layers.encoder_layer_3.register_forward_hook(
            get_activation("encoder_layer_3")
        )
        output = self.model(x)
        return [
            activation["encoder_layer_0"][:, 1:, :]
            .reshape(-1, 14, 14, 768)
            .permute(0, 3, 1, 2),
            activation["encoder_layer_1"][:, 1:, :]
            .reshape(-1, 14, 14, 768)
            .permute(0, 3, 1, 2),
            activation["encoder_layer_2"][:, 1:, :]
            .reshape(-1, 14, 14, 768)
            .permute(0, 3, 1, 2),
            activation["encoder_layer_3"][:, 1:, :]
            .reshape(-1, 14, 14, 768)
            .permute(0, 3, 1, 2),
        ]
