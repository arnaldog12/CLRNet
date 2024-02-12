from torch import nn
from torchvision.models.vision_transformer import (
    VisionTransformer,
    ViT_B_16_Weights,
    vit_b_16,
)

from clrnet.models.registry import BACKBONES


@BACKBONES.register_module
class VisionTransformerWrapper(nn.Module):
    def __init__(self, cfg=None):
        super(VisionTransformerWrapper, self).__init__()
        self.cfg = cfg
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    def forward(self, x):
        return self.model(x)
