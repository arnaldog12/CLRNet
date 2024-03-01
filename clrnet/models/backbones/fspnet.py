from torch import nn

from clrnet.models.backbones.fspnet_model.FSPNet_model import FSPNet
from clrnet.models.registry import BACKBONES


@BACKBONES.register_module
class FSPNetWrapper(nn.Module):
    def __init__(self, cfg=None):
        super(FSPNetWrapper, self).__init__()
        self.cfg = cfg
        self.model = FSPNet(cfg.ckpt, cfg.img_h)

    def forward(self, x):
        return self.model.forward(x)
