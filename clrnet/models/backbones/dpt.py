import torch
from torch import nn
from torchvision.models.vision_transformer import VisionTransformer
from transformers import DPTImageProcessor, DPTModel

from clrnet.models.registry import BACKBONES

PRETRAINED_CHECKPOINT = "Intel/dpt-large"


@BACKBONES.register_module
class DPTWrapper(nn.Module):
    def __init__(self, cfg=None):
        super(DPTWrapper, self).__init__()
        self.cfg = cfg
        self.processor = DPTImageProcessor.from_pretrained(PRETRAINED_CHECKPOINT)
        self.model = DPTModel.from_pretrained(PRETRAINED_CHECKPOINT)

    def forward(self, x):
        inputs = self.processor(images=x, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
        return last_hidden_state[:, 1:, :].reshape(-1, 24, 24, 1024).permute(0, 3, 1, 2)
