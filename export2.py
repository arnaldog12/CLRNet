import torch

from clrnet.models.registry import build_net
from clrnet.utils.config import Config
from clrnet.utils.net_utils import load_network

config_path = "configs/clrnet/clr_resnet18_culane.py"
model_path = "C:/Users/arnal/Downloads/ResNet18_CUlane.pth"

cfg = Config.fromfile(config_path)

net = build_net(cfg)
load_network(net, model_path)

torch_input = torch.randn(1, 3, cfg.img_h, cfg.img_w)
onnx_program = torch.onnx.export(net, torch_input, "my_image_classifier.onnx", verbose=True, export_params=False)
