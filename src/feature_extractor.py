from turtle import forward
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Any
from . import config
from .device import device
from typing import OrderedDict


class SwinFeatureExtractor(nn.Module):
    def __init__(self):
        super(SwinFeatureExtractor, self).__init__()
        self.model = models.swin_t(weights="DEFAULT").to(device)
        self.layer1 = self.model.features[0:3]
        self.layer2 = self.model.features[3:5]
        self.layer3 = self.model.features[5:8]
        self.lateral_conv4 = nn.Conv2d(768, config.fpn_feat_channels, 1, 1)
        self.lateral_conv3 = nn.Conv2d(384, config.fpn_feat_channels, 1, 1)
        self.lateral_conv2 = nn.Conv2d(192, config.fpn_feat_channels, 1, 1)
        self.upscale = lambda input: F.interpolate(input, scale_factor=2)
        self.freeze_params()

    def freeze_params(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x_4: Tensor = self.layer1(x)
        x_8 = self.layer2(x_4)
        x_16 = self.layer3(x_8)

        x_4 = x_4.permute([0, 3, 1, 2])
        x_8 = x_8.permute([0, 3, 1, 2])
        x_16 = x_16.permute([0, 3, 1, 2])

        p4 = self.lateral_conv4(x_16)
        p3 = self.lateral_conv3(x_8) + self.upscale(p4)
        p2 = self.lateral_conv2(x_4) + self.upscale(p3)

        return [p2, p3, p4]


class ResnetFPNFeactureExtractor(nn.Module):
    def __init__(self):
        super(ResnetFPNFeactureExtractor, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)

        self.conv2 = nn.Sequential(
            self.resnet50.conv1,
            self.resnet50.bn1,
            self.resnet50.relu,
            self.resnet50.maxpool,
            self.resnet50.layer1
        )
        self.conv3 = self.resnet50.layer2
        self.conv4 = self.resnet50.layer3
        self.conv5 = self.resnet50.layer4

        self.lateral_conv5 = nn.Conv2d(2048, config.fpn_feat_channels, 1, 1)
        self.lateral_conv4 = nn.Conv2d(1024, config.fpn_feat_channels, 1, 1)
        self.lateral_conv3 = nn.Conv2d(512, config.fpn_feat_channels, 1, 1)
        self.lateral_conv2 = nn.Conv2d(256, config.fpn_feat_channels, 1, 1)

        self.upscale = lambda input: F.interpolate(input, scale_factor=2)
        self.freeze_params()

    def freeze_params(self):
        modules = [
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5
        ]
        for module in modules:
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    for param in layer.parameters():
                        param.requires_grad = False

    def forward(self, x):
        c2 = self.conv2(x)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        p5 = self.lateral_conv5(c5)
        p4 = self.lateral_conv4(c4) + self.upscale(p5)
        p3 = self.lateral_conv3(c3) + self.upscale(p4)
        p2 = self.lateral_conv2(c2) + self.upscale(p3)

        return [p2, p3, p4, p5]


if __name__ == "__main__":
    t = torch.randn((3, 3, 224, 224)).to(device)
    feat_extractor = SwinFeatureExtractor()
    out = feat_extractor(t)
    print(out.shape)
