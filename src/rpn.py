import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary
from .device import device
from .anchor_generator import AnchorGenerator
from .box_utils import decode_deltas_to_boxes
from .context_block import context_block2d
from . import config
from typing import cast
from torch import Tensor


class RPNHead(nn.Module):
    def __init__(self, in_channel):
        super(RPNHead, self).__init__()
        n_anchors = len(config.anchor_ratios) * len(config.anchor_scales)
        self.conv1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.conv_logits = nn.Conv2d(in_channel, n_anchors * 2, 1, 1)
        self.conv_deltas = nn.Conv2d(in_channel, n_anchors * 4, 1, 1)
        self.weight_init()

    def weight_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, feature):
        feature = F.relu(self.conv1(feature))
        logits = self.conv_logits(feature)
        deltas = self.conv_deltas(feature)
        return logits, deltas


class RPN(nn.Module):
    def __init__(self, in_channel):
        super(RPN, self).__init__()
        self.multi_scale_anchors = AnchorGenerator().get_multi_scale_anchors()
        self.flattened_multi_scale_anchors = AnchorGenerator().get_flattened_multi_scale_anchors()
        self.rpn_head = RPNHead(in_channel)

    def forward(self, features):
        batch_size = features.shape[0]
        logits, deltas = self.rpn_head(features)

        logits = logits.permute(0, 2, 3, 1).reshape(batch_size, -1, 2)
        deltas = deltas.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        return logits, deltas
