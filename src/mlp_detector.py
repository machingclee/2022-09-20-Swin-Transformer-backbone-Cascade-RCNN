import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary
from . import config


class MLPDetector(nn.Module):
    def __init__(self, in_channels=512):
        super(MLPDetector, self).__init__()
        self.in_channels = in_channels
        self.mlp_head = nn.Sequential(
            nn.Linear(in_channels * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.cls_score = nn.Linear(4096, config.n_classes)
        self.bbox_pred = nn.Linear(4096, config.n_classes * 4)

    def forward(self, pooling):
        x = pooling.reshape(-1, self.in_channels * 7 * 7)
        x = self.mlp_head(x)
        scores_logits = self.cls_score(x)
        deltas = self.bbox_pred(x)
        return scores_logits, deltas
