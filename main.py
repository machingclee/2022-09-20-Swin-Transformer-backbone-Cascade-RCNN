from re import L
import torch
import torch.nn as nn
from src.train import train_with_nan
from src.cascade_faster_rcnn import CascadeFasterRCNN
from src.faster_rcnn_swin_transformer_fpn import FasterRCNNSWinFPN
from src.feature_extractor import Resnet50FPNFeactureExtractor
from src.faster_rcnn_resnet50_fpn import FasterRCNNResnet50FPN
from src.device import device
from src.rust_classifier import RustClassifier
from PIL import Image


def build_model():
    model = CascadeFasterRCNN(
        fpn_feature_extractor=Resnet50FPNFeactureExtractor().to(device),
        fpn_feat_channels=Resnet50FPNFeactureExtractor.fpn_feat_channels
    ).to(device)
    return model


def main():
    model_path = None
    model = build_model()

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    model.train()

    train_with_nan(
        model,
        build_model,
        lr=1e-5,
        start_epoch=1,
        epoches=31,
        save_weight_interval=1
    )


if __name__ == "__main__":
    main()
