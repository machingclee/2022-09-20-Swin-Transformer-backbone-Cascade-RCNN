import torch
from src.train import train_with_nan
from src.cascade_faster_rcnn_swin_transformer_fpn import CascadeFasterRCNNSWinFPN
from src.faster_rcnn_swin_transformer_fpn import FasterRCNNSWinFPN
from src.device import device
from src.rust_classifier import RustClassifier
from PIL import Image


def main():
    model_path = "pths/model_epoch_1.pth"
    model = CascadeFasterRCNNSWinFPN().to(device)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    model.train()

    train_with_nan(
        model,
        lr=1e-5,
        start_epoch=2,
        epoches=31,
        save_weight_interval=1
    )


if __name__ == "__main__":
    main()
