import torch
from src.train import train_with_nan
from src.faster_rcnn_swin_transformer_fpn import FasterRCNNSWinFPN
from src.device import device
from src.rust_classifier import RustClassifier
from PIL import Image


def main():
    model_path = None
    faster_rcnn = FasterRCNNSWinFPN().to(device)

    if model_path is not None:
        faster_rcnn.load_state_dict(torch.load(model_path))

    faster_rcnn.train()

    train_with_nan(
        faster_rcnn,
        lr=1e-5,
        start_epoch=1,
        epoches=40,
        save_weight_interval=1
    )


if __name__ == "__main__":
    main()
