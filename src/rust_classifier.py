import torch

from .visualize import inference
from . import config
from .faster_rcnn_renset_fpn import FasterRCNNSWinFPN
from .device import device
from PIL import Image
from typing import Tuple, List, Optional


class RustClassifier:
    classifier = None

    def get_classifer(self):
        if RustClassifier.classifier is None:
            model_weight_path = config.serve_model_weight_path
            faster_rcnn = FasterRCNNSWinFPN().to(device)
            faster_rcnn.load_state_dict(torch.load(model_weight_path))
            faster_rcnn.eval()
            RustClassifier.classifier = faster_rcnn

        return RustClassifier.classifier

    def inference_on_image(self, img, return_box=False):
        # type: (Image.Image, Optional[bool]) -> Tuple[float, List[float]]
        faster_rcnn = self.get_classifer()
        max_score, max_score_box = inference(faster_rcnn, img)

        if return_box:
            return max_score, max_score_box
        else:
            return max_score
