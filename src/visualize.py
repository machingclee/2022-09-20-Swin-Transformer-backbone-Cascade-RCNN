from optparse import Option
import torch
import torch.nn as nn
import numpy as np
from .faster_rcnn_swin_transformer_fpn import FasterRCNNSWinFPN as FasterRCNN
from .dataset import AnnotationDataset, torch_img_transform
from glob import glob
from torch.utils.data import DataLoader
from PIL import ImageDraw, Image, ImageFont
from copy import deepcopy
from .device import device
from . import config
from .dataset import resize_and_padding
from torchvision.ops import nms
from typing import TypedDict, List, Optional, Tuple

class DetectionResult(TypedDict, total=False):
    label: List[str]
    bbox: List[List[float]]
    score: List[List[float]]

def inference(fast_rcnn: nn.Module = None, pillow_img=None):
    # type: (Optional[nn.Module], Optional[Image.Image]) -> Tuple[float, List[float]]
    """
    return: 
        max_score: float
        max_score_box: List[float] in terms of [xmin, ymin, xmax, ymax]
    """
    img = pillow_img
        
    img, padding_window, (ori_w, ori_h) = resize_and_padding(img, return_window=True)
    x_scale = ori_w/padding_window[0]
    y_scale = ori_h/padding_window[1]
    box_scaling = torch.as_tensor([x_scale, y_scale, x_scale, y_scale]).to(device)
    img = torch_img_transform(img)
    scores, boxes, cls_idxes, rois = fast_rcnn(img[None,...])
    boxes = boxes * box_scaling
    
    if len(boxes) == 0:
        return 0, []
    # draw = ImageDraw.Draw(img_original_size)
    # for score, box, cls_idx in zip(scores, boxes, cls_idxes):
    #     xmin, ymin, xmax, ymax=box
    #     draw.rectangle(((xmin, ymin), (xmax, ymax)), outline = 'blue', width = 3)
    #     draw.text(
    #         (xmin, max(ymin - 20, 4)),
    #         "{}: {:.2f}".format(cls_names[cls_idx.item()], score.item()), "blue",
    #         font=font
    #     )
    # img_original_size.show()
    
    max_score_index = torch.argmax(scores).detach().cpu().item()
    max_score = scores[max_score_index].detach().cpu().item()
    max_score_box = boxes[max_score_index].detach().cpu().numpy()
    
    return max_score, list(max_score_box)
    
def visualize(fast_rcnn: nn.Module = None, image_name:Optional[str]=None, cls_names = config.labels):
    font = ImageFont.truetype(config.font_path, size=16)
    # type: (...) -> None
    dataset = AnnotationDataset(mode="test")

    img, gt_boxes, img_basename = next(iter(DataLoader(dataset, shuffle=True, batch_size=1)))

    with torch.no_grad():
        if fast_rcnn is None:
            fast_rcnn = FasterRCNN().to(device)

        fast_rcnn.eval()

        img_ori = denormalize(deepcopy(img).squeeze(0))

        scores, boxes, cls_idxes, rois = fast_rcnn(img)

        draw = ImageDraw.Draw(img_ori)

        # for roi in rois:
        #     draw.rectangle(((roi[0], roi[1]), (roi[2], roi[3])), outline=(255, 255, 255, 10), width=1)
        
        for box_info in gt_boxes:
            x1, y1, x2, y2, cls_idx = box_info
            cls_idx = cls_idx - 1
            draw.rectangle(((x1, y1), (x2, y2)), outline=(0, 255, 0, 255), width=1)
            draw.text(
                (x1, max(y1 - 20, 4)),
                "{}".format(cls_names[int(cls_idx.item())]), (0, 255, 0),
                font=font
            )
        draw.text(
            (10, config.input_height - 40),
            img_basename[0],
            font=font
        )

        for score, box, cls_idx in zip(scores, boxes, cls_idxes):
            xmin, ymin, xmax, ymax=box
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline='blue', width=1)
            draw.text(
                (xmin, max(ymin - 20, 4)),
                "{}: {:.2f}".format(cls_names[cls_idx.item()], score.item()), "blue",
                font=font
            )

        if image_name is None:
            imgname = "test.jpg"
        else:
            imgname = image_name 
            
        # img_ori = img_ori.crop(padding_window)
        # img_ori.resize(original_wh)
        img_ori.save("performance_check/{}".format(imgname))
        img_ori.save("performance_check/latest.jpg".format(imgname))
    
        fast_rcnn.train()
    
    
def denormalize(tensor_img):
    mean = torch.as_tensor([0.485, 0.456, 0.406])
    std = torch.as_tensor([0.229, 0.224, 0.225])
    tensor = tensor_img.permute(1,2,0) 
    mean = mean.expand_as(tensor)
    std = std.expand_as(tensor)
    tensor = tensor * std + mean
    tensor = tensor * 255
    tensor = tensor.detach().cpu().numpy().astype("uint8")
    img = Image.fromarray(tensor)
    return img
    
    

if __name__=="__main__":
    visualize()
        
        
        