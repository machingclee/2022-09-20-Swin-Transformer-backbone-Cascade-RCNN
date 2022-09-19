from cProfile import label
from matplotlib import image
import numpy as np
import torch
import albumentations as A
import json
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from . import config
from torchvision import transforms
from copy import deepcopy
from typing import List, TypedDict
from glob import glob
from random import shuffle

to_tensor = transforms.ToTensor()

torch_img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def resize_img(img):
    """
    img:  Pillow image
    """
    h, w = img.height, img.width
    if h >= w:
        ratio = config.input_height / h
        new_h, new_w = int(h * ratio), int(w * ratio)
    else:
        ratio = config.input_width / w
        new_h, new_w = int(h * ratio), int(w * ratio)

    img = img.resize((new_w, new_h), Image.BILINEAR)
    return img, (w, h)


def pad_img(img):
    h = img.height
    w = img.width
    img = np.array(img)
    img = np.pad(img, pad_width=((0, config.input_height - h), (0, config.input_width - w), (0, 0)), mode="constant")
    img = Image.fromarray(img)
    assert img.height == config.input_height
    assert img.width == config.input_width
    return img


def resize_and_padding(img, return_window=False):
    img, (ori_w, ori_h) = resize_img(img)
    w = img.width
    h = img.height
    padding_window = (w, h)
    img = pad_img(img)

    if not return_window:
        return img
    else:
        return img, padding_window, (ori_w, ori_h)


albumentation_transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0, rotate_limit=10, p=0.7),
    # A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.1),
    A.HorizontalFlip(p=0.5),
    A.GaussNoise(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.5),
    A.OneOf([
        A.Blur(blur_limit=3, p=0.5),
        # A.ColorJitter(p=0.1)
    ], p=0.8),

    A.LongestMaxSize(max_size=config.input_height, interpolation=1, p=1),
    A.PadIfNeeded(
        min_height=config.input_height,
        min_width=config.input_height,
        border_mode=0,
        value=(0, 0, 0),
        position="top_left"
    ),
],
    p=1,
    bbox_params=A.BboxParams(format="pascal_voc", min_area=0.1)
)

albumentation_minimal_transform = A.Compose([
    A.LongestMaxSize(max_size=config.input_height, interpolation=1, p=1),
    A.PadIfNeeded(
        min_height=config.input_height,
        min_width=config.input_height,
        border_mode=0,
        value=(0, 0, 0),
        position="top_left"
    ),
],
    p=1, bbox_params=A.BboxParams(format="pascal_voc", min_area=0.1)
)


def data_minimal_augmentation(img, boxes=None):
    # in order to fit into the network
    if isinstance(img, Image.Image):
        img = np.array(img)
    if boxes is None:
        boxes = [[1, 2, 3, 4, 0]]
        transformed = albumentation_minimal_transform(image=img, bboxes=boxes)
        return transformed["image"]
    else:
        transformed = albumentation_minimal_transform(image=img, bboxes=boxes)
        return transformed["image"], transformed["bboxes"]


Segmentation = List[float]
Box = List[float]


class Target(TypedDict):
    image_path: str
    boxes: List[List[float]]
    segmentations: List[List[float]]


class AnnotationDataset(Dataset):
    def __init__(self, mode="train"):
        assert mode in ["train", "test"]
        self.mode = mode
        super(AnnotationDataset, self).__init__()
        self.annotations = {}

        annotation_files = ["datasets/rust_signboards_round_3.json"]
        negative_sample_dir = "datasets/normal"
        self.data: List[Target] = []

        imageId_imagePath = {}
        self.cls_names = []
        self.cat_coco_ids = [0] # 0 for background

        # add negative samples
        for img in glob(f"{negative_sample_dir}/*.jpg"):
            data: Target = {
                "boxes": [],
                "image_path": img,
                "segmentations": []
            }
            self.data.append(data)

        # add positive samples
        for annotation_file in annotation_files:
            with open(annotation_file, "r") as f:
                annotations_ = json.load(f)
            images = annotations_["images"]

            categories = annotations_["categories"]
            annotations = annotations_["annotations"]

            for img in images:
                imageId_imagePath.update({img["id"]: img["path"]})
            for cat in categories:
                self.cls_names.append(cat["name"])
                self.cat_coco_ids.append(cat["id"])

            for anno in annotations:
                image_id = anno["image_id"]
                category_id = anno["category_id"]
                image_path = imageId_imagePath[image_id]

                bbox = anno["bbox"]
                x, y, w, h = bbox
                xmin = float(x)
                ymin = float(y)
                xmax = xmin + float(w)
                ymax = ymin + float(h)
                cls_idx = self.cat_coco_ids.index(category_id)
                bbox = [xmin, ymin, xmax, ymax, cls_idx]
                segmentation = anno["segmentation"]

                target = next((target for target in self.data if target.get("image_path") == image_path), None)
                if target is not None:
                    target["boxes"].append(bbox)
                    target["segmentations"].append(segmentation)
                else:
                    target: Target = {
                        "image_path": image_path,
                        "boxes": [bbox],
                        "segmentations": [segmentation]
                    }
                    self.data.append(target)

    def __getitem__(self, index):
        data = self.data[index]

        if self.mode == "test":
            # performance check on rusty-signboard only
            shuffle(self.data)
            data = next((data for data in self.data if len(data.get("boxes")) > 0))

        img_path = data["image_path"]
        boxes = data["boxes"]
        boxes = np.array(boxes)

        img = Image.open(img_path.replace("/datasets", "datasets"))
        img = np.array(img)

        if self.mode == "train":
            if len(boxes) > 0:
                transformed = albumentation_transform(image=img, bboxes=boxes)
                img = transformed["image"]
                boxes = transformed["bboxes"]
                img = torch_img_transform(img)

                if len(boxes) == 0:
                    return img, torch.as_tensor([]), torch.as_tensor([])

                new_boxes = np.array(boxes)

                boxes_ = torch.as_tensor(new_boxes[..., 0:4]).float()
                cls_idxes = torch.as_tensor(new_boxes[..., 4]).float()
                
                return img, boxes_, cls_idxes
            else:
                # as there are already plenty of normal signboards, we just pad that signboards
                transformed = albumentation_transform(image=img, bboxes=[[1,1,2,2,0]])
                img = transformed["image"]                
                img = torch_img_transform(img)
                return img, torch.as_tensor([]), torch.as_tensor([])
        else:
            transformed = albumentation_minimal_transform(image=img, bboxes=boxes)
            img = transformed["image"]
            boxes = transformed["bboxes"]
            img = torch_img_transform(img)
            return img, boxes, img_path

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = AnnotationDataset()
    img, boxes_, cls_idxes, mask_pooling_pred_targets = dataset[0]
    print(img)
    print(boxes_)
    print(cls_idxes)
    print(mask_pooling_pred_targets)
