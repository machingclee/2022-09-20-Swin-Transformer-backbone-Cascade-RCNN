import torch
import numpy as np
import math
import torchvision
from .anchor_generator import AnchorGenerator
from . import config
from torch import Tensor
from .device import device
from typing import List, Tuple, cast
from .utils import random_choice


def assign_targets_to_anchors_or_proposals(
    target_boxes,
    multi_scale_anchors,
    n_sample,
    pos_sample_ratio,
    pos_iou_thresh,
    neg_iou_thresh,
    target_cls_indexes=None,
):
    target_boxes = target_boxes.to(device)
    """
    return:
        labels: [len(anchors), ], -1 = neg, 0 = ignore, 1 = pos
        target: [len(anchors), 4]
    """
    multi_scale_labels = []
    multi_scale_distributed_cls_indexes = []
    multi_scale_distributed_targets = []

    for anchors in multi_scale_anchors:
        if len(anchors) == 0:
            continue
        labels = torch.zeros((len(anchors),), dtype=torch.int32).to(device)
        distributed_cls_indexes = torch.zeros((len(anchors),), dtype=torch.float32).to(device)
        distributed_targets = torch.zeros((len(anchors), 4), dtype=torch.float32).to(device)

        if len(target_boxes) == 0:
            chosen_idxes = random_choice(torch.arange(len(anchors)), n_sample)
            labels[chosen_idxes] = -1
            multi_scale_labels.append(labels)
            multi_scale_distributed_cls_indexes.append(distributed_cls_indexes)
            multi_scale_distributed_targets.append(distributed_targets)
            continue

        ious = box_iou(anchors, target_boxes)
        max_iou_anchor_index = torch.argmax(ious, dim=0)  # return k anchor indexes corresponding to k targets
        labels[max_iou_anchor_index] = 1
        distributed_targets[max_iou_anchor_index] = target_boxes

        if target_cls_indexes is not None:
            distributed_cls_indexes[max_iou_anchor_index] = target_cls_indexes

        max_iou_target_idx_per_anchor = torch.argmax(ious, dim=1)
        max_ious = ious[torch.arange(len(anchors)), max_iou_target_idx_per_anchor]  # == torch.max(ious, dim=1)
        pos_index = torch.where(max_ious > pos_iou_thresh)[0]
        neg_mask = max_ious < neg_iou_thresh

        distributed_targets[pos_index] = target_boxes[max_iou_target_idx_per_anchor[pos_index]]

        if target_cls_indexes is not None:
            distributed_cls_indexes[pos_index] = target_cls_indexes[max_iou_target_idx_per_anchor[pos_index]]

        labels[pos_index] = 1
        pos_mask = labels == 1
        non_pos_mask = torch.logical_not(pos_mask).to(device)
        labels[non_pos_mask * neg_mask] = -1

        pos_ratio = pos_sample_ratio
        n_desired_pos_sample = n_sample * pos_ratio

        actual_pos_anchor_index = torch.where(labels == 1)[0]
        n_actual_pos_anchor = len(actual_pos_anchor_index)

        if n_actual_pos_anchor > n_desired_pos_sample:
            surplus = n_actual_pos_anchor - n_desired_pos_sample
            discarded_index = random_choice(actual_pos_anchor_index, surplus)
            labels[discarded_index] = 0

        n_desired_neg_sample = n_sample - torch.sum(labels == 1)

        actual_neg_anchor_index = torch.where(labels == -1)[0]
        n_actual_neg_anchor = len(actual_neg_anchor_index)

        if n_actual_neg_anchor > n_desired_neg_sample:
            surplus = n_actual_neg_anchor - n_desired_neg_sample
            discarded_index = random_choice(actual_neg_anchor_index, surplus)
            labels[discarded_index] = 0

        multi_scale_labels.append(labels)
        multi_scale_distributed_targets.append(distributed_targets)
        multi_scale_distributed_cls_indexes.append(distributed_cls_indexes)

    flattened_multi_scale_labels = torch.cat(multi_scale_labels, dim=0)
    flattened_multi_scale_distributed_targets = torch.cat(multi_scale_distributed_targets, dim=0)
    flattened_multi_scale_distributed_cls_indexes = torch.cat(multi_scale_distributed_cls_indexes, dim=0)

    return (
        flattened_multi_scale_labels,
        flattened_multi_scale_distributed_targets,
        flattened_multi_scale_distributed_cls_indexes
    )


def encode_boxes_to_deltas(distributed_targets, anchor_or_proposals, weights=[1, 1, 1, 1]):
    epsilon = 1e-4
    anchor_or_proposals = anchor_or_proposals.to(device)
    anchors_x1 = anchor_or_proposals[:, 0].unsqueeze(1)
    anchors_y1 = anchor_or_proposals[:, 1].unsqueeze(1)
    anchors_x2 = anchor_or_proposals[:, 2].unsqueeze(1)
    anchors_y2 = anchor_or_proposals[:, 3].unsqueeze(1)

    target_boxes_x1 = distributed_targets[:, 0].unsqueeze(1)
    target_boxes_y1 = distributed_targets[:, 1].unsqueeze(1)
    target_boxes_x2 = distributed_targets[:, 2].unsqueeze(1)
    target_boxes_y2 = distributed_targets[:, 3].unsqueeze(1)

    an_widths = anchors_x2 - anchors_x1
    an_heights = anchors_y2 - anchors_y1
    an_ctr_x = anchors_x1 + 0.5 * an_widths
    an_ctr_y = anchors_y1 + 0.5 * an_heights

    gt_widths = target_boxes_x2 - target_boxes_x1
    gt_heights = target_boxes_y2 - target_boxes_y1
    gt_ctr_x = target_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = target_boxes_y1 + 0.5 * gt_heights

    wx, wy, ww, wh = weights
    targets_dx = wx * (gt_ctr_x - an_ctr_x) / (an_widths + epsilon)
    targets_dy = wy * (gt_ctr_y - an_ctr_y) / (an_heights + epsilon)
    targets_dw = ww * torch.log((gt_widths + epsilon) / (an_widths + epsilon))
    targets_dh = wh * torch.log((gt_heights + epsilon) / (an_heights + epsilon))

    deltas = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return deltas


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    boxes1 = boxes1.to(device)
    boxes2 = boxes2.to(device)
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    if len(boxes1.shape) == 1:
        boxes1 = boxes1[None, ...]
    if len(boxes2.shape) == 1:
        boxes2 = boxes2[None, ...]

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    #  When the shapes do not match,
    #  the shape of the returned output tensor follows the broadcasting rules
    upper_left = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    lower_right = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    wh = (lower_right - upper_left).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    epsilon = 1e-8
    iou = inter / (area1[:, None] + area2 - inter + epsilon)
    return iou


def decode_deltas_to_boxes(deltas, anchors, weights=[1, 1, 1, 1]):
    deltas = deltas.to(device)
    anchors = anchors.to(device)
    if not isinstance(anchors, (list, tuple)):
        anchors = [anchors]
    assert isinstance(anchors, (list, tuple))
    assert isinstance(deltas, torch.Tensor)
    n_boxes_per_image = [b.size(0) for b in anchors]
    concat_boxes = torch.cat(anchors, dim=0).squeeze(0)

    box_sum = 0
    for val in n_boxes_per_image:
        box_sum += val
    # single mean single feature scale,
    # there are many scales in fpn and each scales contain many boxes
    pred_boxes = decode_single(
        deltas, concat_boxes, weights=weights
    )

    if box_sum > 0:
        if len(deltas.shape) == 3:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)

    return pred_boxes


def decode_single(deltas, anchors, weights):
    if len(deltas.shape) == 3:
        # this happens when deltas is pred_deltas of roi,
        # we predict deltas for each class, like [128, 4, 4]
        N = anchors.shape[0]
        anchors = anchors.reshape(N, -1, 4)

    anchors = anchors.to(deltas.dtype)
    bbox_xform_clip = math.log(1000. / 16)
    # xmin, ymin, xmax, ymax
    widths = anchors[..., 2] - anchors[..., 0]
    heights = anchors[..., 3] - anchors[..., 1]

    ctr_x = anchors[..., 0] + 0.5 * widths
    ctr_y = anchors[..., 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[..., 0] / wx
    dy = deltas[..., 1] / wy
    dw = deltas[..., 2] / ww
    dh = deltas[..., 3] / wh

    # limit max value, prevent sending too large values into torch.exp()
    # bbox_xform_clip=math.log(1000. / 16)   4.135
    dw = torch.clamp(dw, max=bbox_xform_clip)
    dh = torch.clamp(dh, max=bbox_xform_clip)

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    xmins = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
    ymins = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
    xmaxs = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
    ymaxs = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h

    if len(deltas.shape) == 3:
        pred_boxes = torch.cat(
            [
                xmins.unsqueeze(2),
                ymins.unsqueeze(2),
                xmaxs.unsqueeze(2),
                ymaxs.unsqueeze(2)
            ],
            dim=2)
    else:
        pred_boxes = torch.cat(
            [
                xmins.unsqueeze(1),
                ymins.unsqueeze(1),
                xmaxs.unsqueeze(1),
                ymaxs.unsqueeze(1)
            ],
            dim=1)

    return pred_boxes


def clip_boxes_to_image(boxes, size=config.image_shape):
    dim = boxes.dim()
    boxes_x = boxes[..., 0:: 2]  # x1, x2
    boxes_y = boxes[..., 1:: 2]  # y1, y2
    height, width = size

    boxes_x = boxes_x.clamp(min=0, max=width)
    boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def remove_small_boxes(boxes, min_length):
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = torch.logical_and(torch.ge(ws, min_length), torch.ge(hs, min_length))
    keep = torch.where(keep)[0]
    return keep
