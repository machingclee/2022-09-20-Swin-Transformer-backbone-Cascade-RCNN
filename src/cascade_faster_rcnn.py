from random import random
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
from einops import rearrange, reduce, repeat
from .context_block import context_block2d
from .feature_extractor import Resnet50FPNFeactureExtractor, SwinFeatureExtractor
from .box_utils import (
    assign_targets_to_anchors_or_proposals, clip_boxes_to_image,
    decode_deltas_to_boxes, encode_boxes_to_deltas, remove_small_boxes,
    decode_single
)
from .rpn import RPN
from . import config
from typing import OrderedDict, cast
from torch import Tensor
from torchvision.ops import nms
from .utils import random_choice, smooth_l1_loss
from .device import device
from torchvision.ops import MultiScaleRoIAlign
from PIL import Image, ImageDraw
from .mlp_detector import MLPDetector

cce_loss = nn.CrossEntropyLoss()


class CascadeFasterRCNN(nn.Module):
    def __init__(self, fpn_feature_extractor: nn.Module, fpn_feat_channels):
        super(CascadeFasterRCNN, self).__init__()
        self.feature_extractor = fpn_feature_extractor
        self.rpn = RPN(in_channel=fpn_feat_channels).to(device)
        self.mlp_detectors = nn.ModuleList([
            MLPDetector(in_channels=fpn_feat_channels).to(device)
            for _ in range(len(config.cascade_proposal_ious))
        ])

        self.roi_align = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=[7, 7],
            sampling_ratio=2
        )

    def filter_small_rois(self, logits, rois):
        rois = rois.squeeze(1).unsqueeze(0)
        hs = rois[..., 2] - rois[..., 0]
        ws = rois[..., 3] - rois[..., 1]
        keep_mask = (hs >= config.min_size) * (ws >= config.min_size)
        logits = logits[keep_mask]
        rois = rois[keep_mask]
        return logits, rois

    def batch_filter_by_nms(self, logits, levels, rois, n_pre_nms, n_post_nms, thresh, n_levels=4):
        level_dist = [len(torch.where(levels == l)[0]) for l in range(n_levels)]

        scores = logits.softmax(dim=1)[:, 1]
        scores_by_level = torch.split(scores, level_dist)
        rois_by_level = torch.split(rois, level_dist)

        sorted_scores_by_level = []
        sorted_rois_by_levels = []
        sorted_levels = []

        for level, (scores_curr_level, rois_curr_level) in enumerate(zip(scores_by_level, rois_by_level)):
            order = scores_curr_level.ravel().argsort(descending=True)
            order = order[:n_pre_nms]
            top_n_score_by_level = scores_curr_level[order]
            top_n_rois_by_levle = rois_curr_level[order]
            sorted_scores_by_level.append(top_n_score_by_level)
            sorted_rois_by_levels.append(top_n_rois_by_levle)
            sorted_levels.append(torch.as_tensor([level] * len(order)).to(device))

        scores = torch.cat(sorted_scores_by_level, dim=0)
        rois = torch.cat(sorted_rois_by_levels, dim=0)
        levels = torch.cat(sorted_levels, dim=0)

        separation_val = torch.max(rois) + 1
        separation_shifts = levels * separation_val
        shifted_rois = rois + separation_shifts[:, None]

        keep = nms(shifted_rois, scores, thresh)
        keep = keep[:n_post_nms]
        logits = logits[keep]
        rois = rois[keep]
        levels = levels[keep]

        keep = remove_small_boxes(rois, min_length=config.prediction_min_width_height)
        logits = logits[keep]
        rois = rois[keep]
        levels = levels[keep]

        return logits, rois, levels

    def get_rpn_loss(self, target_boxes, flattened_pred_deltas, flattened_pred_fg_bg_logit):
        flattend_labels, flattended_distributed_targets, _ = assign_targets_to_anchors_or_proposals(
            target_boxes,
            self.rpn.multi_scale_anchors,
            n_sample=config.rpn_n_sample,
            pos_sample_ratio=config.rpn_pos_ratio,
            pos_iou_thresh=config.target_pos_iou_thres,
            neg_iou_thresh=config.target_neg_iou_thres,
            target_cls_indexes=None
        )
        flattend_labels = flattend_labels.to(device)
        pos_mask = flattend_labels == 1
        keep_mask = torch.abs(flattend_labels) == 1

        target_deltas = encode_boxes_to_deltas(flattended_distributed_targets, self.rpn.flattened_multi_scale_anchors)
        objectness_label = torch.zeros_like(flattend_labels, device=device, dtype=torch.long)
        objectness_label[flattend_labels == 1] = 1.0

        if torch.sum(pos_mask) > 0:
            rpn_reg_loss = smooth_l1_loss(flattened_pred_deltas[pos_mask], target_deltas[pos_mask])
        else:
            rpn_reg_loss = torch.sum(flattened_pred_deltas) * 0

        rpn_cls_loss = cce_loss(flattened_pred_fg_bg_logit.squeeze(0)[keep_mask], objectness_label[keep_mask])

        return rpn_cls_loss, rpn_reg_loss

    def get_roi_loss(self, labels, distributed_targets_to_roi, rois, pred_deltas, cls_logits, distributed_cls_index):
        target_deltas = encode_boxes_to_deltas(
            distributed_targets_to_roi, rois, weights=config.roi_head_encode_weights
        )
        N = cls_logits.shape[0]
        pred_deltas = pred_deltas.reshape(N, -1, 4)

        target_deltas = target_deltas[labels == 1]
        keep_mask = torch.abs(labels) == 1
        sub_labels = labels[keep_mask]

        distributed_cls_index = distributed_cls_index[keep_mask]
        pos_idx = torch.where(sub_labels == 1)[0]
        neg_idx = torch.where(sub_labels == -1)[0]
        classes = distributed_cls_index[pos_idx]

        n_pos = len(pos_idx)
        n_neg = len(neg_idx)

        if n_pos > 0:
            roi_reg_loss = smooth_l1_loss(
                target_deltas,
                pred_deltas[pos_idx, classes.long()]
            )
            roi_cls_loss = n_pos * cce_loss(cls_logits[pos_idx], distributed_cls_index[pos_idx].long())
            roi_cls_loss += n_neg * cce_loss(cls_logits[neg_idx], distributed_cls_index[neg_idx].long())
        else:
            roi_cls_loss = n_neg * cce_loss(cls_logits[neg_idx], distributed_cls_index[neg_idx].long())
            roi_reg_loss = torch.sum(pred_deltas) * 0

        roi_cls_loss = roi_cls_loss / (n_pos + n_neg)

        if roi_reg_loss == torch.inf:
            print("inf roi_reg_loss")

        return roi_cls_loss, roi_reg_loss

    def filter_boxes_by_scores_and_size(self, cls_logits, pred_boxes):
        cls_idxes = torch.arange(config.n_classes, device=device)
        cls_idxes = cls_idxes[None, ...].expand_as(cls_logits)

        scores = cls_logits.softmax(dim=1)[:, 1:]
        boxes = pred_boxes
        cls_idxes = cls_idxes[:, 1:]

        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        cls_idxes = cls_idxes.reshape(-1)

        idxes = torch.where(torch.gt(scores, config.pred_score_thresh))[0]
        boxes = boxes[idxes]
        scores = scores[idxes]
        cls_idxes = cls_idxes[idxes]

        boxes = clip_boxes_to_image(boxes)
        keep = remove_small_boxes(boxes, min_length=config.prediction_min_width_height)
        boxes = boxes[keep]
        scores = scores[keep]
        cls_idxes = cls_idxes[keep]

        return scores, boxes, cls_idxes

    def cascade_stage_prediction_in_training(
        self,
        out_feat,
        target_boxes,
        target_cls_indexes,
        rois,
        flattened_pred_fg_bg_logits,
        flattened_levels,
        n_levels,
        proposal_iou,
        stage: int,
        last_stage: bool
    ):
        pred_fg_bg_logits, rois, levels = self.batch_filter_by_nms(
            flattened_pred_fg_bg_logits,
            flattened_levels,
            rois,
            config.n_train_pre_nms,
            config.n_train_post_nms,
            proposal_iou,
            n_levels=n_levels
        )
        level_distribution = [len(torch.where(levels == l)[0]) for l in range(n_levels)]
        flattened_labels, flattened_distributed_targets_to_roi, flattened_distributed_cls_indexes = \
            assign_targets_to_anchors_or_proposals(
                target_boxes,
                torch.split(rois, level_distribution),
                n_sample=config.roi_n_sample,
                pos_sample_ratio=config.roi_pos_ratio,
                pos_iou_thresh=config.roi_pos_iou_thresh,
                neg_iou_thresh=config.roi_neg_iou_thresh,
                target_cls_indexes=target_cls_indexes
            )
        keep_mask = torch.abs(flattened_labels) == 1
        levels = levels[keep_mask]
        p2, p3, p4, p5 = out_feat
        ordered_feat = OrderedDict(
            [("0", p2), ("1", p3), ("2", p4), ("3", p5)]
        )

        pooling = self.roi_align(
            ordered_feat,
            [rois[keep_mask]],
            image_shapes=[config.image_shape]
        )
        cls_logits, roi_pred_deltas = self.mlp_detectors[stage](pooling)

        if not last_stage:
            cls_logits = cls_logits[:, 1:]
            roi_pred_deltas = roi_pred_deltas[:, 4:]

        roi_cls_loss, roi_reg_loss = None, None

        if last_stage:
            roi_cls_loss, roi_reg_loss = self.get_roi_loss(
                flattened_labels,
                flattened_distributed_targets_to_roi,
                rois,
                roi_pred_deltas,
                cls_logits,
                flattened_distributed_cls_indexes
            )
        rois = rois[keep_mask]
        pred_fg_bg_logits = pred_fg_bg_logits[keep_mask]
        rois = decode_deltas_to_boxes(roi_pred_deltas, rois)
        rois = rois.squeeze(0)

        return roi_cls_loss, roi_reg_loss, cls_logits, rois, pred_fg_bg_logits, levels

    def cascade_stage_prediction_in_eval(
        self,
        out_feat,
        rois,
        flattened_pred_fg_bg_logits,
        flattened_levels,
        proposal_iou,
        stage: int,
        n_levels
    ):
        pred_fg_bg_logits, rois, levels = self.batch_filter_by_nms(
            flattened_pred_fg_bg_logits,
            flattened_levels,
            rois,
            config.n_eval_pre_nms,
            config.n_eval_post_nms,
            proposal_iou,
            n_levels
        )

        pred_fg_bg_logits = pred_fg_bg_logits[:config.roi_n_sample]
        rois = rois[:config.roi_n_sample]
        levels = levels[:config.roi_n_sample]

        p2, p3, p4, p5 = out_feat
        ordered_feat = OrderedDict(
            [("0", p2), ("1", p3), ("2", p4), ("3", p5)]
        )

        pooling = self.roi_align(
            ordered_feat,
            [rois],
            image_shapes=[config.image_shape]
        )

        cls_logits, roi_pred_deltas = self.mlp_detectors[stage](pooling)
        rois = decode_deltas_to_boxes(roi_pred_deltas, rois)
        rois = clip_boxes_to_image(rois)
        rois = rois.squeeze(0)

        return rois, pred_fg_bg_logits, levels, cls_logits

    def forward(
        self,
        x,
        target_boxes=None,
        target_cls_indexes=None
    ):
        x = x.to(device)

        if target_boxes is not None:
            target_boxes = target_boxes.to(device)

        if target_cls_indexes is not None:
            target_cls_indexes = target_cls_indexes.to(device)

        if self.training:
            assert target_boxes is not None
            assert target_cls_indexes is not None

        out_feat = self.feature_extractor(x)

        features = out_feat
        pred_fg_bg_logits = []
        pred_deltas = []
        levels = []
        n_levels = len(features)

        for level, feature in enumerate(features):
            logits, deltas = self.rpn(feature)
            logits = logits.squeeze(0)
            deltas = deltas.squeeze(0)
            pred_fg_bg_logits.append(logits)
            pred_deltas.append(deltas)
            levels.append(torch.as_tensor([level] * len(logits)))

        flattened_pred_fg_bg_logits = torch.cat(pred_fg_bg_logits, dim=0)
        flattened_pred_deltas = torch.cat(pred_deltas, dim=0)
        flattened_levels = torch.cat(levels, dim=0).to(device)

        rois = decode_deltas_to_boxes(
            flattened_pred_deltas.detach().clone(),
            self.rpn.flattened_multi_scale_anchors
        )
        rois = clip_boxes_to_image(rois)
        rois = rois.squeeze(0)

        if self.training:
            rpn_cls_loss, rpn_reg_loss = self.get_rpn_loss(
                target_boxes, flattened_pred_deltas, flattened_pred_fg_bg_logits
            )

            pred_fg_bg_logits = flattened_pred_fg_bg_logits
            levels = flattened_levels

            for i in range(len(config.cascade_proposal_ious)):
                roi_cls_loss, roi_reg_loss, cls_logits, rois, pred_fg_bg_logits, levels = \
                    self.cascade_stage_prediction_in_training(
                        out_feat,
                        target_boxes,
                        target_cls_indexes,
                        rois,
                        pred_fg_bg_logits,
                        levels,
                        n_levels,
                        config.cascade_proposal_ious[i],
                        stage=i,
                        last_stage=False if i < len(config.cascade_proposal_ious) - 1 else True
                    )

            return rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss
        else:
            fg_bg_logits = flattened_pred_fg_bg_logits
            levels = flattened_levels

            for i in range(len(config.cascade_proposal_ious)):
                rois, fg_bg_logits, levels, cls_logits = self.cascade_stage_prediction_in_eval(
                    out_feat,
                    rois,
                    fg_bg_logits,
                    levels,
                    config.cascade_proposal_ious[i],
                    stage=i,
                    n_levels=len(features)
                )

            pred_boxes = rois
            scores, boxes, cls_idxes = self.filter_boxes_by_scores_and_size(cls_logits, pred_boxes)
            cls_idxes = cls_idxes - 1

            keep = nms(boxes, scores, 0.2)
            scores = scores[keep]
            boxes = boxes[keep]
            cls_idxes = cls_idxes[keep]

            return scores, boxes, cls_idxes, rois
