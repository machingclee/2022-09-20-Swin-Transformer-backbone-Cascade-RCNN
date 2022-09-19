from operator import index
import numpy as np
import torch
from . import config
from .device import device


class AnchorGenerator():
    multi_scale_anchors = None
    flattened_multi_scale_anchors = None

    def __init__(self):
        self.grid_nums = config.img_shapes

    def get_multi_scale_anchors(self):
        if AnchorGenerator.multi_scale_anchors is not None:
            return AnchorGenerator.multi_scale_anchors

        ratios = torch.as_tensor(config.anchor_ratios)
        scales = torch.as_tensor(config.anchor_scales)

        hs = scales[None, ...] * torch.sqrt(ratios)[..., None]
        hs = hs.reshape((len(ratios) * len(scales),))

        ws = scales[None, ...] / torch.sqrt(ratios)[..., None]
        ws = ws.reshape((len(ratios) * len(scales),))

        hs = hs[..., None]
        ws = ws[..., None]

        base_anchors_coor_shift = torch.cat(
            [-ws / 2, -hs / 2, ws / 2, hs / 2],
            dim=-1
        )
        base_anchors_coor_shift = base_anchors_coor_shift.reshape((-1, 4))

        anchors_ = []
        for (grid_num_y, grid_num_x) in self.grid_nums:
            stride_y = config.input_height // grid_num_y
            stride_x = config.input_width // grid_num_x
            center_ys = torch.arange(1, (grid_num_y + 1)) * stride_y - stride_y / 2
            center_xs = torch.arange(1, (grid_num_x + 1)) * stride_x - stride_x / 2
            center_ys, center_xs = torch.meshgrid(center_ys, center_xs)
            center_ys = center_ys[..., None]
            center_xs = center_xs[..., None]
            centers = torch.cat(
                [center_xs, center_ys, center_xs, center_ys],
                dim=-1
            )
            centers = centers.reshape((-1, 4))

            anchors_curr_grid_scale = centers[:, None, :] + base_anchors_coor_shift[None, ...]
            anchors_curr_grid_scale = anchors_curr_grid_scale.reshape((-1, 4)).to(device)

            anchors_.append(anchors_curr_grid_scale)

        AnchorGenerator.multi_scale_anchors = anchors_

        return AnchorGenerator.multi_scale_anchors

    def get_flattened_multi_scale_anchors(self):
        assert AnchorGenerator.multi_scale_anchors is not None

        if AnchorGenerator.flattened_multi_scale_anchors is not None:
            return AnchorGenerator.flattened_multi_scale_anchors

        AnchorGenerator.flattened_multi_scale_anchors = torch.cat(AnchorGenerator.multi_scale_anchors, dim=0)
        return AnchorGenerator.flattened_multi_scale_anchors


if __name__ == "__main__":
    anchor_gen = AnchorGenerator()
    anchors = anchor_gen.get_multi_scale_anchors()
    print(anchors.shape)
