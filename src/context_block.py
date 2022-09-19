import torch
import torch.nn as nn


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        nn.init.constant_(m[-1].weight, val=0)
        m[-1].inited = True
    else:
        nn.init.constant_(m.weight, val=0)
        m.inited = True


class context_block2d(nn.Module):
    def __init__(self, dim, ratio=4):
        super(context_block2d, self).__init__()
        self.dim = dim

        self.conv_mask = nn.Conv2d(dim, 1, kernel_size=1)  # context Modeling
        self.softmax = nn.Softmax(dim=2)
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(self.dim, self.dim // ratio, kernel_size=1),
            nn.LayerNorm([self.dim // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // ratio, self.dim, kernel_size=1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv_mask.weight, mode='fan_in')
        self.conv_mask.inited = True
        last_zero_init(self.channel_add_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W] 添加一个维度
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)  # softmax操作
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)
        return context

    def forward(self, x):
        out = x
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        out = out + channel_add_term
        return out
