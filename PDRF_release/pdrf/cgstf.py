"""CGSTF - Condition-Guided Spatial Transform Fusion (paper Fig. 3).

Estimates a dense displacement field from the conditional features and
applies differentiable warping to align backbone features with the
conditional context before fusion.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ConditionGuidedSpatialTransformFusion(nn.Module):
    def __init__(self, channels, reduction=16, max_offset=0.2):
        super().__init__()
        # 条件特征生成空间位移场
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 2, 3, padding=1),  # 输出Δx, Δy
            nn.Tanh()  # 保证位移在[-1, 1]之间
        )
        self.max_offset = max_offset  # 控制最大位移比例

        # 融合层（可自定义）
        self.fuse = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        # 1. 生成位移场
        offset = self.offset_conv(x2)  # [B, 2, H, W]
        offset = offset * self.max_offset  # 控制最大偏移

        # 2. 构造采样网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x1.device),
            torch.linspace(-1, 1, W, device=x1.device),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), 2)  # [H, W, 2]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]
        # offset: [B, 2, H, W] -> [B, H, W, 2]
        offset = offset.permute(0, 2, 3, 1)
        sampling_grid = grid + offset  # [B, H, W, 2]

        # 3. 对主特征做空间采样
        x1_warped = F.grid_sample(x1, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)

        # 4. 融合
        out = self.fuse(torch.cat([x1_warped, x2], dim=1))
        return out

CGSTF = ConditionGuidedSpatialTransformFusion
