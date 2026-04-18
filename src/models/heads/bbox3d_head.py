"""
src/models/heads/bbox3d_head.py
-------------------------------
职责：从 decoder token 回归 3D BBox 参数。

用法：
    from src.models.heads.bbox3d_head import BBox3DHead

    head = BBox3DHead(hidden_dim=256, num_layers=2, out_dim=7)
    pred_boxes = head(decoder_tokens)
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn
from torch.nn import functional as F


class BBox3DHead(nn.Module):
    """
    作用：将 decoder token 映射为 7D 3D BBox 参数。

    输入：
        decoder_tokens: Tensor(B, Q, H) decoder 输出 token
    输出：
        Tensor(B, Q, 7)，格式为 (cx, cy, cz, l, w, h, yaw)
    """

    def __init__(self, hidden_dim: int = 256, num_layers: int = 2, out_dim: int = 7):
        super().__init__()
        if int(out_dim) != 7:
            raise ValueError("BBox3DHead currently only supports out_dim=7")

        hidden_dim = int(hidden_dim)
        num_layers = max(int(num_layers), 1)
        layers: List[nn.Module] = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.output = nn.Linear(hidden_dim, int(out_dim))

    def forward(self, decoder_tokens: torch.Tensor) -> torch.Tensor:
        """
        作用：回归 3D BBox 参数，并约束尺寸项为正。

        输入：
            decoder_tokens: Tensor(B, Q, H)
        输出：
            Tensor(B, Q, 7)
        """
        hidden = self.mlp(decoder_tokens)
        raw_boxes = self.output(hidden).to(torch.float32)
        center = raw_boxes[..., 0:3]
        size = F.softplus(raw_boxes[..., 3:6]) + 1e-6
        yaw = raw_boxes[..., 6:7]
        pred_boxes = torch.cat([center, size, yaw], dim=-1)
        return pred_boxes
