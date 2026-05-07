"""
src/models/heads/center3d_head.py
---------------------------------
职责：从 decoder token 回归 normalized 3D center 坐标。

用法：
    from src.models.heads.center3d_head import Center3DHead

    head = Center3DHead(hidden_dim=256, num_layers=2)
    pred_centers = head(decoder_tokens)
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn


class Center3DHead(nn.Module):
    """
    作用：将 decoder token 映射为 3D center 坐标。

    输入：
        decoder_tokens: Tensor(B, Q, H) decoder 输出 token
    输出：
        Tensor(B, Q, 3)，格式为 (cx, cy, cz)
    """

    def __init__(self, hidden_dim: int = 256, num_layers: int = 2, out_dim: int = 3):
        """
        用法: head = Center3DHead(hidden_dim=256, num_layers=2)
        作用: 初始化 3D center 回归 head
        输入: hidden_dim: decoder token 维度；num_layers: MLP 层数；out_dim: 输出维度，必须为 3
        输出: Center3DHead 实例
        """
        super().__init__()
        if int(out_dim) != 3:
            raise ValueError("Center3DHead currently only supports out_dim=3")

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
        用法: pred_centers = head(decoder_tokens)
        作用: 回归 normalized 3D center 坐标
        输入: decoder_tokens: Tensor(B, Q, H)
        输出: Tensor(B, Q, 3)
        """
        hidden = self.mlp(decoder_tokens)
        return self.output(hidden).to(torch.float32)
