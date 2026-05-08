"""
src/models/heads/size_yaw_head.py
---------------------------------
职责：从目标物体 query 与放置位 query 的融合特征回归 box size 与 yaw。

用法：
    from src.models.heads.size_yaw_head import SizeYawHead

    head = SizeYawHead(input_dim=512, hidden_dim=256, num_layers=2)
    pred_size_yaw = head(fused_query_tokens)
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn


class SizeYawHead(nn.Module):
    """
    作用：将融合后的 query token 映射为 3D box 尺寸与 yaw。

    输入：
        fused_tokens: Tensor(B, H2)，由 object query 与 placement query 拼接得到
    输出：
        Tensor(B, 4)，格式为 (log_l, log_w, log_h, yaw)
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            num_layers: int = 2,
            out_dim: int = 4):
        """
        用法: head = SizeYawHead(input_dim=512, hidden_dim=256, num_layers=2)
        作用: 初始化 size/yaw 回归 head
        输入: input_dim: 融合 query 特征维度；hidden_dim: MLP 隐层维度；num_layers: MLP 层数；out_dim: 输出维度，必须为 4
        输出: SizeYawHead 实例
        """
        super().__init__()
        if int(out_dim) != 4:
            raise ValueError("SizeYawHead currently only supports out_dim=4")

        input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        num_layers = max(int(num_layers), 1)

        layers: List[nn.Module] = []
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            current_dim = hidden_dim
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.output = nn.Linear(current_dim, int(out_dim))

    def forward(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        """
        用法: pred_size_yaw = head(fused_tokens)
        作用: 回归 normalized box 的 log-size 与 yaw
        输入: fused_tokens: Tensor(B,H2)
        输出: Tensor(B,4)
        """
        hidden = self.mlp(fused_tokens)
        return self.output(hidden).to(torch.float32)
