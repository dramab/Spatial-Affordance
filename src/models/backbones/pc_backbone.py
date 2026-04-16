"""
src/models/backbones/pc_backbone.py
-----------------------------------
职责：点云 backbone 统一分发入口。

当前支持：
- voxelnet：输出稠密 voxel embedding 及辅助稀疏信息

用法：
    from src.models.backbones.pc_backbone import PCBackbone

    backbone = PCBackbone(
        {
            "type": "voxelnet",
            "voxel_size_cm": [2.0, 2.0, 2.0],
            "point_cloud_range_cm": [-80.0, -80.0, -10.0, 80.0, 80.0, 120.0],
        }
    )
    outputs = backbone(points_xyz, point_feats)
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional

import torch
from torch import nn

from src.models.backbones.voxelnet_encoder import VoxelNetEncoder


def _cfg_get(cfg: Mapping[str, Any] | object, key: str, default: Any = None) -> Any:
    """
    从 dict 或对象中统一读取配置。

    输入:
        cfg: 配置对象
        key: str 配置键
        default: 默认值
    输出:
        任意配置值
    """
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class PCBackbone(nn.Module):
    """
    点云 backbone 包装器。

    输入:
        points_xyz: Tensor(B, N, 3) 点云 xyz
        point_feats: Tensor(B, N, F) 可选点特征
    输出:
        dict，字段由具体 backbone 决定
    """

    def __init__(self, cfg: Mapping[str, Any] | object):
        super().__init__()
        self.cfg = cfg
        backbone_type = str(_cfg_get(cfg, "type", "voxelnet")).lower()
        if backbone_type != "voxelnet":
            raise ValueError(f"unsupported pc backbone type: {backbone_type}")

        voxel_size_cm = _cfg_get(cfg, "voxel_size_cm", (2.0, 2.0, 2.0))
        point_cloud_range_cm = _cfg_get(
            cfg,
            "point_cloud_range_cm",
            # (x_min, y_min, z_min, x_max, y_max, z_max)，单位 cm
            (-80.0, -80.0, -10.0, 80.0, 80.0, 120.0),
        )
        max_points_per_voxel = int(_cfg_get(cfg, "max_points_per_voxel", 32))
        max_voxels = int(_cfg_get(cfg, "max_voxels", 20000))
        input_feature_dim = int(_cfg_get(cfg, "input_feature_dim", 6))
        svfe_hidden_channels = int(_cfg_get(cfg, "svfe_hidden_channels", 32))
        svfe_out_channels = int(_cfg_get(cfg, "svfe_out_channels", 128))
        cml_channels = tuple(_cfg_get(cfg, "cml_channels", (128, 256, 256)))
        return_dense = bool(_cfg_get(cfg, "return_dense", True))

        self.backbone = VoxelNetEncoder(
            voxel_size_cm=voxel_size_cm,
            point_cloud_range_cm=point_cloud_range_cm,
            max_points_per_voxel=max_points_per_voxel,
            max_voxels=max_voxels,
            input_feature_dim=input_feature_dim,
            svfe_hidden_channels=svfe_hidden_channels,
            svfe_out_channels=svfe_out_channels,
            cml_channels=cml_channels,
            return_dense=return_dense,
        )
        self.out_channels = self.backbone.out_channels

    def forward(
            self,
            points_xyz: torch.Tensor,
            point_feats: Optional[torch.Tensor] = None) -> MutableMapping[str, Any]:
        """
        执行点云编码。

        输入:
            points_xyz: Tensor(B, N, 3) 点云坐标
            point_feats: Tensor(B, N, F) 额外点特征
        输出:
            dict 编码结果
        """
        return self.backbone(points_xyz, point_feats)
