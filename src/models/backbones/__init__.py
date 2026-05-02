"""
src/models/backbones/__init__.py
--------------------------------
职责：导出 backbone 相关公共接口。
"""

from src.models.backbones.image_backbone import ImageBackbone
from src.models.backbones.pc_backbone import PCBackbone
from src.models.backbones.pointtransformer_v3_encoder import PointTransformerV3Encoder
from src.models.backbones.voxel_token_utils import (
    build_padded_voxel_tokens,
    flatten_voxel_grid_for_transformer,
)
from src.models.backbones.voxelnet_encoder import VoxelNetEncoder, voxelize_points

__all__ = [
    "ImageBackbone",
    "PCBackbone",
    "PointTransformerV3Encoder",
    "VoxelNetEncoder",
    "build_padded_voxel_tokens",
    "flatten_voxel_grid_for_transformer",
    "voxelize_points",
]
