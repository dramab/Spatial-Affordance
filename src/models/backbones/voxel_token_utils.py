"""
src/models/backbones/voxel_token_utils.py
-----------------------------------------
职责：将稠密体素特征转换为便于 Transformer 使用的 token 序列。

用法：
    from src.models.backbones.voxel_token_utils import flatten_voxel_grid_for_transformer

    token_dict = flatten_voxel_grid_for_transformer(
        dense_voxel_feats=outputs["dense_voxel_feats"],
        valid_mask=outputs["valid_mask"],
        grid_meta=outputs["grid_meta"],
    )
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def _compute_voxel_centers_cm(
        coords_bzyx: torch.Tensor,
        voxel_size_cm: Tuple[float, float, float],
        point_cloud_range_cm: Tuple[float, float, float, float, float, float]) -> torch.Tensor:
    """
    将体素索引转换为体素中心的世界坐标。

    输入:
        coords_bzyx: Tensor(K, 4) 体素索引，顺序为 (batch, z, y, x)
        voxel_size_cm: (3,) 体素尺寸
        point_cloud_range_cm: (6,) 点云空间范围
    输出:
        Tensor(K, 3) 体素中心坐标，顺序为 (x, y, z)
    """
    if coords_bzyx.shape[0] == 0:
        return coords_bzyx.new_zeros((0, 3), dtype=torch.float32)

    vx, vy, vz = (float(v) for v in voxel_size_cm)
    x_min, y_min, z_min, _, _, _ = (float(v) for v in point_cloud_range_cm)
    x = x_min + (coords_bzyx[:, 3].to(torch.float32) + 0.5) * vx
    y = y_min + (coords_bzyx[:, 2].to(torch.float32) + 0.5) * vy
    z = z_min + (coords_bzyx[:, 1].to(torch.float32) + 0.5) * vz
    return torch.stack([x, y, z], dim=-1)


def flatten_voxel_grid_for_transformer(
        dense_voxel_feats: torch.Tensor,
        valid_mask: torch.Tensor,
        grid_meta: Dict[str, object]) -> Dict[str, torch.Tensor]:
    """
    将稠密 voxel grid 展平为 token 序列。

    输入:
        dense_voxel_feats: Tensor(B, C, D, H, W) 稠密体素特征
        valid_mask: BoolTensor(B, 1, D, H, W) 有效体素 mask
        grid_meta: dict，包含 voxel_size_cm 与 point_cloud_range_cm
    输出:
        dict，包含 voxel_tokens、voxel_coords_cm、token_mask、sparse_coords
    """
    if dense_voxel_feats.ndim != 5:
        raise ValueError("dense_voxel_feats must have shape (B, C, D, H, W)")
    if valid_mask.shape != (dense_voxel_feats.shape[0], 1, *dense_voxel_feats.shape[2:]):
        raise ValueError("valid_mask shape must be (B, 1, D, H, W)")

    batch_idx, _, z_idx, y_idx, x_idx = valid_mask.nonzero(as_tuple=True)
    sparse_coords = torch.stack([batch_idx, z_idx, y_idx, x_idx], dim=-1)
    voxel_tokens = dense_voxel_feats[batch_idx, :, z_idx, y_idx, x_idx]
    voxel_tokens = voxel_tokens.to(torch.float32)
    voxel_coords_cm = _compute_voxel_centers_cm(
        sparse_coords,
        tuple(grid_meta["voxel_size_cm"]),
        tuple(grid_meta["point_cloud_range_cm"]),
    ).to(dense_voxel_feats.device)
    token_mask = torch.ones(
        (voxel_tokens.shape[0],), dtype=torch.bool, device=dense_voxel_feats.device)

    return {
        "voxel_tokens": voxel_tokens,
        "voxel_coords_cm": voxel_coords_cm,
        "token_mask": token_mask,
        "sparse_coords": sparse_coords,
    }
