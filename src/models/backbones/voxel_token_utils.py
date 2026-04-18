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

    batch_token_dict = build_padded_voxel_tokens(
        dense_voxel_feats=outputs["dense_voxel_feats"],
        valid_mask=outputs["valid_mask"],
        grid_meta=outputs["grid_meta"],
    )
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def _compute_voxel_centers(
        coords_bzyx: torch.Tensor,
        voxel_size: Tuple[float, float, float],
        point_cloud_range: Tuple[float, float, float, float, float, float]) -> torch.Tensor:
    """
    将体素索引转换为体素中心坐标。

    输入:
        coords_bzyx: Tensor(K, 4) 体素索引，顺序为 (batch, z, y, x)
        voxel_size: (3,) 体素尺寸
        point_cloud_range: (6,) 点云空间范围
    输出:
        Tensor(K, 3) 体素中心坐标，顺序为 (x, y, z)
    """
    if coords_bzyx.shape[0] == 0:
        return coords_bzyx.new_zeros((0, 3), dtype=torch.float32)

    vx, vy, vz = (float(v) for v in voxel_size)
    x_min, y_min, z_min, _, _, _ = (float(v) for v in point_cloud_range)
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
        grid_meta: dict，包含 voxel_size 与 point_cloud_range
    输出:
        dict，包含：
        - voxel_tokens: Tensor(K, C) 展平后的有效体素特征
        - voxel_coords: Tensor(K, 3) 体素中心坐标 (x, y, z)
        - token_mask: BoolTensor(K,) 有效 token 掩码（恒为 True）
        - sparse_coords: LongTensor(K, 4) 稀疏体素索引 (batch, z, y, x)
    """
    if dense_voxel_feats.ndim != 5:
        raise ValueError("dense_voxel_feats must have shape (B, C, D, H, W)")
    if valid_mask.shape != (dense_voxel_feats.shape[0], 1, *dense_voxel_feats.shape[2:]):
        raise ValueError("valid_mask shape must be (B, 1, D, H, W)")

    batch_idx, _, z_idx, y_idx, x_idx = valid_mask.nonzero(as_tuple=True)
    sparse_coords = torch.stack([batch_idx, z_idx, y_idx, x_idx], dim=-1)
    voxel_tokens = dense_voxel_feats[batch_idx, :, z_idx, y_idx, x_idx]
    voxel_tokens = voxel_tokens.to(torch.float32)
    voxel_coords = _compute_voxel_centers(
        sparse_coords,
        tuple(grid_meta["voxel_size"]),
        tuple(grid_meta["point_cloud_range"]),
    ).to(dense_voxel_feats.device)
    token_mask = torch.ones(
        (voxel_tokens.shape[0],), dtype=torch.bool, device=dense_voxel_feats.device)

    return {
        "voxel_tokens": voxel_tokens,
        "voxel_coords": voxel_coords,
        "token_mask": token_mask,
        "sparse_coords": sparse_coords,
    }


def build_padded_voxel_tokens(
        dense_voxel_feats: torch.Tensor,
        valid_mask: torch.Tensor,
        grid_meta: Dict[str, object]) -> Dict[str, torch.Tensor]:
    """
    将稠密体素特征转换为 batch-first 的稀疏 token 序列。

    输入:
        dense_voxel_feats: Tensor(B, C, D, H, W) 稠密体素特征
        valid_mask: BoolTensor(B, 1, D, H, W) 有效体素 mask
        grid_meta: dict，包含 voxel_size 与 point_cloud_range
    输出:
        dict，包含：
        - tokens: Tensor(B, L, C) batch-first 体素 token
        - token_mask: BoolTensor(B, L) True 表示有效 token
        - token_pos: Tensor(B, L, 3) 用于位置编码的体素中心坐标
    """
    flat_dict = flatten_voxel_grid_for_transformer(
        dense_voxel_feats=dense_voxel_feats,
        valid_mask=valid_mask,
        grid_meta=grid_meta,
    )
    batch_size = int(dense_voxel_feats.shape[0])
    token_counts = valid_mask.view(batch_size, -1).sum(dim=1).to(torch.long)
    max_tokens = int(token_counts.max().item()) if batch_size > 0 else 0
    channels = int(dense_voxel_feats.shape[1])
    device = dense_voxel_feats.device

    tokens = dense_voxel_feats.new_zeros((batch_size, max_tokens, channels))
    token_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool, device=device)
    token_pos = dense_voxel_feats.new_zeros((batch_size, max_tokens, 3), dtype=torch.float32)
    num_valid_tokens = int(flat_dict["voxel_tokens"].shape[0])
    if num_valid_tokens > 0 and max_tokens > 0:
        batch_indices = flat_dict["sparse_coords"][:, 0].to(torch.long)
        # flatten 输出天然按 batch 聚合，这里直接恢复每个 token 在各自 batch 内的相对位置。
        batch_offsets = torch.cumsum(token_counts, dim=0) - token_counts
        token_indices = (
            torch.arange(num_valid_tokens, device=device, dtype=torch.long) -
            batch_offsets[batch_indices]
        )
        tokens[batch_indices, token_indices] = flat_dict["voxel_tokens"]
        token_mask[batch_indices, token_indices] = True
        token_pos[batch_indices, token_indices] = flat_dict["voxel_coords"]

    return {
        "tokens": tokens,
        "token_mask": token_mask,
        "token_pos": token_pos,
    }
