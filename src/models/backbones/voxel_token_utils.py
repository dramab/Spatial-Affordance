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
        dict，包含：
        - voxel_tokens: Tensor(K, C) 展平后的有效体素特征
        - voxel_coords_cm: Tensor(K, 3) 体素中心坐标 (x, y, z)，单位 cm
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


def _normalize_voxel_coords(
        voxel_coords_cm: torch.Tensor,
        point_cloud_range_cm: Tuple[float, float, float, float, float, float]) -> torch.Tensor:
    """
    将体素中心坐标归一化到 [-1, 1]。

    输入:
        voxel_coords_cm: Tensor(K, 3) 体素中心坐标，顺序为 (x, y, z)
        point_cloud_range_cm: (6,) 点云空间范围
    输出:
        Tensor(K, 3) 归一化坐标
    """
    if voxel_coords_cm.shape[0] == 0:
        return voxel_coords_cm.new_zeros((0, 3), dtype=torch.float32)

    x_min, y_min, z_min, x_max, y_max, z_max = (
        float(v) for v in point_cloud_range_cm
    )
    mins = voxel_coords_cm.new_tensor([x_min, y_min, z_min], dtype=torch.float32)
    maxs = voxel_coords_cm.new_tensor([x_max, y_max, z_max], dtype=torch.float32)
    spans = (maxs - mins).clamp_min(1e-6)
    normalized = (voxel_coords_cm.to(torch.float32) - mins) / spans
    return normalized * 2.0 - 1.0


def build_padded_voxel_tokens(
        dense_voxel_feats: torch.Tensor,
        valid_mask: torch.Tensor,
        grid_meta: Dict[str, object]) -> Dict[str, torch.Tensor]:
    """
    将稠密体素特征转换为 batch-first 的稀疏 token 序列。

    输入:
        dense_voxel_feats: Tensor(B, C, D, H, W) 稠密体素特征
        valid_mask: BoolTensor(B, 1, D, H, W) 有效体素 mask
        grid_meta: dict，包含 voxel_size_cm 与 point_cloud_range_cm
    输出:
        dict，包含：
        - tokens: Tensor(B, L, C) batch-first 体素 token
        - token_mask: BoolTensor(B, L) True 表示有效 token
        - token_pos: Tensor(B, L, 3) 归一化体素中心坐标
        - token_coords_cm: Tensor(B, L, 3) 原始体素中心坐标（cm）
        - sparse_coords: LongTensor(B, L, 4) 稀疏体素索引，padding 为 -1
        - token_counts: LongTensor(B,) 每个 batch 的有效 token 数
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
    token_coords_cm = dense_voxel_feats.new_zeros((batch_size, max_tokens, 3), dtype=torch.float32)
    token_pos = dense_voxel_feats.new_zeros((batch_size, max_tokens, 3), dtype=torch.float32)
    sparse_coords = torch.full((batch_size, max_tokens, 4), -1, dtype=torch.long, device=device)

    offset = 0
    point_cloud_range_cm = tuple(grid_meta["point_cloud_range_cm"])
    for batch_idx in range(batch_size):
        count = int(token_counts[batch_idx].item())
        if count == 0:
            continue
        next_offset = offset + count
        batch_tokens = flat_dict["voxel_tokens"][offset:next_offset]
        batch_coords_cm = flat_dict["voxel_coords_cm"][offset:next_offset]
        batch_sparse_coords = flat_dict["sparse_coords"][offset:next_offset]

        tokens[batch_idx, :count] = batch_tokens
        token_mask[batch_idx, :count] = True
        token_coords_cm[batch_idx, :count] = batch_coords_cm
        token_pos[batch_idx, :count] = _normalize_voxel_coords(
            batch_coords_cm,
            point_cloud_range_cm=point_cloud_range_cm,
        )
        sparse_coords[batch_idx, :count] = batch_sparse_coords
        offset = next_offset

    return {
        "tokens": tokens,
        "token_mask": token_mask,
        "token_pos": token_pos,
        "token_coords_cm": token_coords_cm,
        "sparse_coords": sparse_coords,
        "token_counts": token_counts,
    }
