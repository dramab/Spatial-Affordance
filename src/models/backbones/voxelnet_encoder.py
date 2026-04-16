"""
src/models/backbones/voxelnet_encoder.py
----------------------------------------
职责：基于 VoxelNet 前半段结构的场景点云稠密体素编码器。

功能：
- 将整场景点云按固定范围体素化
- 使用 VFE / SVFE 聚合体素内点特征
- 使用 3D 卷积中层提取稠密 voxel embedding
- 输出 dense grid、稀疏体素特征、有效体素 mask 与网格元信息

用法：
    from src.models.backbones.voxelnet_encoder import VoxelNetEncoder

    encoder = VoxelNetEncoder(
        voxel_size_cm=(2.0, 2.0, 2.0),
        point_cloud_range_cm=(-80.0, -80.0, -10.0, 80.0, 80.0, 120.0),
    )
    outputs = encoder(points_xyz, point_feats)
    dense_feats = outputs["dense_voxel_feats"]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class VoxelGridSpec:
    """
    固定体素网格规格。

    输入:
        voxel_size_cm: Sequence[float] 体素尺寸，顺序为 (x, y, z)
        point_cloud_range_cm: Sequence[float] 点云范围，
            顺序为 (x_min, y_min, z_min, x_max, y_max, z_max)
    输出:
        VoxelGridSpec 实例
    """

    voxel_size_cm: Tuple[float, float, float]
    point_cloud_range_cm: Tuple[float, float, float, float, float, float]

    @property
    def grid_shape_xyz(self) -> Tuple[int, int, int]:
        """
        计算体素网格在 xyz 方向上的尺寸。

        输入:
            无
        输出:
            (grid_x, grid_y, grid_z)
        """
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range_cm
        vx, vy, vz = self.voxel_size_cm
        grid_x = int(round((x_max - x_min) / vx))
        grid_y = int(round((y_max - y_min) / vy))
        grid_z = int(round((z_max - z_min) / vz))
        if min(grid_x, grid_y, grid_z) <= 0:
            raise ValueError("invalid voxel grid shape derived from config")
        return grid_x, grid_y, grid_z

    @property
    def grid_shape_dhw(self) -> Tuple[int, int, int]:
        """
        返回适配 3D 卷积的网格尺寸。

        输入:
            无
        输出:
            (depth_z, height_y, width_x)
        """
        grid_x, grid_y, grid_z = self.grid_shape_xyz
        return grid_z, grid_y, grid_x


def _as_3tuple(value: Sequence[float] | float, name: str) -> Tuple[float, float, float]:
    """
    将标量或长度为 3 的序列标准化为三元组。

    输入:
        value: 标量或长度为 3 的序列
        name: str 参数名，用于报错
    输出:
        (x, y, z) 三元组
    """
    if isinstance(value, (int, float)):
        scalar = float(value)
        return (scalar, scalar, scalar)
    if len(value) != 3:
        raise ValueError(f"{name} must be a float or length-3 sequence")
    return (float(value[0]), float(value[1]), float(value[2]))


def _as_6tuple(value: Sequence[float], name: str) -> Tuple[float, float, float, float, float, float]:
    """
    将长度为 6 的序列标准化为六元组。

    输入:
        value: 长度为 6 的序列
        name: str 参数名，用于报错
    输出:
        (x_min, y_min, z_min, x_max, y_max, z_max)
    """
    if len(value) != 6:
        raise ValueError(f"{name} must be a length-6 sequence")
    return tuple(float(v) for v in value)


def build_voxel_grid_spec(
        voxel_size_cm: Sequence[float] | float,
        point_cloud_range_cm: Sequence[float]) -> VoxelGridSpec:
    """
    构造固定体素网格规格。

    输入:
        voxel_size_cm: float 或 (3,) 体素尺寸，单位 cm
        point_cloud_range_cm: (6,) 点云空间范围，单位 cm
    输出:
        VoxelGridSpec 规格对象
    """
    return VoxelGridSpec(
        voxel_size_cm=_as_3tuple(voxel_size_cm, "voxel_size_cm"),
        point_cloud_range_cm=_as_6tuple(point_cloud_range_cm, "point_cloud_range_cm"),
    )


def voxelize_points(
        points_xyz: torch.Tensor,
        point_feats: Optional[torch.Tensor],
        grid_spec: VoxelGridSpec,
        max_points_per_voxel: int,
        max_voxels: int) -> Dict[str, torch.Tensor]:
    """
    将整批场景点云体素化为固定范围的稀疏体素表示。

    输入:
        points_xyz: Tensor(B, N, 3) 点云 xyz，单位 cm
        point_feats: Tensor(B, N, F) 额外点特征，可为 None
        grid_spec: VoxelGridSpec 固定体素网格规格
        max_points_per_voxel: int 每个体素保留的最多点数
        max_voxels: int 每个 batch 最多保留的体素数
    输出:
        dict，各字段说明如下（设 K 为所有 batch 的有效体素总数，
        T = max_points_per_voxel，C = 6 + F 为特征维度，B 为 batch size）：
        - voxel_features: Tensor(K, T, C)
            每个体素内各点的拼接特征（坐标、相对均值坐标、可选额外特征）
        - voxel_coords: LongTensor(K, 4)
            体素在稠密网格中的索引，顺序为 (batch_idx, z, y, x)
        - voxel_num_points: LongTensor(K,)
            每个体素内实际包含的有效点数（不超过 T）
        - voxel_point_mask: BoolTensor(K, T)
            True 表示该体素内对应槽位为有效点，False 为 padding
        - points_per_batch: LongTensor(B,)
            每个 batch 最终保留的有效体素数量
        - dropped_points: LongTensor(B,)
            每个 batch 因超出 point_cloud_range 而被丢弃的点数
        - grid_shape_dhw: LongTensor(3,)
            稠密体素网格的形状 (D, H, W)
    """
    if points_xyz.ndim != 3 or points_xyz.shape[-1] != 3:
        raise ValueError("points_xyz must have shape (B, N, 3)")
    if point_feats is not None and point_feats.shape[:2] != points_xyz.shape[:2]:
        raise ValueError("point_feats must align with points_xyz on batch and point dims")

    spec = grid_spec
    vx, vy, vz = spec.voxel_size_cm
    x_min, y_min, z_min, x_max, y_max, z_max = spec.point_cloud_range_cm

    feat_dim = 6 + (0 if point_feats is None else int(point_feats.shape[-1]))
    device = points_xyz.device
    dtype = points_xyz.dtype

    voxel_features = []
    voxel_coords = []
    voxel_num_points = []
    points_per_batch = []
    dropped_points = []

    for batch_idx in range(points_xyz.shape[0]):
        xyz = points_xyz[batch_idx]
        extras = None if point_feats is None else point_feats[batch_idx]

        valid = (
            (xyz[:, 0] >= x_min) & (xyz[:, 0] < x_max) &
            (xyz[:, 1] >= y_min) & (xyz[:, 1] < y_max) &
            (xyz[:, 2] >= z_min) & (xyz[:, 2] < z_max)
        )
        xyz = xyz[valid]
        if extras is not None:
            extras = extras[valid]

        dropped_points.append(int((~valid).sum().item()))

        if xyz.shape[0] == 0:
            points_per_batch.append(0)
            continue

        coords_x = torch.floor((xyz[:, 0] - x_min) / vx).long()
        coords_y = torch.floor((xyz[:, 1] - y_min) / vy).long()
        coords_z = torch.floor((xyz[:, 2] - z_min) / vz).long()
        coords_xyz = torch.stack([coords_x, coords_y, coords_z], dim=-1)

        unique_coords, inverse = torch.unique(
            coords_xyz, dim=0, sorted=True, return_inverse=True)
        if unique_coords.shape[0] > max_voxels:
            unique_coords = unique_coords[:max_voxels]
            keep_mask = inverse < max_voxels
            xyz = xyz[keep_mask]
            coords_xyz = coords_xyz[keep_mask]
            inverse = inverse[keep_mask]
            if extras is not None:
                extras = extras[keep_mask]
        num_voxels = unique_coords.shape[0]
        points_per_batch.append(int(num_voxels))

        voxel_xyz = torch.zeros(
            (num_voxels, max_points_per_voxel, 3), dtype=dtype, device=device)
        voxel_extra = None
        if extras is not None:
            voxel_extra = torch.zeros(
                (num_voxels, max_points_per_voxel, extras.shape[-1]),
                dtype=extras.dtype,
                device=device,
            )
        point_counts = torch.zeros(num_voxels, dtype=torch.long, device=device)

        # 收集每个体素包含的点索引，并随机采样至 max_points_per_voxel
        voxel_point_indices: list[list[int]] = [[] for _ in range(num_voxels)]
        for point_idx in range(xyz.shape[0]):
            voxel_idx = int(inverse[point_idx].item())
            voxel_point_indices[voxel_idx].append(point_idx)

        for voxel_idx in range(num_voxels):
            indices = voxel_point_indices[voxel_idx]
            if len(indices) > max_points_per_voxel:
                perm = torch.randperm(len(indices), device=device)
                indices = [indices[i] for i in perm[:max_points_per_voxel].tolist()]
            for count, point_idx in enumerate(indices):
                voxel_xyz[voxel_idx, count] = xyz[point_idx]
                if voxel_extra is not None:
                    voxel_extra[voxel_idx, count] = extras[point_idx]
            point_counts[voxel_idx] = len(indices)

        non_empty = point_counts > 0
        voxel_xyz = voxel_xyz[non_empty]
        point_counts = point_counts[non_empty]
        unique_coords = unique_coords[non_empty]
        if voxel_extra is not None:
            voxel_extra = voxel_extra[non_empty]

        points_mask = (
            torch.arange(max_points_per_voxel, device=device)[None, :] <
            point_counts[:, None]
        )
        mean_xyz = (
            voxel_xyz.sum(dim=1, keepdim=True) /
            point_counts[:, None, None].clamp_min(1).to(dtype)
        )
        rel_xyz = voxel_xyz - mean_xyz
        features = [voxel_xyz, rel_xyz]
        if voxel_extra is not None:
            features.append(voxel_extra)
        feature_tensor = torch.cat(features, dim=-1)
        feature_tensor = feature_tensor * points_mask.unsqueeze(-1).to(dtype)

        coords_zyx = torch.stack(
            [unique_coords[:, 2], unique_coords[:, 1], unique_coords[:, 0]], dim=-1)
        batch_column = torch.full(
            (coords_zyx.shape[0], 1), batch_idx, dtype=torch.long, device=device)
        coords_bzyx = torch.cat([batch_column, coords_zyx], dim=-1)

        voxel_features.append(feature_tensor)
        voxel_coords.append(coords_bzyx)
        voxel_num_points.append(point_counts)

    if voxel_features:
        voxel_features = torch.cat(voxel_features, dim=0)
        voxel_coords = torch.cat(voxel_coords, dim=0)
        voxel_num_points = torch.cat(voxel_num_points, dim=0)
        voxel_point_mask = (
            torch.arange(max_points_per_voxel, device=device)[None, :] <
            voxel_num_points[:, None]
        )
    else:
        voxel_features = torch.zeros(
            (0, max_points_per_voxel, feat_dim), dtype=dtype, device=device)
        voxel_coords = torch.zeros((0, 4), dtype=torch.long, device=device)
        voxel_num_points = torch.zeros((0,), dtype=torch.long, device=device)
        voxel_point_mask = torch.zeros(
            (0, max_points_per_voxel), dtype=torch.bool, device=device)

    return {
        "voxel_features": voxel_features,
        "voxel_coords": voxel_coords,
        "voxel_num_points": voxel_num_points,
        "voxel_point_mask": voxel_point_mask,
        "points_per_batch": torch.tensor(points_per_batch, dtype=torch.long, device=device),
        "dropped_points": torch.tensor(dropped_points, dtype=torch.long, device=device),
        "grid_shape_dhw": torch.tensor(spec.grid_shape_dhw, dtype=torch.long, device=device),
    }


class FCN(nn.Module):
    """
    逐点线性映射层。

    输入:
        features: Tensor(K, T, C_in) 体素内点特征
    输出:
        Tensor(K, T, C_out) 逐点映射后的特征
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        逐点映射并归一化。

        输入:
            features: Tensor(K, T, C_in)
        输出:
            Tensor(K, T, C_out)
        """
        if features.numel() == 0:
            return features.new_zeros(
                (features.shape[0], features.shape[1], self.linear.out_features))
        x = self.linear(features)
        x = x.reshape(-1, x.shape[-1])
        # 只有 1 个有效样本时，BatchNorm1d 在训练态会报错，这里退化为仅激活。
        if x.shape[0] > 1:
            x = self.norm(x)
        x = x.reshape(features.shape[0], features.shape[1], -1)
        return self.act(x)


class VFE(nn.Module):
    """
    Voxel Feature Encoding 层。

    输入:
        features: Tensor(K, T, C_in) 体素内点特征
        point_mask: BoolTensor(K, T) 有效点 mask
    输出:
        Tensor(K, T, C_out) 融合局部上下文后的逐点特征
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if out_channels % 2 != 0:
            raise ValueError("VFE out_channels must be even")
        self.units = out_channels // 2
        self.fcn = FCN(in_channels, self.units)

    def forward(
            self,
            features: torch.Tensor,
            point_mask: torch.Tensor) -> torch.Tensor:
        """
        聚合体素内逐点特征与体素级上下文。

        输入:
            features: Tensor(K, T, C_in)
            point_mask: BoolTensor(K, T)
        输出:
            Tensor(K, T, C_out)
        """
        pwf = self.fcn(features)
        mask = point_mask.unsqueeze(-1)
        pwf = pwf * mask
        aggregated = pwf.max(dim=1, keepdim=True).values.expand_as(pwf)
        concatenated = torch.cat([pwf, aggregated], dim=-1)
        return concatenated * mask


class SVFE(nn.Module):
    """
    Stacked VFE 体素特征提取器。

    输入:
        voxel_features: Tensor(K, T, C_in)
        point_mask: BoolTensor(K, T)
    输出:
        Tensor(K, C_out) 每个体素的稀疏特征
    """

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int = 32,
            out_channels: int = 128):
        super().__init__()
        self.vfe_1 = VFE(in_channels, hidden_channels)
        self.vfe_2 = VFE(hidden_channels, hidden_channels)
        self.fcn = FCN(hidden_channels, out_channels)

    def forward(
            self,
            voxel_features: torch.Tensor,
            point_mask: torch.Tensor) -> torch.Tensor:
        """
        将体素内变长点集编码成单个体素向量。

        输入:
            voxel_features: Tensor(K, T, C_in)
            point_mask: BoolTensor(K, T)
        输出:
            Tensor(K, C_out)
        """
        x = self.vfe_1(voxel_features, point_mask)
        x = self.vfe_2(x, point_mask)
        x = self.fcn(x) * point_mask.unsqueeze(-1)
        return x.max(dim=1).values


class ConvMiddleLayers(nn.Module):
    """
    稠密体素网格的 3D 卷积中层。

    输入:
        dense_tensor: Tensor(B, C_in, D, H, W)
    输出:
        Tensor(B, C_out, D, H, W)
    """

    def __init__(self, in_channels: int, channels: Sequence[int]):
        super().__init__()
        layers = []
        prev = in_channels
        for out_channels in channels:
            layers.extend([
                nn.Conv3d(prev, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
            ])
            prev = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, dense_tensor: torch.Tensor) -> torch.Tensor:
        """
        提取稠密 voxel 特征。

        输入:
            dense_tensor: Tensor(B, C_in, D, H, W)
        输出:
            Tensor(B, C_out, D, H, W)
        """
        return self.net(dense_tensor)


def voxel_indexing(
        sparse_features: torch.Tensor,
        coords_bzyx: torch.Tensor,
        batch_size: int,
        grid_shape_dhw: Sequence[int]) -> torch.Tensor:
    """
    将稀疏体素特征散射回稠密 5D 网格。

    输入:
        sparse_features: Tensor(K, C) 稀疏体素特征
        coords_bzyx: Tensor(K, 4) 稀疏坐标，顺序为 (batch, z, y, x)
        batch_size: int batch 大小
        grid_shape_dhw: (3,) 网格尺寸 (D, H, W)
    输出:
        Tensor(B, C, D, H, W) 稠密体素特征
    """
    depth, height, width = (int(v) for v in grid_shape_dhw)
    channels = int(sparse_features.shape[-1])
    dense = sparse_features.new_zeros((batch_size, channels, depth, height, width))
    if sparse_features.shape[0] == 0:
        return dense
    batch_idx = coords_bzyx[:, 0].long()
    z_idx = coords_bzyx[:, 1].long()
    y_idx = coords_bzyx[:, 2].long()
    x_idx = coords_bzyx[:, 3].long()
    dense[batch_idx, :, z_idx, y_idx, x_idx] = sparse_features
    return dense


class VoxelNetEncoder(nn.Module):
    """
    基于 VoxelNet 前半段结构的稠密体素编码器。

    输入:
        points_xyz: Tensor(B, N, 3) 场景点云坐标，单位 cm
        point_feats: Tensor(B, N, F) 额外点特征，可为 None
    输出:
        dict，包含 dense_voxel_feats、valid_mask、sparse_voxel_feats、
        sparse_coords、grid_meta 等字段
    """

    def __init__(
            self,
            voxel_size_cm: Sequence[float] | float,
            point_cloud_range_cm: Sequence[float],
            max_points_per_voxel: int = 32,
            max_voxels: int = 20000,
            input_feature_dim: int = 6,
            svfe_hidden_channels: int = 32,
            svfe_out_channels: int = 128,
            cml_channels: Sequence[int] = (128, 256, 256),
            return_dense: bool = True):
        super().__init__()
        self.grid_spec = build_voxel_grid_spec(voxel_size_cm, point_cloud_range_cm)
        self.max_points_per_voxel = int(max_points_per_voxel)
        self.max_voxels = int(max_voxels)
        self.return_dense = bool(return_dense)

        self.svfe = SVFE(
            in_channels=int(input_feature_dim),
            hidden_channels=int(svfe_hidden_channels),
            out_channels=int(svfe_out_channels),
        )
        self.cml = ConvMiddleLayers(
            in_channels=int(svfe_out_channels),
            channels=tuple(int(v) for v in cml_channels),
        )
        self.out_channels = int(cml_channels[-1]) if len(cml_channels) > 0 else int(svfe_out_channels)

    def forward(
            self,
            points_xyz: torch.Tensor,
            point_feats: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor | Dict[str, object]]:
        """
        编码整场景点云为稠密 voxel embedding。

        输入:
            points_xyz: Tensor(B, N, 3) 点云坐标，单位 cm
            point_feats: Tensor(B, N, F) 可选附加特征
        输出:
            dict，至少包含：
                dense_voxel_feats: Tensor(B, C, D, H, W)
                    经 3D 卷积后的稠密体素特征，空体素位置为 0
                valid_mask: BoolTensor(B, 1, D, H, W)
                    标记稠密网格中哪些体素包含有效点
                sparse_voxel_feats: Tensor(K, C)
                    从稠密网格中按有效体素坐标提取的 CML 后稀疏特征，K 为有效体素总数
                sparse_coords: LongTensor(K, 4)
                    有效体素在稠密网格中的索引，顺序为 (batch_idx, z, y, x)
                grid_meta: dict 元信息
                    包含 voxel_size_cm、point_cloud_range_cm、grid_shape_dhw
        """
        voxel_dict = voxelize_points(
            points_xyz=points_xyz,
            point_feats=point_feats,
            grid_spec=self.grid_spec,
            max_points_per_voxel=self.max_points_per_voxel,
            max_voxels=self.max_voxels,
        )
        sparse_voxel_feats = self.svfe(
            voxel_dict["voxel_features"],
            voxel_dict["voxel_point_mask"],
        )

        dense_input = voxel_indexing(
            sparse_features=sparse_voxel_feats,
            coords_bzyx=voxel_dict["voxel_coords"],
            batch_size=int(points_xyz.shape[0]),
            grid_shape_dhw=self.grid_spec.grid_shape_dhw,
        )
        dense_feats = self.cml(dense_input)

        valid_mask = dense_input.abs().sum(dim=1, keepdim=True) > 0
        sparse_after_cml = dense_feats[
            voxel_dict["voxel_coords"][:, 0].long(),
            :,
            voxel_dict["voxel_coords"][:, 1].long(),
            voxel_dict["voxel_coords"][:, 2].long(),
            voxel_dict["voxel_coords"][:, 3].long(),
        ] if voxel_dict["voxel_coords"].shape[0] > 0 else dense_feats.new_zeros((0, dense_feats.shape[1]))

        outputs: Dict[str, torch.Tensor | Dict[str, object]] = {
            "dense_voxel_feats": dense_feats if self.return_dense else dense_input,
            "valid_mask": valid_mask,
            "sparse_voxel_feats": sparse_after_cml,
            "sparse_coords": voxel_dict["voxel_coords"],
            "voxel_num_points": voxel_dict["voxel_num_points"],
            "points_per_batch": voxel_dict["points_per_batch"],
            "dropped_points": voxel_dict["dropped_points"],
            "grid_meta": {
                "voxel_size_cm": self.grid_spec.voxel_size_cm,
                "point_cloud_range_cm": self.grid_spec.point_cloud_range_cm,
                "grid_shape_dhw": self.grid_spec.grid_shape_dhw,
            },
        }
        return outputs
