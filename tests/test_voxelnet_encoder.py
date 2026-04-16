"""
tests/test_voxelnet_encoder.py
------------------------------
职责：测试 VoxelNet 稠密体素编码器及其 token 展平工具。

测试内容：
- test_voxelize_points_discards_out_of_range_points：验证越界点会被过滤
- test_voxelnet_encoder_returns_stable_dense_shape：验证固定范围下输出 shape 稳定
- test_scale_sensitive_voxelization_changes_occupied_extent：
  验证尺度变大时占据体素范围同步增大
- test_flatten_voxel_grid_for_transformer_matches_valid_mask：
  验证 dense grid 展平后的 token 数与 valid mask 一致
- test_build_padded_voxel_tokens_returns_batch_first_layout：
  验证 batch-first 体素 token 的 padding、mask 和位置坐标正确
- test_pc_backbone_voxelnet_instantiation：验证统一 backbone 入口可实例化

用法：
    pytest tests/test_voxelnet_encoder.py -v
"""

import torch

from src.models.backbones import (
    PCBackbone,
    VoxelNetEncoder,
    build_padded_voxel_tokens,
    flatten_voxel_grid_for_transformer,
    voxelize_points,
)
from src.models.backbones.voxelnet_encoder import build_voxel_grid_spec


def _make_points_batch(num_points: int, scale: float = 1.0) -> torch.Tensor:
    """
    构造一个规则立方体点云 batch。

    输入:
        num_points: int 每轴采样数
        scale: float 立方体边长缩放系数
    输出:
        Tensor(1, N, 3) 单 batch 点云，单位 cm
    """
    axis = torch.linspace(-10.0 * scale, 10.0 * scale, steps=num_points)
    xx, yy, zz = torch.meshgrid(axis, axis, axis, indexing="ij")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=-1)
    return pts.unsqueeze(0)


def test_voxelize_points_discards_out_of_range_points():
    """
    验证越界点会被过滤，且保留点被正确分配到体素。

    输入:
        无，内部构造 3 个点，其中 1 个越界
    输出:
        无，通过断言验证结果
    """
    points_xyz = torch.tensor([
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [100.0, 100.0, 100.0]]
    ], dtype=torch.float32)
    spec = build_voxel_grid_spec(
        voxel_size_cm=(2.0, 2.0, 2.0),
        point_cloud_range_cm=(-10.0, -10.0, -10.0, 10.0, 10.0, 10.0),
    )
    voxel_dict = voxelize_points(
        points_xyz=points_xyz,
        point_feats=None,
        grid_spec=spec,
        max_points_per_voxel=5,
        max_voxels=20,
    )

    assert int(voxel_dict["dropped_points"][0].item()) == 1
    assert voxel_dict["voxel_coords"].shape[0] == 1
    assert voxel_dict["voxel_num_points"].tolist() == [2]
    assert voxel_dict["voxel_coords"][0].tolist() == [0, 5, 5, 5]


def test_voxelnet_encoder_returns_stable_dense_shape():
    """
    验证固定空间范围下，不同点数输入的 dense 输出 shape 保持一致。

    输入:
        无，内部构造两组点数不同的点云
    输出:
        无，通过断言验证结果
    """
    encoder = VoxelNetEncoder(
        voxel_size_cm=(5.0, 5.0, 5.0),
        point_cloud_range_cm=(-20.0, -20.0, -20.0, 20.0, 20.0, 20.0),
        max_points_per_voxel=16,
        max_voxels=256,
        input_feature_dim=6,
        svfe_hidden_channels=32,
        svfe_out_channels=64,
        cml_channels=(64, 64),
    )

    out_small = encoder(_make_points_batch(3))
    out_large = encoder(_make_points_batch(5))

    assert out_small["dense_voxel_feats"].shape == out_large["dense_voxel_feats"].shape
    assert out_small["dense_voxel_feats"].shape == (1, 64, 8, 8, 8)
    assert out_small["valid_mask"].shape == (1, 1, 8, 8, 8)


def test_scale_sensitive_voxelization_changes_occupied_extent():
    """
    验证尺度变大时，占据的体素跨度也会随之增大。

    输入:
        无，内部构造大小不同的规则立方体点云
    输出:
        无，通过断言验证结果
    """
    small_points = _make_points_batch(4, scale=1.0)
    large_points = _make_points_batch(4, scale=2.0)

    spec = build_voxel_grid_spec(
        voxel_size_cm=(5.0, 5.0, 5.0),
        point_cloud_range_cm=(-30.0, -30.0, -30.0, 30.0, 30.0, 30.0),
    )
    small_voxels = voxelize_points(
        points_xyz=small_points,
        point_feats=None,
        grid_spec=spec,
        max_points_per_voxel=16,
        max_voxels=512,
    )
    large_voxels = voxelize_points(
        points_xyz=large_points,
        point_feats=None,
        grid_spec=spec,
        max_points_per_voxel=16,
        max_voxels=512,
    )

    small_extent = small_voxels["voxel_coords"][:, 1:].amax(dim=0) - small_voxels["voxel_coords"][:, 1:].amin(dim=0)
    large_extent = large_voxels["voxel_coords"][:, 1:].amax(dim=0) - large_voxels["voxel_coords"][:, 1:].amin(dim=0)

    assert torch.all(large_extent > small_extent)


def test_flatten_voxel_grid_for_transformer_matches_valid_mask():
    """
    验证 dense grid 展平后的 token 数与 valid mask 一致。

    输入:
        无，内部构造一组规则点云
    输出:
        无，通过断言验证结果
    """
    encoder = VoxelNetEncoder(
        voxel_size_cm=(5.0, 5.0, 5.0),
        point_cloud_range_cm=(-20.0, -20.0, -20.0, 20.0, 20.0, 20.0),
        max_points_per_voxel=8,
        max_voxels=256,
        input_feature_dim=6,
        svfe_hidden_channels=16,
        svfe_out_channels=32,
        cml_channels=(32,),
    )
    outputs = encoder(_make_points_batch(4))
    token_dict = flatten_voxel_grid_for_transformer(
        dense_voxel_feats=outputs["dense_voxel_feats"],
        valid_mask=outputs["valid_mask"],
        grid_meta=outputs["grid_meta"],
    )

    valid_count = int(outputs["valid_mask"].sum().item())
    assert token_dict["voxel_tokens"].shape[0] == valid_count
    assert token_dict["voxel_coords_cm"].shape == (valid_count, 3)
    assert token_dict["sparse_coords"].shape == (valid_count, 4)
    assert token_dict["token_mask"].dtype == torch.bool


def test_pc_backbone_voxelnet_instantiation():
    """
    验证统一 backbone 入口可实例化并返回 voxelnet 输出。

    输入:
        无，内部构造一组规则点云
    输出:
        无，通过断言验证结果
    """
    backbone = PCBackbone({
        "type": "voxelnet",
        "voxel_size_cm": [5.0, 5.0, 5.0],
        "point_cloud_range_cm": [-20.0, -20.0, -20.0, 20.0, 20.0, 20.0],
        "max_points_per_voxel": 8,
        "max_voxels": 256,
        "input_feature_dim": 6,
        "svfe_hidden_channels": 16,
        "svfe_out_channels": 32,
        "cml_channels": [32],
    })
    outputs = backbone(_make_points_batch(4))

    assert "dense_voxel_feats" in outputs
    assert "grid_meta" in outputs
    assert outputs["dense_voxel_feats"].shape[1] == 32


def test_build_padded_voxel_tokens_returns_batch_first_layout():
    """
    验证 batch-first 体素 token 输出具备正确的 padding 与 mask。

    输入:
        无，内部构造两个点云样本
    输出:
        无，通过断言验证结果
    """
    encoder = VoxelNetEncoder(
        voxel_size_cm=(5.0, 5.0, 5.0),
        point_cloud_range_cm=(-20.0, -20.0, -20.0, 20.0, 20.0, 20.0),
        max_points_per_voxel=8,
        max_voxels=256,
        input_feature_dim=6,
        svfe_hidden_channels=16,
        svfe_out_channels=32,
        cml_channels=(32,),
    )
    points_xyz = torch.cat([
        _make_points_batch(4, scale=1.0),
        _make_points_batch(4, scale=0.6),
    ], dim=0)
    outputs = encoder(points_xyz)

    token_dict = build_padded_voxel_tokens(
        dense_voxel_feats=outputs["dense_voxel_feats"],
        valid_mask=outputs["valid_mask"],
        grid_meta=outputs["grid_meta"],
    )

    assert token_dict["tokens"].shape[0] == 2
    assert token_dict["token_mask"].shape[:2] == token_dict["tokens"].shape[:2]
    assert token_dict["token_pos"].shape[-1] == 3
    assert token_dict["token_coords_cm"].shape[-1] == 3
    assert token_dict["sparse_coords"].shape[-1] == 4
    assert token_dict["token_counts"].shape == (2,)
    assert torch.all(token_dict["token_pos"][token_dict["token_mask"]] <= 1.0 + 1e-6)
    assert torch.all(token_dict["token_pos"][token_dict["token_mask"]] >= -1.0 - 1e-6)

    for batch_idx in range(2):
        valid_count = int(token_dict["token_counts"][batch_idx].item())
        assert int(token_dict["token_mask"][batch_idx].sum().item()) == valid_count
        if valid_count < token_dict["tokens"].shape[1]:
            assert not torch.any(token_dict["token_mask"][batch_idx, valid_count:])
            assert torch.all(token_dict["sparse_coords"][batch_idx, valid_count:] == -1)
