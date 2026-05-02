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

import pytest
import torch

from src.models.backbones import (
    PCBackbone,
    PointTransformerV3Encoder,
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
        Tensor(1, N, 3) 单 batch normalized 点云
    """
    axis = torch.linspace(-1.0 * scale, 1.0 * scale, steps=num_points)
    xx, yy, zz = torch.meshgrid(axis, axis, axis, indexing="ij")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=-1)
    return pts.unsqueeze(0)


def _make_light_ptv3_cfg(enable_flash: bool = False) -> dict:
    """
    构造轻量化 PointTransformerV3 encoder-only 测试配置。

    输入:
        enable_flash: bool 是否启用 FlashAttention
    输出:
        dict PTv3 点云 backbone 配置
    """
    return {
        "type": "pointtransformerv3",
        "grid_size": 0.1,
        "point_cloud_range": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        "in_channels": 3,
        "order": ["z"],
        "stride": [2],
        "enc_depths": [1, 1],
        "enc_channels": [8, 16],
        "enc_num_head": [1, 2],
        "enc_patch_size": [16, 16],
        "enable_flash": enable_flash,
        "enable_rpe": False,
        "upcast_attention": False,
        "upcast_softmax": False,
        "drop_path": 0.0,
        "shuffle_orders": False,
    }


def test_voxelize_points_discards_out_of_range_points():
    """
    验证越界点会被过滤，且保留点被正确分配到体素。

    输入:
        无，内部构造 3 个点，其中 1 个越界
    输出:
        无，通过断言验证结果
    """
    points_xyz = torch.tensor([
        [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [1.2, 1.2, 1.2]]
    ], dtype=torch.float32)
    spec = build_voxel_grid_spec(
        voxel_size=(0.25, 0.25, 0.25),
        point_cloud_range=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
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
    assert voxel_dict["voxel_coords"][0].tolist() == [0, 4, 4, 4]


def test_voxelize_points_respects_sorted_max_voxel_truncation():
    """
    验证 max_voxels 截断时，仍保留排序后最靠前的体素。

    输入:
        无，内部构造 5 个落在不同体素内的点
    输出:
        无，通过断言验证结果
    """
    points_xyz = torch.tensor([
        [
            [-0.9, 0.0, 0.0],
            [-0.4, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [0.9, 0.0, 0.0],
        ]
    ], dtype=torch.float32)
    spec = build_voxel_grid_spec(
        voxel_size=(0.5, 0.5, 0.5),
        point_cloud_range=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
    )

    voxel_dict = voxelize_points(
        points_xyz=points_xyz,
        point_feats=None,
        grid_spec=spec,
        max_points_per_voxel=4,
        max_voxels=3,
    )

    assert int(voxel_dict["points_per_batch"][0].item()) == 3
    assert voxel_dict["voxel_coords"].tolist() == [
        [0, 2, 2, 0],
        [0, 2, 2, 1],
        [0, 2, 2, 2],
    ]


def test_build_voxel_grid_spec_uses_normalized_defaults():
    """
    验证 normalized 默认网格参数可以导出 80x80x80 的体素网格。

    输入:
        无
    输出:
        无，通过断言验证结果
    """
    spec = build_voxel_grid_spec(
        voxel_size=(0.025, 0.025, 0.025),
        point_cloud_range=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
    )

    assert spec.grid_shape_xyz == (80, 80, 80)
    assert spec.grid_shape_dhw == (80, 80, 80)


def test_voxelnet_encoder_returns_stable_dense_shape():
    """
    验证固定空间范围下，不同点数输入的 dense 输出 shape 保持一致。

    输入:
        无，内部构造两组点数不同的点云
    输出:
        无，通过断言验证结果
    """
    encoder = VoxelNetEncoder(
        voxel_size=(0.25, 0.25, 0.25),
        point_cloud_range=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        max_points_per_voxel=16,
        max_voxels=256,
        input_feature_dim=6,
        svfe_hidden_channels=8,
        svfe_out_channels=8,
        cml_channels=(8,),
    )

    out_small = encoder(_make_points_batch(3))
    out_large = encoder(_make_points_batch(5))

    assert out_small["dense_voxel_feats"].shape == out_large["dense_voxel_feats"].shape
    assert out_small["dense_voxel_feats"].shape == (1, 8, 8, 8, 8)
    assert out_small["valid_mask"].shape == (1, 1, 8, 8, 8)


def test_scale_sensitive_voxelization_changes_occupied_extent():
    """
    验证尺度变大时，占据的体素跨度也会随之增大。

    输入:
        无，内部构造大小不同的规则立方体点云
    输出:
        无，通过断言验证结果
    """
    small_points = _make_points_batch(4, scale=0.5)
    large_points = _make_points_batch(4, scale=1.0)

    spec = build_voxel_grid_spec(
        voxel_size=(0.25, 0.25, 0.25),
        point_cloud_range=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
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


def test_voxelize_points_caps_points_per_voxel_without_duplicates():
    """
    验证单个体素内点数超限时，会无放回采样到 max_points_per_voxel。

    输入:
        无，内部构造 6 个落在同一体素内的不同点
    输出:
        无，通过断言验证结果
    """
    torch.manual_seed(0)
    points_xyz = torch.tensor([
        [
            [0.00, 0.00, 0.00],
            [0.05, 0.02, 0.01],
            [0.10, 0.03, 0.02],
            [0.12, 0.04, 0.03],
            [0.15, 0.05, 0.04],
            [0.18, 0.06, 0.05],
        ]
    ], dtype=torch.float32)
    spec = build_voxel_grid_spec(
        voxel_size=(0.5, 0.5, 0.5),
        point_cloud_range=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
    )

    voxel_dict = voxelize_points(
        points_xyz=points_xyz,
        point_feats=None,
        grid_spec=spec,
        max_points_per_voxel=4,
        max_voxels=8,
    )

    selected_xyz = voxel_dict["voxel_features"][0, :4, :3]
    selected_xyz_unique = torch.unique(selected_xyz, dim=0)
    source_xyz = points_xyz[0]

    assert voxel_dict["voxel_coords"].shape[0] == 1
    assert voxel_dict["voxel_num_points"].tolist() == [4]
    assert selected_xyz_unique.shape[0] == 4
    for point in selected_xyz_unique:
        matches = torch.isclose(source_xyz, point.unsqueeze(0)).all(dim=1)
        assert torch.any(matches)


def test_flatten_voxel_grid_for_transformer_matches_valid_mask():
    """
    验证 dense grid 展平后的 token 数与 valid mask 一致。

    输入:
        无，内部构造一组规则点云
    输出:
        无，通过断言验证结果
    """
    encoder = VoxelNetEncoder(
        voxel_size=(0.25, 0.25, 0.25),
        point_cloud_range=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
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
    assert token_dict["voxel_coords"].shape == (valid_count, 3)
    assert token_dict["sparse_coords"].shape == (valid_count, 4)
    assert token_dict["token_mask"].dtype == torch.bool
    assert torch.all(token_dict["voxel_coords"] <= 1.0)
    assert torch.all(token_dict["voxel_coords"] >= -1.0)


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
        "voxel_size": [0.25, 0.25, 0.25],
        "point_cloud_range": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
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
    assert outputs["grid_meta"]["point_cloud_range"] == (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)


def test_pointtransformer_v3_encoder_filters_invalid_points():
    """
    验证 PTv3 适配器会过滤 padding NaN 和越界点。

    输入:
        无，内部构造包含有效点、NaN padding 和越界点的 batch
    输出:
        无，通过断言验证展开后的有效点
    """
    pytest.importorskip("spconv.pytorch")
    pytest.importorskip("torch_scatter")

    encoder = PointTransformerV3Encoder(_make_light_ptv3_cfg(enable_flash=False))
    points_xyz = torch.tensor([
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.2, 0.2],
            [float("nan"), float("nan"), float("nan")],
            [1.2, 0.0, 0.0],
        ],
        [
            [-0.4, -0.4, -0.4],
            [0.8, 0.8, 0.8],
            [0.0, 1.1, 0.0],
            [float("nan"), float("nan"), float("nan")],
        ],
    ], dtype=torch.float32)

    flat_xyz, flat_feats, batch_indices = encoder._flatten_valid_points(points_xyz, None)

    assert flat_xyz.shape == (4, 3)
    assert flat_feats.shape == (4, 3)
    assert batch_indices.tolist() == [0, 0, 1, 1]
    assert torch.isfinite(flat_xyz).all()


def test_pc_backbone_pointtransformer_v3_encoder_only_forward():
    """
    验证统一 backbone 可实例化 PTv3 encoder-only，并返回 batch-first token。

    输入:
        无，内部构造随机 normalized 点云
    输出:
        无，通过断言验证 token 输出和 decoder 不存在
    """
    pytest.importorskip("spconv.pytorch")
    pytest.importorskip("torch_scatter")
    pytest.importorskip("flash_attn")
    if not torch.cuda.is_available():
        pytest.skip("PointTransformerV3 FlashAttention forward requires CUDA")

    device = torch.device("cuda")
    backbone = PCBackbone(_make_light_ptv3_cfg(enable_flash=True)).to(device).eval()
    points_xyz = torch.rand((2, 64, 3), device=device) * 1.6 - 0.8
    points_xyz[0, -4:] = float("nan")

    with torch.no_grad():
        outputs = backbone(points_xyz)

    assert not hasattr(backbone.backbone.model, "dec")
    assert set(outputs.keys()) == {"tokens", "token_mask", "token_pos"}
    assert outputs["tokens"].shape[:2] == outputs["token_mask"].shape
    assert outputs["token_pos"].shape[:2] == outputs["token_mask"].shape
    assert outputs["tokens"].shape[-1] == 16
    assert outputs["token_mask"].dtype == torch.bool
    assert outputs["token_mask"].any()


def test_pointtransformer_v3_non_flash_handles_empty_middle_sample():
    """
    验证 PTv3 非 Flash 路径可处理 batch 中间的空样本。

    输入:
        无，内部构造第二个样本全为 NaN padding 的 batch
    输出:
        无，通过断言验证空样本 token mask 为空
    """
    pytest.importorskip("spconv.pytorch")
    pytest.importorskip("torch_scatter")
    if not torch.cuda.is_available():
        pytest.skip("PointTransformerV3 sparse convolution forward requires CUDA")

    device = torch.device("cuda")
    backbone = PCBackbone(_make_light_ptv3_cfg(enable_flash=False)).to(device).eval()
    points_xyz = torch.rand((3, 64, 3), device=device) * 1.6 - 0.8
    points_xyz[1] = float("nan")

    with torch.no_grad():
        outputs = backbone(points_xyz)

    assert outputs["token_mask"].shape[0] == 3
    assert outputs["token_mask"][0].any()
    assert not outputs["token_mask"][1].any()
    assert outputs["token_mask"][2].any()


def test_build_padded_voxel_tokens_returns_batch_first_layout():
    """
    验证 batch-first 体素 token 输出具备正确的 padding 与 mask。

    输入:
        无，内部构造两个点云样本
    输出:
        无，通过断言验证结果
    """
    encoder = VoxelNetEncoder(
        voxel_size=(0.25, 0.25, 0.25),
        point_cloud_range=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
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
    assert torch.all(token_dict["token_pos"][token_dict["token_mask"]] <= 1.0 + 1e-6)
    assert torch.all(token_dict["token_pos"][token_dict["token_mask"]] >= -1.0 - 1e-6)

    for batch_idx in range(2):
        valid_count = int(token_dict["token_mask"][batch_idx].sum().item())
        assert int(token_dict["token_mask"][batch_idx].sum().item()) == valid_count
        if valid_count < token_dict["tokens"].shape[1]:
            assert not torch.any(token_dict["token_mask"][batch_idx, valid_count:])


def test_build_padded_voxel_tokens_preserves_flatten_batch_order():
    """
    验证 batch-first token 恢复为展平序列后，顺序与 flatten 输出保持一致。

    输入:
        无，内部构造 3 个点云样本
    输出:
        无，通过断言验证结果
    """
    encoder = VoxelNetEncoder(
        voxel_size=(0.25, 0.25, 0.25),
        point_cloud_range=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        max_points_per_voxel=8,
        max_voxels=256,
        input_feature_dim=6,
        svfe_hidden_channels=16,
        svfe_out_channels=32,
        cml_channels=(32,),
    )
    points_xyz = torch.cat([
        _make_points_batch(4, scale=1.0),
        _make_points_batch(4, scale=0.8),
        _make_points_batch(4, scale=0.6),
    ], dim=0)
    outputs = encoder(points_xyz)
    flat_dict = flatten_voxel_grid_for_transformer(
        dense_voxel_feats=outputs["dense_voxel_feats"],
        valid_mask=outputs["valid_mask"],
        grid_meta=outputs["grid_meta"],
    )
    token_dict = build_padded_voxel_tokens(
        dense_voxel_feats=outputs["dense_voxel_feats"],
        valid_mask=outputs["valid_mask"],
        grid_meta=outputs["grid_meta"],
    )

    rebuilt_tokens = []
    rebuilt_coords = []
    for batch_idx in range(points_xyz.shape[0]):
        valid_count = int(token_dict["token_mask"][batch_idx].sum().item())
        rebuilt_tokens.append(token_dict["tokens"][batch_idx, :valid_count])
        rebuilt_coords.append(token_dict["token_pos"][batch_idx, :valid_count])

    rebuilt_tokens = torch.cat(rebuilt_tokens, dim=0)
    rebuilt_coords = torch.cat(rebuilt_coords, dim=0)
    assert torch.allclose(rebuilt_tokens, flat_dict["voxel_tokens"])
    assert torch.allclose(rebuilt_coords, flat_dict["voxel_coords"])
