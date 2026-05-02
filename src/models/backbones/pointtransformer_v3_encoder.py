"""
src/models/backbones/pointtransformer_v3_encoder.py
---------------------------------------------------
职责：将 PointTransformerV3 encoder-only 输出适配为项目统一点云 token。

用法：
    from src.models.backbones.pointtransformer_v3_encoder import PointTransformerV3Encoder

    encoder = PointTransformerV3Encoder({
        "grid_size": 0.05,
        "point_cloud_range": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        "enable_flash": True,
    })
    outputs = encoder(points_xyz)
    tokens = outputs["tokens"]
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional, Sequence

import torch
from torch import nn

from src.models.common import cfg_get


_PTV3_OPTIONAL_DEPS_MESSAGE = (
    "PointTransformerV3 backbone requires optional dependencies: addict, "
    "spconv, torch_scatter, and flash_attn when enable_flash=True. "
    "For the current spatial environment, install them with commands such as: "
    "pip install addict spconv-cu124 && "
    "pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html"
)


def _as_tuple(value: Sequence[Any] | Any, name: str) -> tuple[Any, ...]:
    """
    作用：将配置中的列表或元组标准化为 tuple。

    输入：
        value: 配置值
        name: str 配置名称，用于报错
    输出：
        tuple[Any, ...] 标准化后的配置值
    """
    if isinstance(value, (list, tuple)):
        return tuple(value)
    raise ValueError(f"{name} must be a sequence")


def _as_6float_tuple(value: Sequence[float], name: str) -> tuple[float, float, float, float, float, float]:
    """
    作用：校验并转换长度为 6 的空间范围配置。

    输入：
        value: Sequence[float] 空间范围
        name: str 配置名称，用于报错
    输出：
        tuple[float, float, float, float, float, float] 标准化空间范围
    """
    items = _as_tuple(value, name)
    if len(items) != 6:
        raise ValueError(f"{name} must contain exactly 6 values")
    return tuple(float(v) for v in items)


def _get_ptv3_kwargs(cfg: Mapping[str, Any] | object) -> dict[str, Any]:
    """
    作用：从项目配置中提取 PointTransformerV3 原生构造参数。

    输入：
        cfg: Mapping[str, Any] | object 点云 backbone 配置
    输出：
        dict[str, Any] 可传给 PointTransformerV3 的参数
    """
    sequence_keys = {
        "order",
        "stride",
        "enc_depths",
        "enc_channels",
        "enc_num_head",
        "enc_patch_size",
        "pdnorm_conditions",
    }
    passthrough_keys = [
        "in_channels",
        "order",
        "stride",
        "enc_depths",
        "enc_channels",
        "enc_num_head",
        "enc_patch_size",
        "mlp_ratio",
        "qkv_bias",
        "qk_scale",
        "attn_drop",
        "proj_drop",
        "drop_path",
        "pre_norm",
        "shuffle_orders",
        "enable_rpe",
        "enable_flash",
        "upcast_attention",
        "upcast_softmax",
        "pdnorm_bn",
        "pdnorm_ln",
        "pdnorm_decouple",
        "pdnorm_adaptive",
        "pdnorm_affine",
        "pdnorm_conditions",
    ]

    kwargs: dict[str, Any] = {"cls_mode": True}
    for key in passthrough_keys:
        value = cfg_get(cfg, key, None)
        if value is None:
            continue
        if key == "order" and isinstance(value, str):
            kwargs[key] = value
        else:
            kwargs[key] = _as_tuple(value, key) if key in sequence_keys else value
    return kwargs


def _load_point_transformer_v3():
    """
    作用：导入 vendored PointTransformerV3，并在依赖缺失时给出明确提示。

    输入：
        无
    输出：
        PointTransformerV3 类
    """
    try:
        from src.models.backbones.pointtransformer_v3.model import PointTransformerV3
    except ImportError as exc:
        raise ImportError(_PTV3_OPTIONAL_DEPS_MESSAGE) from exc
    return PointTransformerV3


class PointTransformerV3Encoder(nn.Module):
    """
    作用：封装 PointTransformerV3 encoder，并输出 batch-first 点云 token。

    输入：
        points_xyz: Tensor(B, N, 3) normalized 点云坐标
        point_feats: Tensor(B, N, F) 可选点特征
    输出：
        dict，包含 tokens、token_mask、token_pos
    """

    def __init__(self, cfg: Mapping[str, Any] | object):
        super().__init__()
        PointTransformerV3 = _load_point_transformer_v3()

        self.grid_size = float(cfg_get(cfg, "grid_size", cfg_get(cfg, "voxel_size", 0.05)))
        if self.grid_size <= 0.0:
            raise ValueError("grid_size must be positive")
        self.point_cloud_range = _as_6float_tuple(
            cfg_get(cfg, "point_cloud_range", (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)),
            "point_cloud_range",
        )

        ptv3_kwargs = _get_ptv3_kwargs(cfg)
        ptv3_kwargs["cls_mode"] = True
        self.in_channels = int(ptv3_kwargs.get("in_channels", 3))
        ptv3_kwargs["in_channels"] = self.in_channels
        try:
            self.model = PointTransformerV3(**ptv3_kwargs)
        except AssertionError as exc:
            if "flash_attn" in str(exc):
                raise ImportError(_PTV3_OPTIONAL_DEPS_MESSAGE) from exc
            raise

        enc_channels = tuple(int(v) for v in ptv3_kwargs.get("enc_channels", (32, 64, 128, 256, 512)))
        self.out_channels = int(enc_channels[-1])

    def _flatten_valid_points(
            self,
            points_xyz: torch.Tensor,
            point_feats: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        作用：过滤 padding 和越界点，并展开为 PTv3 所需的稀疏 batch 表示。

        输入：
            points_xyz: Tensor(B, N, 3) 点云坐标
            point_feats: Tensor(B, N, F) 可选点特征
        输出：
            tuple，依次为 flat_xyz、flat_feats、batch_indices
        """
        if points_xyz.ndim != 3 or points_xyz.shape[-1] != 3:
            raise ValueError("points_xyz must have shape (B, N, 3)")
        if point_feats is not None and point_feats.shape[:2] != points_xyz.shape[:2]:
            raise ValueError("point_feats must align with points_xyz on batch and point dims")

        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        finite_mask = torch.isfinite(points_xyz).all(dim=-1)
        range_mask = (
            (points_xyz[..., 0] >= x_min) & (points_xyz[..., 0] < x_max) &
            (points_xyz[..., 1] >= y_min) & (points_xyz[..., 1] < y_max) &
            (points_xyz[..., 2] >= z_min) & (points_xyz[..., 2] < z_max)
        )
        valid_mask = finite_mask & range_mask
        if point_feats is not None:
            valid_mask = valid_mask & torch.isfinite(point_feats).all(dim=-1)

        batch_indices, _ = valid_mask.nonzero(as_tuple=True)
        flat_xyz = points_xyz[valid_mask].to(torch.float32)
        if point_feats is None:
            flat_feats = flat_xyz
        else:
            flat_feats = torch.cat([flat_xyz, point_feats[valid_mask].to(torch.float32)], dim=-1)

        if int(flat_feats.shape[-1]) != self.in_channels:
            raise ValueError(
                f"PointTransformerV3 in_channels={self.in_channels}, "
                f"but input feature dim is {int(flat_feats.shape[-1])}"
            )
        return flat_xyz, flat_feats, batch_indices.to(torch.long)

    def _build_grid_coord(self, flat_xyz: torch.Tensor) -> torch.Tensor:
        """
        作用：将 normalized 坐标转换为 PTv3 serialization 使用的离散网格坐标。

        输入：
            flat_xyz: Tensor(M, 3) 有效点坐标
        输出：
            IntTensor(M, 3) 离散网格坐标
        """
        origin = flat_xyz.new_tensor(self.point_cloud_range[:3])
        grid_coord = torch.floor((flat_xyz - origin) / self.grid_size)
        return grid_coord.to(torch.int32)

    def _compact_batch_indices(self, batch_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        作用：移除空样本造成的 batch 编号空洞，避免 PTv3 内部出现 0 点 batch。

        输入：
            batch_indices: LongTensor(M,) 原始 batch 编号
        输出：
            tuple，依次为 compact_batch_indices、original_batch_lookup
        """
        original_batch_lookup = torch.unique(batch_indices, sorted=True)
        compact_batch_indices = torch.searchsorted(original_batch_lookup, batch_indices)
        return compact_batch_indices.to(torch.long), original_batch_lookup.to(torch.long)

    def _pad_encoder_tokens(
            self,
            point: MutableMapping[str, torch.Tensor],
            batch_size: int,
            device: torch.device) -> dict[str, torch.Tensor]:
        """
        作用：将 PTv3 稀疏点输出整理为 batch-first token。

        输入：
            point: PTv3 输出的 Point 字典
            batch_size: int batch 大小
            device: torch.device 输出设备
        输出：
            dict，包含 tokens、token_mask、token_pos
        """
        point_batch = point["batch"].to(torch.long)
        token_counts = torch.bincount(point_batch, minlength=batch_size).to(torch.long)
        max_tokens = int(token_counts.max().item()) if batch_size > 0 else 0
        tokens = point["feat"].new_zeros((batch_size, max_tokens, self.out_channels))
        token_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool, device=device)
        token_pos = point["coord"].new_zeros((batch_size, max_tokens, 3), dtype=torch.float32)

        if int(point["feat"].shape[0]) > 0 and max_tokens > 0:
            point_feats = point["feat"].to(torch.float32)
            point_coord = point["coord"].to(torch.float32)
            for batch_idx in range(batch_size):
                batch_mask = point_batch == batch_idx
                count = int(batch_mask.sum().item())
                if count == 0:
                    continue
                tokens[batch_idx, :count] = point_feats[batch_mask]
                token_mask[batch_idx, :count] = True
                token_pos[batch_idx, :count] = point_coord[batch_mask]

        return {
            "tokens": tokens,
            "token_mask": token_mask,
            "token_pos": token_pos,
        }

    def forward(
            self,
            points_xyz: torch.Tensor,
            point_feats: Optional[torch.Tensor] = None) -> MutableMapping[str, Any]:
        """
        作用：执行 PTv3 encoder-only 点云编码。

        输入：
            points_xyz: Tensor(B, N, 3) 点云坐标
            point_feats: Tensor(B, N, F) 可选点特征
        输出：
            dict，包含 tokens、token_mask、token_pos
        """
        batch_size = int(points_xyz.shape[0])
        device = points_xyz.device
        flat_xyz, flat_feats, batch_indices = self._flatten_valid_points(points_xyz, point_feats)
        if int(flat_xyz.shape[0]) == 0:
            return {
                "tokens": points_xyz.new_zeros((batch_size, 0, self.out_channels)),
                "token_mask": torch.zeros((batch_size, 0), dtype=torch.bool, device=device),
                "token_pos": points_xyz.new_zeros((batch_size, 0, 3), dtype=torch.float32),
            }

        compact_batch_indices, original_batch_lookup = self._compact_batch_indices(batch_indices)
        data_dict = {
            "coord": flat_xyz,
            "grid_coord": self._build_grid_coord(flat_xyz),
            "feat": flat_feats,
            "batch": compact_batch_indices,
            "grid_size": self.grid_size,
        }
        point = self.model(data_dict)
        point["batch"] = original_batch_lookup[point["batch"].to(torch.long)]
        return self._pad_encoder_tokens(point, batch_size=batch_size, device=device)
