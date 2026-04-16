"""
src/models/multimodal_model.py
------------------------------
职责：构建点云、图像、文本共享 encoder 的多模态 3D BBox 预测模型。

用法：
    from src.models.multimodal_model import MultimodalModel

    model = MultimodalModel(model_cfg)
    outputs = model(
        points_xyz=points_xyz,
        images=images,
        text_inputs=["place the cup near the chair"],
    )
    pred_boxes = outputs["pred_boxes"]
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn

from src.models.backbones import (
    ImageBackbone,
    PCBackbone,
    build_padded_voxel_tokens,
)
from src.models.encoders import TextEncoder
from src.models.fusion import MultimodalDecoder, UnifiedMultimodalEncoder
from src.models.heads import BBox3DHead


def _cfg_get(cfg: Mapping[str, Any] | object, key: str, default: Any = None) -> Any:
    """
    作用：从 dict 或对象中统一读取配置。

    输入：
        cfg: 配置对象
        key: str 配置键
        default: 默认值
    输出：
        配置值
    """
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class MultimodalModel(nn.Module):
    """
    作用：将三种模态编码为统一 memory，并输出 7D 3D BBox。

    输入：
        points_xyz: Tensor(B, N, 3) 点云坐标，可为 None
        point_feats: Tensor(B, N, F) 点特征，可为 None
        images: Tensor(B, 3, H, W) 图像，可为 None
        text_inputs: list[str] 或 tokenizer 输出字典，可为 None
    输出：
        dict，包含编码结果、decoder 输出以及 pred_boxes
    """

    def __init__(self, cfg: Mapping[str, Any] | object):
        super().__init__()
        self.cfg = cfg

        image_backbone_cfg = _cfg_get(cfg, "image_backbone", {})
        pc_backbone_cfg = _cfg_get(cfg, "pc_backbone", {})
        text_encoder_cfg = _cfg_get(cfg, "text_encoder", {})
        fusion_cfg = _cfg_get(cfg, "fusion", {})
        decoder_cfg = _cfg_get(cfg, "decoder", {})
        bbox3d_head_cfg = _cfg_get(cfg, "bbox3d_head", {})

        self.image_backbone = ImageBackbone(image_backbone_cfg)
        self.pc_backbone = PCBackbone(pc_backbone_cfg)
        self.text_encoder = TextEncoder(text_encoder_cfg)

        hidden_dim = int(_cfg_get(fusion_cfg, "hidden_dim", 256))
        self.multimodal_encoder = UnifiedMultimodalEncoder(
            point_dim=int(self.pc_backbone.out_channels),
            image_dim=int(self.image_backbone.out_channels),
            text_dim=int(self.text_encoder.out_channels),
            hidden_dim=hidden_dim,
            num_layers=int(_cfg_get(fusion_cfg, "num_layers", 3)),
            num_heads=int(_cfg_get(fusion_cfg, "num_heads", 8)),
            dropout=float(_cfg_get(fusion_cfg, "dropout", 0.1)),
        )
        self.decoder = MultimodalDecoder(
            hidden_dim=hidden_dim,
            num_layers=int(_cfg_get(decoder_cfg, "num_layers", 3)),
            num_heads=int(_cfg_get(decoder_cfg, "num_heads", 8)),
            dropout=float(_cfg_get(decoder_cfg, "dropout", 0.1)),
            num_queries=int(_cfg_get(decoder_cfg, "num_queries", 1)),
        )
        self.bbox3d_head = BBox3DHead(
            hidden_dim=int(_cfg_get(bbox3d_head_cfg, "hidden_dim", hidden_dim)),
            num_layers=int(_cfg_get(bbox3d_head_cfg, "num_layers", 2)),
            out_dim=int(_cfg_get(bbox3d_head_cfg, "out_dim", 7)),
        )

    def _encode_point_inputs(
            self,
            points_xyz: Optional[torch.Tensor],
            point_feats: Optional[torch.Tensor]) -> Optional[dict[str, torch.Tensor]]:
        """
        作用：将点云输入转换为统一 encoder 可消费的 batch-first token。

        输入：
            points_xyz: Tensor(B, N, 3) 点云坐标
            point_feats: Tensor(B, N, F) 点特征
        输出：
            dict 或 None，包含 tokens、token_mask、token_pos
        """
        if points_xyz is None:
            return None
        point_outputs = self.pc_backbone(points_xyz, point_feats)
        token_dict = build_padded_voxel_tokens(
            dense_voxel_feats=point_outputs["dense_voxel_feats"],
            valid_mask=point_outputs["valid_mask"],
            grid_meta=point_outputs["grid_meta"],
        )
        return {
            "tokens": token_dict["tokens"],
            "token_mask": token_dict["token_mask"],
            "token_pos": token_dict["token_pos"],
            "token_coords_cm": token_dict["token_coords_cm"],
            "sparse_coords": token_dict["sparse_coords"],
            "token_counts": token_dict["token_counts"],
        }

    def _encode_image_inputs(self, images: Optional[torch.Tensor]) -> Optional[dict[str, torch.Tensor]]:
        """
        作用：整理图像模态的 token 输出。

        输入：
            images: Tensor(B, 3, H, W) 图像
        输出：
            dict 或 None，包含 tokens、token_mask、token_pos
        """
        if images is None:
            return None
        return self.image_backbone(images)

    def _encode_text_inputs(
            self,
            text_inputs: Optional[Sequence[str] | Mapping[str, torch.Tensor]]) -> Optional[dict[str, torch.Tensor]]:
        """
        作用：整理文本模态的 token 输出。

        输入：
            text_inputs: list[str] 或 tokenizer 输出字典
        输出：
            dict 或 None，包含 tokens、token_mask
        """
        if text_inputs is None:
            return None
        return self.text_encoder(text_inputs)

    def forward(
            self,
            points_xyz: Optional[torch.Tensor] = None,
            point_feats: Optional[torch.Tensor] = None,
            images: Optional[torch.Tensor] = None,
            text_inputs: Optional[Sequence[str] | Mapping[str, torch.Tensor]] = None) -> dict[str, Any]:
        """
        作用：执行统一多模态编码、解码与 3D BBox 回归。

        输入：
            points_xyz: Tensor(B, N, 3) 点云坐标
            point_feats: Tensor(B, N, F) 点特征
            images: Tensor(B, 3, H, W) RGB 图像
            text_inputs: list[str] 或 tokenizer 输出字典
        输出：
            dict，包含 point/image/text 编码结果、memory、decoder_tokens、pred_boxes
        """
        point_dict = self._encode_point_inputs(points_xyz, point_feats)
        image_dict = self._encode_image_inputs(images)
        text_dict = self._encode_text_inputs(text_inputs)

        memory_dict = self.multimodal_encoder(
            point_inputs=point_dict,
            image_inputs=image_dict,
            text_inputs=text_dict,
        )
        decoder_dict = self.decoder(
            memory=memory_dict["memory"],
            memory_mask=memory_dict["memory_mask"],
        )
        pred_boxes = self.bbox3d_head(decoder_dict["decoder_tokens"])

        return {
            "point_outputs": point_dict,
            "image_outputs": image_dict,
            "text_outputs": text_dict,
            "memory": memory_dict["memory"],
            "memory_mask": memory_dict["memory_mask"],
            "memory_pos": memory_dict["memory_pos"],
            "modality_lengths": memory_dict["modality_lengths"],
            "decoder_tokens": decoder_dict["decoder_tokens"],
            "query_embed": decoder_dict["query_embed"],
            "pred_boxes": pred_boxes,
        }
