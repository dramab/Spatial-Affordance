"""
src/models/multimodal_model.py
------------------------------
职责：构建点云、图像、文本共享 encoder 的双 query 多模态 3D BBox 预测模型。

用法：
    from src.models.multimodal_model import MultimodalModel

    model = MultimodalModel(model_cfg)
    outputs = model(
        points_xyz=points_xyz,
        images=images,
        text_inputs=["place the cup near the chair"],
    )
    pred_boxes_norm = outputs["pred_boxes_norm"]
    pred_object_centers_norm = outputs["pred_object_centers_norm"]
    pred_placement_centers_norm = outputs["pred_placement_centers_norm"]
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
from src.models.common import cfg_get
from src.models.encoders import TextEncoder
from src.models.fusion import MultimodalDecoder, UnifiedMultimodalEncoder
from src.models.heads import Center3DHead, SizeYawHead


class MultimodalModel(nn.Module):
    """
    作用：将三种模态编码为统一 memory，并用 object/placement 双 query 输出 3D BBox 与目标物体中心。

    输入：
        points_xyz: Tensor(B, N, 3) 点云坐标，可为 None
        point_feats: Tensor(B, N, F) 点特征，可为 None
        images: Tensor(B, 3, H, W) 图像，可为 None
        text_inputs: list[str] 或 tokenizer 输出字典，可为 None
    输出：
        dict，包含编码结果、decoder 输出、pred_boxes_norm、pred_object_centers_norm 与 pred_placement_centers_norm
    """

    OBJECT_QUERY_INDEX = 0
    PLACEMENT_QUERY_INDEX = 1
    REQUIRED_NUM_QUERIES = 2

    def __init__(self, cfg: Mapping[str, Any] | object):
        """
        用法: model = MultimodalModel(model_cfg)
        作用: 初始化双 query 多模态 3D BBox 预测模型
        输入: cfg: Mapping 或配置对象，包含 backbone、fusion、decoder 与 head 配置
        输出: MultimodalModel 实例
        """
        super().__init__()

        image_backbone_cfg = cfg_get(cfg, "image_backbone", {})
        pc_backbone_cfg = cfg_get(cfg, "pc_backbone", {})
        text_encoder_cfg = cfg_get(cfg, "text_encoder", {})
        fusion_cfg = cfg_get(cfg, "fusion", {})
        decoder_cfg = cfg_get(cfg, "decoder", {})
        object_center_head_cfg = cfg_get(cfg, "object_center_head", {})
        placement_center_head_cfg = cfg_get(cfg, "placement_center_head", {})
        size_yaw_head_cfg = cfg_get(cfg, "size_yaw_head", {})

        self.image_backbone = ImageBackbone(image_backbone_cfg)
        self.pc_backbone = PCBackbone(pc_backbone_cfg)
        self.text_encoder = TextEncoder(text_encoder_cfg)

        hidden_dim = int(cfg_get(fusion_cfg, "hidden_dim", 256))
        self.multimodal_encoder = UnifiedMultimodalEncoder(
            point_dim=int(self.pc_backbone.out_channels),
            image_dim=int(self.image_backbone.out_channels),
            text_dim=int(self.text_encoder.out_channels),
            hidden_dim=hidden_dim,
            num_layers=int(cfg_get(fusion_cfg, "num_layers", 3)),
            num_heads=int(cfg_get(fusion_cfg, "num_heads", 8)),
            dropout=float(cfg_get(fusion_cfg, "dropout", 0.1)),
        )
        num_queries = int(cfg_get(decoder_cfg, "num_queries", self.REQUIRED_NUM_QUERIES))
        if num_queries != self.REQUIRED_NUM_QUERIES:
            raise ValueError(
                "MultimodalModel requires decoder.num_queries=2: "
                "query 0 is object, query 1 is placement"
            )

        self.decoder = MultimodalDecoder(
            hidden_dim=hidden_dim,
            num_layers=int(cfg_get(decoder_cfg, "num_layers", 3)),
            num_heads=int(cfg_get(decoder_cfg, "num_heads", 8)),
            dropout=float(cfg_get(decoder_cfg, "dropout", 0.1)),
            num_queries=num_queries,
        )
        self.object_center_head = Center3DHead(
            hidden_dim=int(cfg_get(object_center_head_cfg, "hidden_dim", hidden_dim)),
            num_layers=int(cfg_get(object_center_head_cfg, "num_layers", 2)),
            out_dim=int(cfg_get(object_center_head_cfg, "out_dim", 3)),
        )
        self.placement_center_head = Center3DHead(
            hidden_dim=int(cfg_get(placement_center_head_cfg, "hidden_dim", hidden_dim)),
            num_layers=int(cfg_get(placement_center_head_cfg, "num_layers", 2)),
            out_dim=int(cfg_get(placement_center_head_cfg, "out_dim", 3)),
        )
        self.size_yaw_head = SizeYawHead(
            input_dim=hidden_dim * 2,
            hidden_dim=int(cfg_get(size_yaw_head_cfg, "hidden_dim", hidden_dim)),
            num_layers=int(cfg_get(size_yaw_head_cfg, "num_layers", 2)),
            out_dim=int(cfg_get(size_yaw_head_cfg, "out_dim", 4)),
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
        if {"tokens", "token_mask", "token_pos"}.issubset(point_outputs.keys()):
            return {
                "tokens": point_outputs["tokens"],
                "token_mask": point_outputs["token_mask"],
                "token_pos": point_outputs["token_pos"],
            }
        token_dict = build_padded_voxel_tokens(
            dense_voxel_feats=point_outputs["dense_voxel_feats"],
            valid_mask=point_outputs["valid_mask"],
            grid_meta=point_outputs["grid_meta"],
        )
        return {
            "tokens": token_dict["tokens"],
            "token_mask": token_dict["token_mask"],
            "token_pos": token_dict["token_pos"],
        }

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
            dict，包含 memory、memory_mask、decoder_tokens、
            pred_boxes_norm、pred_object_centers_norm、pred_placement_centers_norm、
            pred_size_yaw_norm 与 modality_lengths
        """
        point_dict = self._encode_point_inputs(points_xyz, point_feats)
        image_dict = None if images is None else self.image_backbone(images)
        text_dict = None if text_inputs is None else self.text_encoder(text_inputs)

        memory_dict = self.multimodal_encoder(
            point_inputs=point_dict,
            image_inputs=image_dict,
            text_inputs=text_dict,
        )
        decoder_dict = self.decoder(
            memory=memory_dict["memory"],
            memory_mask=memory_dict["memory_mask"],
        )
        decoder_tokens = decoder_dict["decoder_tokens"]
        object_query = decoder_tokens[:, self.OBJECT_QUERY_INDEX, :]
        placement_query = decoder_tokens[:, self.PLACEMENT_QUERY_INDEX, :]

        pred_object_centers_norm = self.object_center_head(object_query)
        pred_placement_centers_norm = self.placement_center_head(placement_query)
        fused_size_yaw_tokens = torch.cat([object_query, placement_query], dim=-1)
        pred_size_yaw_norm = self.size_yaw_head(fused_size_yaw_tokens)
        pred_boxes_norm = torch.cat([pred_placement_centers_norm, pred_size_yaw_norm], dim=-1)

        return {
            "memory": memory_dict["memory"],
            "memory_mask": memory_dict["memory_mask"],
            "modality_lengths": memory_dict["modality_lengths"],
            "decoder_tokens": decoder_tokens,
            "pred_boxes_norm": pred_boxes_norm,
            "pred_object_centers_norm": pred_object_centers_norm,
            "pred_placement_centers_norm": pred_placement_centers_norm,
            "pred_size_yaw_norm": pred_size_yaw_norm,
        }
