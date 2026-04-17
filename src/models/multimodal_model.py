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
from src.models.common import cfg_get
from src.models.encoders import TextEncoder
from src.models.fusion import MultimodalDecoder, UnifiedMultimodalEncoder
from src.models.heads import BBox3DHead


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

        image_backbone_cfg = cfg_get(cfg, "image_backbone", {})
        pc_backbone_cfg = cfg_get(cfg, "pc_backbone", {})
        text_encoder_cfg = cfg_get(cfg, "text_encoder", {})
        fusion_cfg = cfg_get(cfg, "fusion", {})
        decoder_cfg = cfg_get(cfg, "decoder", {})
        bbox3d_head_cfg = cfg_get(cfg, "bbox3d_head", {})

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
        self.decoder = MultimodalDecoder(
            hidden_dim=hidden_dim,
            num_layers=int(cfg_get(decoder_cfg, "num_layers", 3)),
            num_heads=int(cfg_get(decoder_cfg, "num_heads", 8)),
            dropout=float(cfg_get(decoder_cfg, "dropout", 0.1)),
            num_queries=int(cfg_get(decoder_cfg, "num_queries", 1)),
        )
        self.bbox3d_head = BBox3DHead(
            hidden_dim=int(cfg_get(bbox3d_head_cfg, "hidden_dim", hidden_dim)),
            num_layers=int(cfg_get(bbox3d_head_cfg, "num_layers", 2)),
            out_dim=int(cfg_get(bbox3d_head_cfg, "out_dim", 7)),
        )

    @staticmethod
    def _denormalize_boxes(
            pred_boxes_norm: torch.Tensor,
            box_norm_meta: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """
        作用：根据场景中心和平移尺度恢复 camera_aligned 坐标系下的 box。

        输入：
            pred_boxes_norm: Tensor(B, Q, 7) normalized box
            box_norm_meta: dict，至少包含:
                - scene_center: Tensor(B, 3)
                - scene_scale: Tensor(B,) 或 Tensor(B, 1)
        输出：
            Tensor(B, Q, 7) 去归一化后的 camera_aligned box
        """
        if "scene_center" not in box_norm_meta or "scene_scale" not in box_norm_meta:
            raise ValueError("box_norm_meta must contain scene_center and scene_scale")

        scene_center = box_norm_meta["scene_center"].to(
            device=pred_boxes_norm.device,
            dtype=pred_boxes_norm.dtype,
        )
        scene_scale = box_norm_meta["scene_scale"].to(
            device=pred_boxes_norm.device,
            dtype=pred_boxes_norm.dtype,
        )
        if scene_center.ndim != 2 or scene_center.shape[-1] != 3:
            raise ValueError("scene_center must have shape (B, 3)")
        if scene_scale.ndim == 1:
            scene_scale = scene_scale.unsqueeze(-1)
        if scene_scale.ndim != 2 or scene_scale.shape[-1] != 1:
            raise ValueError("scene_scale must have shape (B,) or (B, 1)")
        if scene_center.shape[0] != pred_boxes_norm.shape[0] or scene_scale.shape[0] != pred_boxes_norm.shape[0]:
            raise ValueError("box_norm_meta batch size must match pred_boxes_norm")

        denormalized = pred_boxes_norm.clone()
        denormalized[..., :3] = denormalized[..., :3] * scene_scale.unsqueeze(1) + scene_center.unsqueeze(1)
        denormalized[..., 3:6] = denormalized[..., 3:6] * scene_scale.unsqueeze(1)
        return denormalized

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
        }

    def forward(
            self,
            points_xyz: Optional[torch.Tensor] = None,
            point_feats: Optional[torch.Tensor] = None,
            images: Optional[torch.Tensor] = None,
            text_inputs: Optional[Sequence[str] | Mapping[str, torch.Tensor]] = None,
            box_norm_meta: Optional[Mapping[str, torch.Tensor]] = None) -> dict[str, Any]:
        """
        作用：执行统一多模态编码、解码与 3D BBox 回归。

        输入：
            points_xyz: Tensor(B, N, 3) 点云坐标
            point_feats: Tensor(B, N, F) 点特征
            images: Tensor(B, 3, H, W) RGB 图像
            text_inputs: list[str] 或 tokenizer 输出字典
            box_norm_meta: dict，可选场景去归一化参数，至少包含
                scene_center 与 scene_scale
        输出：
            dict，包含 memory、memory_mask、decoder_tokens、
            pred_boxes_norm、pred_boxes 与 modality_lengths
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
        pred_boxes_norm = self.bbox3d_head(decoder_dict["decoder_tokens"])
        pred_boxes = (
            self._denormalize_boxes(pred_boxes_norm, box_norm_meta)
            if box_norm_meta is not None else pred_boxes_norm
        )

        return {
            "memory": memory_dict["memory"],
            "memory_mask": memory_dict["memory_mask"],
            "modality_lengths": memory_dict["modality_lengths"],
            "decoder_tokens": decoder_dict["decoder_tokens"],
            "pred_boxes_norm": pred_boxes_norm,
            "pred_boxes": pred_boxes,
        }
