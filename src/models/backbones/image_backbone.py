"""
src/models/backbones/image_backbone.py
--------------------------------------
职责：基于 ResNet 的 RGB 图像编码器，并输出适配 Transformer 的 token 接口。

用法：
    from src.models.backbones.image_backbone import ImageBackbone

    backbone = ImageBackbone({
        "type": "resnet50",
        "pretrained": False,
        "out_channels": 256,
    })
    outputs = backbone(images)
    image_tokens = outputs["tokens"]
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    ResNet101_Weights,
)
from src.models.common import cfg_get


class ImageBackbone(nn.Module):
    """
    作用：将 RGB 图像编码为卷积特征图与 Transformer token。

    输入：
        images: Tensor(B, 3, H, W)，支持 uint8 或 float
    输出：
        dict，包含：
        - tokens: Tensor(B, H'*W', C)
        - token_mask: BoolTensor(B, H'*W')
        - token_pos: Tensor(B, H'*W', 2)，归一化二维坐标
    """

    def __init__(self, cfg: Mapping[str, Any] | object):
        super().__init__()
        backbone_type = str(cfg_get(cfg, "type", "resnet50")).lower()
        pretrained = bool(cfg_get(cfg, "pretrained", True))
        freeze = bool(cfg_get(cfg, "freeze", False))
        out_channels = int(cfg_get(cfg, "out_channels", 256))

        backbone, backbone_out_channels = self._build_backbone(
            backbone_type=backbone_type,
            pretrained=pretrained,
        )
        self.backbone = backbone
        self.out_channels = out_channels
        self.proj = nn.Conv2d(backbone_out_channels, out_channels, kernel_size=1)

        pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        self.register_buffer("pixel_mean", pixel_mean.reshape(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", pixel_std.reshape(1, 3, 1, 1), persistent=False)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _build_backbone(
            self,
            backbone_type: str,
            pretrained: bool) -> tuple[nn.Module, int]:
        """
        作用：构造 ResNet 主干网络。

        输入：
            backbone_type: str 主干类型
            pretrained: bool 是否加载预训练权重
        输出：
            (nn.Module, int)，分别为主干网络与输出通道数
        """
        if backbone_type == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            out_channels = 2048
        elif backbone_type == "resnet101":
            weights = ResNet101_Weights.DEFAULT if pretrained else None
            model = models.resnet101(weights=weights)
            out_channels = 2048
        else:
            raise ValueError(f"unsupported image backbone type: {backbone_type}")

        backbone = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        return backbone, out_channels

    def _normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        作用：将输入图像转换为 float 并做 ImageNet 归一化。

        输入：
            images: Tensor(B, 3, H, W)
        输出：
            Tensor(B, 3, H, W)，归一化后的图像
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (B, 3, H, W)")

        images = images.to(torch.float32)
        if images.max().item() > 1.0:
            images = images / 255.0
        return (images - self.pixel_mean) / self.pixel_std

    def _build_2d_token_pos(
            self,
            batch_size: int,
            feat_h: int,
            feat_w: int,
            device: torch.device) -> torch.Tensor:
        """
        作用：构造二维特征图展平后的 token 坐标。

        输入：
            batch_size: int batch 大小
            feat_h: int 特征图高度
            feat_w: int 特征图宽度
            device: torch.device 输出设备
        输出：
            Tensor(B, feat_h*feat_w, 2)，坐标范围为 [-1, 1]
        """
        ys = torch.linspace(-1.0, 1.0, steps=feat_h, device=device)
        xs = torch.linspace(-1.0, 1.0, steps=feat_w, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        pos = torch.stack([grid_x, grid_y], dim=-1).view(1, feat_h * feat_w, 2)
        return pos.expand(batch_size, -1, -1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        作用：执行图像编码并导出 Transformer 友好的 token。

        输入：
            images: Tensor(B, 3, H, W)
        输出：
            dict，包含 tokens、token_mask、token_pos
        """
        images = self._normalize_images(images)
        feature_map = self.proj(self.backbone(images))
        batch_size, _, feat_h, feat_w = feature_map.shape

        tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
        token_mask = torch.ones(
            (batch_size, feat_h * feat_w),
            dtype=torch.bool,
            device=feature_map.device,
        )
        token_pos = self._build_2d_token_pos(
            batch_size=batch_size,
            feat_h=feat_h,
            feat_w=feat_w,
            device=feature_map.device,
        )

        return {
            "tokens": tokens,
            "token_mask": token_mask,
            "token_pos": token_pos,
        }
