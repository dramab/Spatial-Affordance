"""
src/losses/multimodal_bbox_loss.py
----------------------------------
职责：定义多模态 3D BBox 训练使用的 normalized 7D 回归损失，尺寸项采用 log(size_norm)。

用法：
    from src.losses import MultimodalBBoxLoss

    criterion = MultimodalBBoxLoss(
        center_weight=1.0,
        size_weight=1.0,
        yaw_weight=0.5,
    )
    loss_dict = criterion(
        pred_boxes_norm=outputs["pred_boxes_norm"],
        target_boxes_norm=batch["target_boxes_norm"],
    )
    loss = loss_dict["loss"]
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _squeeze_single_query_boxes(box_tensor: torch.Tensor, tensor_name: str) -> torch.Tensor:
    """
    用法: boxes = _squeeze_single_query_boxes(pred_boxes_norm, "pred_boxes_norm")
    作用: 将 (B, 1, 7) 单 query box 张量整理为 (B, 7)
    输入: box_tensor: Tensor，形状为 (B, 7) 或 (B, 1, 7)；tensor_name: str，报错时使用的张量名
    输出: Tensor(B, 7)，整理后的 box 张量
    """
    if box_tensor.ndim == 2 and box_tensor.shape[-1] == 7:
        return box_tensor.to(torch.float32)
    if box_tensor.ndim == 3 and box_tensor.shape[1] == 1 and box_tensor.shape[2] == 7:
        return box_tensor[:, 0, :].to(torch.float32)
    raise ValueError(
        f"{tensor_name} must have shape (B, 7) or (B, 1, 7), got {tuple(box_tensor.shape)}"
    )


class MultimodalBBoxLoss(nn.Module):
    """
    作用：对 normalized 7D 3D box 计算中心、log 尺寸与偏航角的加权 Smooth L1 损失。

    输入：
        pred_boxes_norm: Tensor(B, 7) 或 Tensor(B, 1, 7)，模型预测框
        target_boxes_norm: Tensor(B, 7) 或 Tensor(B, 1, 7)，监督框
    输出：
        dict，包含总损失 `loss` 以及 `center_loss`、`size_loss`、`yaw_loss`
    """

    def __init__(
            self,
            center_weight: float = 1.0,
            size_weight: float = 1.0,
            yaw_weight: float = 0.5,
            smooth_l1_beta: float = 1.0):
        super().__init__()
        self.center_weight = float(center_weight)
        self.size_weight = float(size_weight)
        self.yaw_weight = float(yaw_weight)
        self.smooth_l1_beta = float(smooth_l1_beta)

        if self.center_weight < 0 or self.size_weight < 0 or self.yaw_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if self.smooth_l1_beta < 0:
            raise ValueError("smooth_l1_beta must be non-negative")

    def forward(
            self,
            pred_boxes_norm: torch.Tensor,
            target_boxes_norm: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        用法: loss_dict = criterion(pred_boxes_norm, target_boxes_norm)
        作用: 计算 normalized 7D 3D box 的加权回归损失，尺寸通道语义为 log(size_norm)
        输入: pred_boxes_norm: Tensor(B, 7) 或 Tensor(B, 1, 7)；target_boxes_norm: Tensor(B, 7) 或 Tensor(B, 1, 7)
        输出: dict，包含 loss、center_loss、size_loss、yaw_loss
        """
        pred_boxes = _squeeze_single_query_boxes(pred_boxes_norm, "pred_boxes_norm")
        target_boxes = _squeeze_single_query_boxes(target_boxes_norm, "target_boxes_norm")

        if pred_boxes.shape != target_boxes.shape:
            raise ValueError(
                "pred_boxes_norm and target_boxes_norm must share the same shape after squeeze, "
                f"got {tuple(pred_boxes.shape)} and {tuple(target_boxes.shape)}"
            )

        center_loss = F.smooth_l1_loss(
            pred_boxes[:, 0:3],
            target_boxes[:, 0:3],
            beta=self.smooth_l1_beta,
            reduction="mean",
        )
        size_loss = F.smooth_l1_loss(
            pred_boxes[:, 3:6],
            target_boxes[:, 3:6],
            beta=self.smooth_l1_beta,
            reduction="mean",
        )
        yaw_loss = F.smooth_l1_loss(
            pred_boxes[:, 6:7],
            target_boxes[:, 6:7],
            beta=self.smooth_l1_beta,
            reduction="mean",
        )

        total_loss = (
            self.center_weight * center_loss +
            self.size_weight * size_loss +
            self.yaw_weight * yaw_loss
        )
        return {
            "loss": total_loss,
            "center_loss": center_loss,
            "size_loss": size_loss,
            "yaw_loss": yaw_loss,
        }
