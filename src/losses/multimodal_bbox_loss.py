"""
src/losses/multimodal_bbox_loss.py
----------------------------------
职责：定义多模态 3D BBox 与移动前物体中心训练使用的 normalized 回归损失。

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
        pred_object_centers_norm=outputs["pred_object_centers_norm"],
        target_object_centers_norm=batch["object_centers_norm"],
    )
    loss = loss_dict["loss"]
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _squeeze_single_query_tensor(
        tensor: torch.Tensor,
        tensor_name: str,
        feature_dim: int) -> torch.Tensor:
    """
    用法: values = _squeeze_single_query_tensor(pred, "pred", 3)
    作用: 将 (B, 1, C) 单 query 张量整理为 (B, C)
    输入: tensor: Tensor，形状为 (B,C) 或 (B,1,C)；tensor_name: str；feature_dim: int
    输出: Tensor(B,C)，整理后的张量
    """
    if tensor.ndim == 2 and tensor.shape[-1] == int(feature_dim):
        return tensor.to(torch.float32)
    if tensor.ndim == 3 and tensor.shape[1] == 1 and tensor.shape[2] == int(feature_dim):
        return tensor[:, 0, :].to(torch.float32)
    raise ValueError(
        f"{tensor_name} must have shape (B, {feature_dim}) or (B, 1, {feature_dim}), "
        f"got {tuple(tensor.shape)}"
    )


def _squeeze_single_query_boxes(box_tensor: torch.Tensor, tensor_name: str) -> torch.Tensor:
    """
    用法: boxes = _squeeze_single_query_boxes(pred_boxes_norm, "pred_boxes_norm")
    作用: 将 (B, 1, 7) 单 query box 张量整理为 (B, 7)
    输入: box_tensor: Tensor，形状为 (B, 7) 或 (B, 1, 7)；tensor_name: str
    输出: Tensor(B, 7)，整理后的 box 张量
    """
    return _squeeze_single_query_tensor(box_tensor, tensor_name, 7)


def _squeeze_single_query_centers(center_tensor: torch.Tensor, tensor_name: str) -> torch.Tensor:
    """
    用法: centers = _squeeze_single_query_centers(pred_centers_norm, "pred_centers_norm")
    作用: 将 (B, 1, 3) 单 query center 张量整理为 (B, 3)
    输入: center_tensor: Tensor，形状为 (B, 3) 或 (B, 1, 3)；tensor_name: str
    输出: Tensor(B, 3)，整理后的 center 张量
    """
    return _squeeze_single_query_tensor(center_tensor, tensor_name, 3)


class MultimodalBBoxLoss(nn.Module):
    """
    作用：对 normalized 7D 3D box 和移动前物体中心计算加权 Smooth L1 损失。

    输入：
        pred_boxes_norm: Tensor(B, 7) 或 Tensor(B, 1, 7)，模型预测框
        target_boxes_norm: Tensor(B, 7) 或 Tensor(B, 1, 7)，监督框
        pred_object_centers_norm: Tensor(B, 3) 或 Tensor(B, 1, 3)，预测移动前物体中心
        target_object_centers_norm: Tensor(B, 3) 或 Tensor(B, 1, 3)，监督移动前物体中心
    输出：
        dict，包含总损失及 center/size/yaw/object_center 分项损失
    """

    def __init__(
            self,
            center_weight: float = 1.0,
            size_weight: float = 1.0,
            yaw_weight: float = 0.5,
            object_center_weight: float | None = None,
            smooth_l1_beta: float = 1.0):
        super().__init__()
        self.center_weight = float(center_weight)
        self.size_weight = float(size_weight)
        self.yaw_weight = float(yaw_weight)
        self.object_center_weight = (
            self.center_weight if object_center_weight is None else float(object_center_weight)
        )
        self.smooth_l1_beta = float(smooth_l1_beta)

        if (
            self.center_weight < 0 or
            self.size_weight < 0 or
            self.yaw_weight < 0 or
            self.object_center_weight < 0
        ):
            raise ValueError("loss weights must be non-negative")
        if self.smooth_l1_beta < 0:
            raise ValueError("smooth_l1_beta must be non-negative")

    def forward(
            self,
            pred_boxes_norm: torch.Tensor,
            target_boxes_norm: torch.Tensor,
            pred_object_centers_norm: torch.Tensor,
            target_object_centers_norm: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        用法: loss_dict = criterion(pred_boxes_norm, target_boxes_norm, pred_centers_norm, target_centers_norm)
        作用: 计算 normalized 7D 3D box 与移动前物体中心的加权回归损失
        输入: pred/target_boxes_norm: Tensor(B,7) 或 Tensor(B,1,7)；pred/target_object_centers_norm: Tensor(B,3) 或 Tensor(B,1,3)
        输出: dict，包含 loss、center_loss、size_loss、yaw_loss、object_center_loss
        """
        pred_boxes = _squeeze_single_query_boxes(pred_boxes_norm, "pred_boxes_norm")
        target_boxes = _squeeze_single_query_boxes(target_boxes_norm, "target_boxes_norm")
        pred_object_centers = _squeeze_single_query_centers(
            pred_object_centers_norm,
            "pred_object_centers_norm",
        )
        target_object_centers = _squeeze_single_query_centers(
            target_object_centers_norm,
            "target_object_centers_norm",
        )

        if pred_boxes.shape != target_boxes.shape:
            raise ValueError(
                "pred_boxes_norm and target_boxes_norm must share the same shape after squeeze, "
                f"got {tuple(pred_boxes.shape)} and {tuple(target_boxes.shape)}"
            )
        if pred_object_centers.shape != target_object_centers.shape:
            raise ValueError(
                "pred_object_centers_norm and target_object_centers_norm must share the same shape after squeeze, "
                f"got {tuple(pred_object_centers.shape)} and {tuple(target_object_centers.shape)}"
            )
        if pred_boxes.shape[0] != pred_object_centers.shape[0]:
            raise ValueError(
                "box and object center batch sizes must match, "
                f"got {pred_boxes.shape[0]} and {pred_object_centers.shape[0]}"
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
        object_center_loss = F.smooth_l1_loss(
            pred_object_centers,
            target_object_centers,
            beta=self.smooth_l1_beta,
            reduction="mean",
        )

        total_loss = (
            self.center_weight * center_loss +
            self.size_weight * size_loss +
            self.yaw_weight * yaw_loss +
            self.object_center_weight * object_center_loss
        )
        return {
            "loss": total_loss,
            "center_loss": center_loss,
            "size_loss": size_loss,
            "yaw_loss": yaw_loss,
            "object_center_loss": object_center_loss,
        }
