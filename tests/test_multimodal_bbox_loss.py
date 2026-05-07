"""
tests/test_multimodal_bbox_loss.py
----------------------------------
职责：测试多模态 3D BBox 回归损失的形状整理与数值计算。

测试内容：
- test_multimodal_bbox_loss_returns_zero_for_identical_boxes：
  验证预测框、物体中心与目标一致时损失为 0
- test_multimodal_bbox_loss_matches_weighted_smooth_l1_sum：
  验证总损失与各分量损失满足加权求和关系
- test_multimodal_bbox_loss_rejects_multi_query_predictions：
  验证多于一个 query 的预测会触发显式报错

用法：
    pytest tests/test_multimodal_bbox_loss.py -v
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from src.losses import MultimodalBBoxLoss


def test_multimodal_bbox_loss_returns_zero_for_identical_boxes():
    """
    作用：验证预测框和移动前物体中心完全一致时，各项损失都为 0。

    输入：
        无，内部构造单 query normalized 7D box
    输出：
        无，通过断言验证结果
    """
    criterion = MultimodalBBoxLoss(
        center_weight=1.0,
        size_weight=1.0,
        yaw_weight=0.5,
        smooth_l1_beta=0.1,
    )
    pred_boxes = torch.tensor(
        [[[0.1, -0.2, 0.3, 0.4, 0.5, 0.6, -0.1]]],
        dtype=torch.float32,
    )
    target_boxes = pred_boxes.clone()
    pred_centers = torch.tensor([[[0.2, -0.1, 0.4]]], dtype=torch.float32)
    target_centers = pred_centers.clone()

    loss_dict = criterion(pred_boxes, target_boxes, pred_centers, target_centers)

    assert loss_dict["loss"].item() == 0.0
    assert loss_dict["center_loss"].item() == 0.0
    assert loss_dict["size_loss"].item() == 0.0
    assert loss_dict["yaw_loss"].item() == 0.0
    assert loss_dict["object_center_loss"].item() == 0.0


def test_multimodal_bbox_loss_matches_weighted_smooth_l1_sum():
    """
    作用：验证总损失等于中心、尺寸与 yaw Smooth L1 损失的加权和。

    输入：
        无，内部构造两组不同的 normalized 7D box
    输出：
        无，通过断言验证结果
    """
    criterion = MultimodalBBoxLoss(
        center_weight=2.0,
        size_weight=3.0,
        yaw_weight=4.0,
        object_center_weight=5.0,
        smooth_l1_beta=1.0,
    )
    pred_boxes = torch.tensor(
        [
            [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 0.2],
            [0.1, 0.6, 1.1, 1.6, 2.1, 2.6, 0.3],
        ],
        dtype=torch.float32,
    )
    target_boxes = torch.tensor(
        [
            [0.2, 0.0, 0.5, 1.0, 1.0, 2.0, -0.1],
            [0.3, 0.1, 0.9, 1.8, 2.4, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    pred_centers = torch.tensor(
        [
            [0.5, -0.5, 0.0],
            [0.1, 0.2, 0.3],
        ],
        dtype=torch.float32,
    )
    target_centers = torch.tensor(
        [
            [0.0, -0.2, 0.4],
            [0.3, 0.0, 0.1],
        ],
        dtype=torch.float32,
    )

    loss_dict = criterion(pred_boxes, target_boxes, pred_centers, target_centers)
    expected_center_loss = F.smooth_l1_loss(
        pred_boxes[:, 0:3],
        target_boxes[:, 0:3],
        beta=1.0,
        reduction="mean",
    )
    expected_size_loss = F.smooth_l1_loss(
        pred_boxes[:, 3:6],
        target_boxes[:, 3:6],
        beta=1.0,
        reduction="mean",
    )
    expected_yaw_loss = F.smooth_l1_loss(
        pred_boxes[:, 6:7],
        target_boxes[:, 6:7],
        beta=1.0,
        reduction="mean",
    )
    expected_object_center_loss = F.smooth_l1_loss(
        pred_centers,
        target_centers,
        beta=1.0,
        reduction="mean",
    )
    expected_total_loss = (
        2.0 * expected_center_loss +
        3.0 * expected_size_loss +
        4.0 * expected_yaw_loss +
        5.0 * expected_object_center_loss
    )

    torch.testing.assert_close(loss_dict["center_loss"], expected_center_loss)
    torch.testing.assert_close(loss_dict["size_loss"], expected_size_loss)
    torch.testing.assert_close(loss_dict["yaw_loss"], expected_yaw_loss)
    torch.testing.assert_close(loss_dict["object_center_loss"], expected_object_center_loss)
    torch.testing.assert_close(loss_dict["loss"], expected_total_loss)


def test_multimodal_bbox_loss_rejects_multi_query_predictions():
    """
    作用：验证多 query 预测不会被静默 squeeze，而是显式报错。

    输入：
        无，内部构造形状为 (B, 2, 7) 的预测框
    输出：
        无，通过断言验证结果
    """
    criterion = MultimodalBBoxLoss()
    pred_boxes = torch.zeros((2, 2, 7), dtype=torch.float32)
    target_boxes = torch.zeros((2, 7), dtype=torch.float32)
    pred_centers = torch.zeros((2, 1, 3), dtype=torch.float32)
    target_centers = torch.zeros((2, 3), dtype=torch.float32)

    try:
        criterion(pred_boxes, target_boxes, pred_centers, target_centers)
    except ValueError as exc:
        assert "must have shape (B, 7) or (B, 1, 7)" in str(exc)
    else:
        raise AssertionError("expected ValueError for multi-query predictions")
