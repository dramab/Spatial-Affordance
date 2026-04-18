"""
tests/test_bbox3d_head.py
-------------------------
职责：测试 3D BBox 回归头的输出约束与反向传播行为。

测试内容：
- test_bbox3d_head_outputs_positive_sizes：
  验证尺寸分量经过约束后始终为正
- test_bbox3d_head_supports_backward_without_inplace_error：
  验证回归头参与损失计算时可以正常反向传播

用法：
    pytest tests/test_bbox3d_head.py -v
"""

from __future__ import annotations

import torch

from src.models.heads import BBox3DHead


def test_bbox3d_head_outputs_positive_sizes():
    """
    作用：验证回归头输出的长宽高分量始终为正值。

    输入：
        无，内部构造随机 decoder token
    输出：
        无，通过断言验证结果
    """
    head = BBox3DHead(hidden_dim=32, num_layers=2, out_dim=7)
    decoder_tokens = torch.randn((2, 1, 32), dtype=torch.float32)

    pred_boxes = head(decoder_tokens)

    assert pred_boxes.shape == (2, 1, 7)
    assert torch.all(pred_boxes[..., 3:6] > 0)


def test_bbox3d_head_supports_backward_without_inplace_error():
    """
    作用：验证回归头不会因 inplace 操作破坏 autograd 反向传播。

    输入：
        无，内部构造需要梯度的 decoder token
    输出：
        无，通过断言验证结果
    """
    head = BBox3DHead(hidden_dim=32, num_layers=2, out_dim=7)
    decoder_tokens = torch.randn((2, 1, 32), dtype=torch.float32, requires_grad=True)
    target_boxes = torch.randn((2, 1, 7), dtype=torch.float32)

    pred_boxes = head(decoder_tokens)
    loss = torch.nn.functional.smooth_l1_loss(pred_boxes, target_boxes)
    loss.backward()

    assert decoder_tokens.grad is not None
    assert torch.isfinite(decoder_tokens.grad).all()
