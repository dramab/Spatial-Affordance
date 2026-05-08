"""
tests/test_size_yaw_head.py
---------------------------
职责：测试 size/yaw 回归头的输出语义与反向传播行为。

测试内容：
- test_size_yaw_head_outputs_raw_log_size_channels：
  验证尺寸分量直接输出 log(size_norm) 回归值
- test_size_yaw_head_supports_backward_without_inplace_error：
  验证回归头参与损失计算时可以正常反向传播

用法：
    pytest tests/test_size_yaw_head.py -v
"""

from __future__ import annotations

import torch

from src.models.heads import SizeYawHead


def test_size_yaw_head_outputs_raw_log_size_channels():
    """
    作用：验证回归头不会对 log(size_norm) 通道施加正值约束。

    输入：
        无，内部构造融合后的双 query token
    输出：
        无，通过断言验证结果
    """
    head = SizeYawHead(input_dim=8, hidden_dim=4, num_layers=1, out_dim=4)
    fused_tokens = torch.zeros((2, 8), dtype=torch.float32)
    expected_size_yaw = torch.tensor([-2.0, -1.0, 0.5, 0.25], dtype=torch.float32)
    with torch.no_grad():
        head.output.weight.zero_()
        head.output.bias.copy_(expected_size_yaw)

    pred_size_yaw = head(fused_tokens)

    assert pred_size_yaw.shape == (2, 4)
    torch.testing.assert_close(pred_size_yaw, expected_size_yaw.view(1, 4).expand(2, 4))


def test_size_yaw_head_supports_backward_without_inplace_error():
    """
    作用：验证回归头不会因 inplace 操作破坏 autograd 反向传播。

    输入：
        无，内部构造需要梯度的融合 query token
    输出：
        无，通过断言验证结果
    """
    head = SizeYawHead(input_dim=64, hidden_dim=32, num_layers=2, out_dim=4)
    fused_tokens = torch.randn((2, 64), dtype=torch.float32, requires_grad=True)
    target_size_yaw = torch.randn((2, 4), dtype=torch.float32)

    pred_size_yaw = head(fused_tokens)
    loss = torch.nn.functional.smooth_l1_loss(pred_size_yaw, target_size_yaw)
    loss.backward()

    assert fused_tokens.grad is not None
    assert torch.isfinite(fused_tokens.grad).all()
