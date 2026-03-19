"""
tests/test_losses.py
---------------------
职责：测试损失函数计算的正确性。

测试内容：
- test_bbox3d_loss_zero：预测等于 GT 时，损失应接近 0
- test_bbox3d_loss_yaw：验证 yaw 损失对角度不连续性的处理
- test_giou3d_range：验证 3D GIoU 值在 [-1, 1] 范围内
- test_grounding_loss：验证整体损失正常反向传播（梯度不为 None）
- test_hungarian_matching：验证 Hungarian 匹配结果的正确性

用法：
    pytest tests/test_losses.py -v
"""
