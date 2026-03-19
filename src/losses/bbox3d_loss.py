"""
src/losses/bbox3d_loss.py
--------------------------
职责：3D bounding box 回归损失计算。

损失组成：
- L1 Loss：对 cx, cy, cz, l, w, h 的绝对误差
- Yaw Loss：对 yaw 角使用 sin/cos 双分支，避免角度不连续
  loss_yaw = |sin(pred) - sin(gt)| + |cos(pred) - cos(gt)|
- 3D GIoU Loss：基于 3D IoU 的广义 IoU 损失（需 box_utils 支持）

输入：
    pred_boxes: Tensor(B, 7)  # 预测 bbox
    gt_boxes:   Tensor(B, 7)  # GT bbox

输出：
    loss_dict: Dict[str, Tensor]  # {"l1": ..., "yaw": ..., "giou3d": ...}

用法：
    criterion = BBox3DLoss(cfg.loss_weights)
    loss_dict = criterion(pred_boxes, gt_boxes)
"""
