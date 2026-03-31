"""
src/losses/grounding_loss.py
-----------------------------
职责：整体 grounding 损失，含 Hungarian matching（多预测框场景）。

功能：
- 若模型输出单个 bbox，直接调用 BBox3DLoss
- 若模型输出多个候选框（DETR 风格），先用 Hungarian 算法匹配预测框与 GT，
  再对匹配对计算 BBox3DLoss
- 汇总各子损失，按权重加权求和，返回总损失和损失字典

输入：
    pred_boxes: Tensor(B, K, 7) 或 Tensor(B, 7)  # K 为候选框数
    gt_boxes:   Tensor(B, 7)

输出：
    total_loss: Tensor(scalar)
    loss_dict:  Dict[str, Tensor]

用法：
    criterion = GroundingLoss(cfg.loss_weights)
    total_loss, loss_dict = criterion(pred_boxes, gt_boxes)
"""
