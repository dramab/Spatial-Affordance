"""
src/models/heads/bbox3d_head.py
---------------------------------
职责：3D bounding box 回归头。

功能：
- 输入融合特征，输出 7-DoF 3D bbox 参数
- 对 yaw 角使用 sin/cos 双分支预测，避免角度不连续性
- 输出格式：(cx, cy, cz, l, w, h, yaw)

输入：
    fused_feats: Tensor(B, L, C)  # 融合后的特征（取 [CLS] token 或 mean pooling）

输出：
    pred_boxes: Tensor(B, 7)  # cx, cy, cz, l, w, h, yaw

用法：
    head = BBox3DHead(cfg.bbox3d_head)
    pred_boxes = head(fused_feats)
"""
