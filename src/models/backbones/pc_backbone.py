"""
src/models/backbones/pc_backbone.py
-------------------------------------
职责：点云特征提取 backbone。

支持：
- PointNet++（MSG）：多尺度分组，逐层下采样并聚合局部特征
- PointTransformer：基于 self-attention 的点云特征提取

输入：
    xyz:    Tensor(B, N, 3)  # 点云坐标
    feats:  Tensor(B, N, 3)  # 点云颜色（RGB，可选）

输出：
    pc_feats: Tensor(B, M, C)  # 下采样后的点特征（M=num_points）
    pc_xyz:   Tensor(B, M, 3)  # 对应的点坐标

用法：
    backbone = PCBackbone(cfg.pc_backbone)
    pc_feats, pc_xyz = backbone(xyz, color)
"""
