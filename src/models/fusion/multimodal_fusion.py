"""
src/models/fusion/multimodal_fusion.py
----------------------------------------
职责：三模态（图像、点云、文本）cross-attention 融合模块。

策略：
- 文本特征作为 Query
- 图像特征和点云特征分别作为 Key/Value（或拼接后作为 Key/Value）
- 多层 Transformer decoder 结构，逐层更新 query
- 输出融合后的特征，用于 bbox 回归

输入：
    img_feats:  Tensor(B, H'*W', C)  # 展平的图像特征
    pc_feats:   Tensor(B, M, C)      # 点云特征
    text_feats: Tensor(B, L, C)      # 文本特征（作为 query）
    text_mask:  Tensor(B, L)         # 文本 padding mask

输出：
    fused_feats: Tensor(B, L, C)  # 融合后的特征

用法：
    fusion = MultimodalFusion(cfg.fusion)
    fused = fusion(img_feats, pc_feats, text_feats, text_mask)
"""
