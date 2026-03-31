"""
src/models/backbones/image_backbone.py
---------------------------------------
职责：RGB 图像特征提取 backbone。

支持：
- ResNet（resnet50 / resnet101）：使用 timm 加载，输出 FPN 特征
- ViT（vit_base / vit_large）：使用 timm 加载，输出 patch token 特征

输入：
    x: Tensor(B, 3, H, W)  # 归一化后的 RGB 图像

输出：
    feats: Tensor(B, C, H', W')  # 下采样后的特征图（C=out_channels）

用法：
    backbone = ImageBackbone(cfg.image_backbone)
    feats = backbone(image)
"""
