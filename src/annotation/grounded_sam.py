"""
src/annotation/grounded_sam.py
--------------------------------
职责：使用 Grounded-DINO + SAM2 获取文字对应的 2D mask。

流程：
    文字 prompt + RGB 图像
        -> Grounded-DINO  -> 2D bbox（含置信度）
        -> SAM2           -> 精细 2D mask

输入：
    image: ndarray(H, W, 3)  # BGR 格式
    text:  str               # 文字描述

输出：
    mask:       ndarray(H, W)  # 二值 mask
    bbox2d:     ndarray(4,)    # [x1, y1, x2, y2]
    confidence: float          # Grounded-DINO 置信度

用法：
    detector = GroundedSAM(cfg)
    mask, bbox2d, conf = detector.predict(image, "the brown wooden chair")
"""
