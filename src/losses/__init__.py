"""
src/losses/__init__.py
----------------------
职责：导出训练时使用的损失函数接口。
"""

from src.losses.multimodal_bbox_loss import MultimodalBBoxLoss

__all__ = ["MultimodalBBoxLoss"]
