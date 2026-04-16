"""
src/models/fusion/__init__.py
-----------------------------
职责：导出多模态融合与 Transformer 编码相关公共接口。
"""

from src.models.fusion.multimodal_transformer import (
    MultimodalDecoder,
    UnifiedMultimodalEncoder,
)

__all__ = [
    "MultimodalDecoder",
    "UnifiedMultimodalEncoder",
]
