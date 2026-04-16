"""
src/models/heads/__init__.py
----------------------------
职责：导出检测头相关公共接口。
"""

from src.models.heads.bbox3d_head import BBox3DHead

__all__ = ["BBox3DHead"]
