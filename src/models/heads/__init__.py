"""
src/models/heads/__init__.py
----------------------------
职责：导出检测头相关公共接口。
"""

from src.models.heads.center3d_head import Center3DHead
from src.models.heads.size_yaw_head import SizeYawHead

__all__ = ["Center3DHead", "SizeYawHead"]
