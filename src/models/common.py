"""
src/models/common.py
--------------------
职责：提供 `src/models` 目录下共享的轻量工具函数。

用法：
    from src.models.common import cfg_get

    hidden_dim = cfg_get(cfg, "hidden_dim", 256)
"""

from __future__ import annotations

from typing import Any, Mapping


def cfg_get(cfg: Mapping[str, Any] | object, key: str, default: Any = None) -> Any:
    """
    作用：从字典或对象中统一读取配置项。

    输入：
        cfg: Mapping[str, Any] | object，配置字典或配置对象
        key: str，配置键名
        default: Any，默认值
    输出：
        Any，对应配置值，不存在时返回 default
    """
    return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)
