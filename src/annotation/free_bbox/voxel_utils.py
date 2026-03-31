"""
src/annotation/free_bbox/voxel_utils.py
----------------------------------------
体素坐标系工具函数。

提供体素参数构建、世界坐标↔体素索引的双向转换。
体素参数 dict 是所有栅格操作的基础。

用法:
    from src.annotation.free_bbox.voxel_utils import make_voxel_params, world_to_voxel, voxel_to_world
"""

import numpy as np


def make_voxel_params(grid_min: np.ndarray, voxel_size: float) -> dict:
    """
    构建体素参数字典，用于坐标↔索引转换。

    转换规则:
        world → index : i = floor((p - origin) / voxel_size)
        index → world : p = origin + (i + 0.5) * voxel_size  (体素中心)

    输入:
        grid_min: (3,) 栅格最小角的世界坐标
        voxel_size: 体素边长（场景单位）
    输出:
        dict {"voxel_size": float, "origin": [x, y, z]}
    """
    return {
        "voxel_size": float(voxel_size),
        "origin":     grid_min.tolist(),
    }


def world_to_voxel(points: np.ndarray, voxel_params: dict) -> np.ndarray:
    """
    世界坐标 → 整数体素索引。

    输入:
        points: (..., 3) 世界坐标
        voxel_params: make_voxel_params() 返回的 dict
    输出:
        (..., 3) int 体素索引
    """
    origin = np.asarray(voxel_params["origin"], dtype=np.float64)
    vs = voxel_params["voxel_size"]
    return np.floor((points - origin) / vs).astype(np.intp)


def voxel_to_world(indices: np.ndarray, voxel_params: dict) -> np.ndarray:
    """
    整数体素索引 → 世界坐标（体素中心）。

    输入:
        indices: (..., 3) int 体素索引
        voxel_params: make_voxel_params() 返回的 dict
    输出:
        (..., 3) float 世界坐标
    """
    origin = np.asarray(voxel_params["origin"], dtype=np.float64)
    vs = voxel_params["voxel_size"]
    return origin + (np.asarray(indices, dtype=np.float64) + 0.5) * vs
