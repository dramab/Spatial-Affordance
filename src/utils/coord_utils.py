"""
src/utils/coord_utils.py
-------------------------
通用坐标变换工具函数。

提供 3D 点变换、相机投影、旋转矩阵构造等基础操作，
不包含任何数据集特定的坐标约定或单位转换。

用法:
    from src.utils.coord_utils import transform_points, project_world, rotation_z_3x3
"""

import numpy as np


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    用 4×4 变换矩阵变换 3D 点集。

    输入:
        points: (N, 3) 3D 点坐标
        T: (4, 4) 齐次变换矩阵
    输出:
        (N, 3) 变换后的 3D 点坐标
    """
    ones = np.ones((len(points), 1), dtype=points.dtype)
    pts_h = np.hstack([points, ones])                    # (N, 4)
    return (T @ pts_h.T).T[:, :3]                        # (N, 3)


def project_world(points_world: np.ndarray, K: np.ndarray,
                  E_w2c: np.ndarray) -> tuple:
    """
    将世界坐标 3D 点投影到图像像素坐标。

    输入:
        points_world: (N, 3) 世界坐标点
        K: (3, 3) 相机内参矩阵
        E_w2c: (4, 4) world→camera 变换矩阵
    输出:
        uv: (N, 2) 像素坐标 [u, v]
        z_cam: (N,) 相机坐标系下的深度值（用于判断是否在相机前方）
    """
    pts_cam = transform_points(points_world, E_w2c)      # (N, 3)
    z_cam = pts_cam[:, 2]
    # 避免除零
    z_safe = np.where(np.abs(z_cam) < 1e-8, 1e-8, z_cam)
    uv = (K @ pts_cam.T).T[:, :2]                        # (N, 2) 未归一化
    uv[:, 0] /= z_safe
    uv[:, 1] /= z_safe
    return uv, z_cam


def rotation_z_3x3(angle_rad: float) -> np.ndarray:
    """
    绕 Z 轴旋转的 3×3 旋转矩阵。

    输入:
        angle_rad: 旋转角度（弧度）
    输出:
        (3, 3) 旋转矩阵
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [ c, -s, 0.0],
        [ s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def compute_placed_transform(bbox3d_canonical: np.ndarray,
                             center_world: np.ndarray,
                             yaw_rad: float,
                             world_up_axis: int = 2) -> np.ndarray:
    """
    计算物体放置到指定位置的 4×4 object→world 变换矩阵。

    将物体规范坐标系的 AABB 中心平移到 center_world，并绕 world_up 轴旋转 yaw_rad。

    输入:
        bbox3d_canonical: (6,) [min_x, min_y, min_z, max_x, max_y, max_z]
        center_world: (3,) 目标世界坐标中心
        yaw_rad: yaw 旋转角度（弧度）
        world_up_axis: 世界坐标系上方向轴索引（默认 2 = Z-up）
    输出:
        (4, 4) object→world 变换矩阵
    """
    obj_center = (bbox3d_canonical[:3] + bbox3d_canonical[3:]) / 2.0
    R = rotation_z_3x3(yaw_rad)
    t = center_world - R @ obj_center
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
