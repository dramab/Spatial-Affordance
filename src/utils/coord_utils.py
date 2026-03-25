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


def rotation_matrix_to_euler_zyx(R: np.ndarray) -> tuple:
    """
    从旋转矩阵提取 ZYX 欧拉角（roll, pitch, yaw）。

    输入:
        R: (3, 3) 旋转矩阵
    输出:
        (roll, pitch, yaw) 弧度，范围 [-π, π]
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def rotation_matrix_from_euler_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    从 ZYX 欧拉角构造旋转矩阵。

    输入:
        roll, pitch, yaw: 弧度
    输出:
        (3, 3) 旋转矩阵
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ], dtype=np.float64)
    return R


def analyze_pose_orientation(T_obj2world: np.ndarray,
                             bbox3d_canonical: np.ndarray,
                             flat_threshold_deg: float = 15.0,
                             upright_threshold_deg: float = 15.0) -> dict:
    """
    分析物体姿态是否为合理的平放或竖立姿态。

    输入:
        T_obj2world: (4, 4) object→world 变换矩阵
        bbox3d_canonical: (6,) 物体 canonical AABB，用于估计长轴/短轴
        flat_threshold_deg: 平放姿态容差（度）
        upright_threshold_deg: 竖立姿态容差（度）
    输出:
        dict 包含:
            - roll, pitch, yaw: 弧度
            - vertical_axis_index: 最接近世界竖直方向的局部轴索引
            - vertical_axis_alignment: 该局部轴与世界竖直方向的对齐程度
            - flat_axis_index: canonical 最短轴索引
            - upright_axis_index: canonical 最长轴索引
            - is_flat: 是否为平放姿态
            - is_upright: 是否为竖立姿态
            - is_reasonable: 是否为合理姿态（平放或竖立）

    判定规则:
        - 先找到最接近世界竖直方向的局部轴
        - 平放: 该轴是 canonical 最短轴，且与竖直方向夹角足够小
        - 竖立: 该轴是 canonical 最长轴，且与竖直方向夹角足够小
        - 其余姿态视为倾斜/不合理姿态

    说明:
        当姿态被判为不合理时，放置规划阶段不会保留原始 roll/pitch，
        而是回退到平放 + yaw 扫描的标准放置姿态。
    """
    R = T_obj2world[:3, :3]
    roll, pitch, yaw = rotation_matrix_to_euler_zyx(R)

    axis_sizes = np.asarray(bbox3d_canonical[3:], dtype=np.float64) - np.asarray(
        bbox3d_canonical[:3], dtype=np.float64)
    flat_axis_index = int(np.argmin(axis_sizes))
    upright_axis_index = int(np.argmax(axis_sizes))

    # 通过局部轴和世界竖直方向的夹角判断姿态，减少欧拉角分解歧义带来的误判。
    up_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    axis_alignments = np.abs(R.T @ up_world)
    vertical_axis_index = int(np.argmax(axis_alignments))
    vertical_axis_alignment = float(axis_alignments[vertical_axis_index])

    flat_alignment_threshold = float(np.cos(np.deg2rad(flat_threshold_deg)))
    upright_alignment_threshold = float(np.cos(np.deg2rad(upright_threshold_deg)))

    is_flat = (vertical_axis_index == flat_axis_index and
               vertical_axis_alignment >= flat_alignment_threshold)
    is_upright = (vertical_axis_index == upright_axis_index and
                  vertical_axis_alignment >= upright_alignment_threshold)

    return {
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "vertical_axis_index": vertical_axis_index,
        "vertical_axis_alignment": vertical_axis_alignment,
        "flat_axis_index": flat_axis_index,
        "upright_axis_index": upright_axis_index,
        "is_flat": is_flat,
        "is_upright": is_upright,
        "is_reasonable": is_flat or is_upright
    }

def compute_placed_transform_with_orientation(bbox3d_canonical: np.ndarray,
                                              center_world: np.ndarray,
                                              roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    计算物体放置变换，支持完整的 roll/pitch/yaw 姿态。

    输入:
        bbox3d_canonical: (6,) [min_x, min_y, min_z, max_x, max_y, max_z]
        center_world: (3,) 目标世界坐标中心
        roll, pitch, yaw: 欧拉角（弧度）
    输出:
        (4, 4) object→world 变换矩阵
    """
    obj_center = (bbox3d_canonical[:3] + bbox3d_canonical[3:]) / 2.0
    R = rotation_matrix_from_euler_zyx(roll, pitch, yaw)
    t = center_world - R @ obj_center
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
