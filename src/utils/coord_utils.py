"""
src/utils/coord_utils.py
-------------------------
通用坐标变换工具函数。

提供 3D 点变换、相机投影、旋转矩阵构造，以及多模态训练中使用的
camera 标准化坐标系构建与正反变换能力。

用法:
    from src.utils.coord_utils import (
        transform_points,
        project_world,
        rotation_z_3x3,
        build_camera_scene_normalizer,
    )
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SceneNormalizationMeta:
    """
    场景标准化元数据。

    作用:
        保存 world -> camera -> normalized 变换链路中的关键参数，
        便于训练阶段做统一坐标标准化，推理阶段恢复真实尺度。

    输入:
        E_w2c: (4, 4) world -> camera 变换矩阵
        scene_center: (3,) camera 坐标系中的场景中心
        scene_scale: float 单标量尺度
        T_world_to_camera: (4, 4) world -> camera
        T_camera_to_world: (4, 4) camera -> world
        T_camera_to_norm: (4, 4) camera -> normalized
        T_norm_to_camera: (4, 4) normalized -> camera
        T_world_to_norm: (4, 4) world -> normalized
        T_norm_to_world: (4, 4) normalized -> world
    输出:
        SceneNormalizationMeta 标准化元数据
    """

    E_w2c: np.ndarray
    scene_center: np.ndarray
    scene_scale: float
    T_world_to_camera: np.ndarray
    T_camera_to_world: np.ndarray
    T_camera_to_norm: np.ndarray
    T_norm_to_camera: np.ndarray
    T_world_to_norm: np.ndarray
    T_norm_to_world: np.ndarray


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


def build_camera_scene_normalizer(
        points_world: np.ndarray,
        E_w2c: np.ndarray,
        center_mode: str = "bbox",
        scale_eps: float = 1e-6) -> SceneNormalizationMeta:
    """
    根据世界坐标点云构建 camera 场景标准化参数。

    作用:
        先将点云变换到相机坐标系，随后按场景中心平移并用单标量尺度
        做归一化，生成正反变换矩阵。

    输入:
        points_world: (N, 3) 世界坐标点云
        E_w2c: (4, 4) world -> camera 变换矩阵
        center_mode: str 场景中心定义，支持 "bbox" 或 "mean"
        scale_eps: float 尺度下限，避免除零
    输出:
        SceneNormalizationMeta 标准化元数据
    """
    points_world = np.asarray(points_world, dtype=np.float64)
    E_w2c = np.asarray(E_w2c, dtype=np.float64)
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError("points_world must have shape (N, 3)")
    if points_world.shape[0] == 0:
        raise ValueError("points_world must contain at least one point")

    T_world_to_camera = E_w2c
    points_camera = transform_points(points_world, T_world_to_camera)

    center_mode = str(center_mode).lower()
    if center_mode == "bbox":
        scene_center = (points_camera.min(axis=0) + points_camera.max(axis=0)) / 2.0
    elif center_mode == "mean":
        scene_center = points_camera.mean(axis=0)
    else:
        raise ValueError(f"unsupported center_mode: {center_mode}")

    camera_min = points_camera.min(axis=0)
    camera_max = points_camera.max(axis=0)
    scene_scale = max(float((camera_max - camera_min).max()) * 0.5, float(scale_eps))

    T_camera_to_norm = np.eye(4, dtype=np.float64)
    T_camera_to_norm[:3, :3] /= scene_scale
    T_camera_to_norm[:3, 3] = -scene_center / scene_scale

    T_norm_to_camera = np.eye(4, dtype=np.float64)
    T_norm_to_camera[:3, :3] *= scene_scale
    T_norm_to_camera[:3, 3] = scene_center

    T_camera_to_world = np.linalg.inv(T_world_to_camera)
    T_world_to_norm = T_camera_to_norm @ T_world_to_camera
    T_norm_to_world = T_camera_to_world @ T_norm_to_camera

    return SceneNormalizationMeta(
        E_w2c=E_w2c,
        scene_center=scene_center.astype(np.float64),
        scene_scale=float(scene_scale),
        T_world_to_camera=T_world_to_camera,
        T_camera_to_world=T_camera_to_world,
        T_camera_to_norm=T_camera_to_norm,
        T_norm_to_camera=T_norm_to_camera,
        T_world_to_norm=T_world_to_norm,
        T_norm_to_world=T_norm_to_world,
    )


def normalize_points_world(
        points_world: np.ndarray,
        meta: SceneNormalizationMeta) -> np.ndarray:
    """
    将世界坐标点云转换到 normalized 坐标系。

    输入:
        points_world: (N, 3) 世界坐标点云
        meta: SceneNormalizationMeta 标准化元数据
    输出:
        (N, 3) normalized 坐标点云
    """
    return transform_points(np.asarray(points_world, dtype=np.float64), meta.T_world_to_norm)


def denormalize_points_to_aligned(
        points_norm: np.ndarray,
        meta: SceneNormalizationMeta) -> np.ndarray:
    """
    将 normalized 坐标点云恢复到 camera 坐标系。

    输入:
        points_norm: (N, 3) normalized 坐标点云
        meta: SceneNormalizationMeta 标准化元数据
    输出:
        (N, 3) camera 坐标点云
    """
    return transform_points(np.asarray(points_norm, dtype=np.float64), meta.T_norm_to_camera)


def denormalize_points_to_world(
        points_norm: np.ndarray,
        meta: SceneNormalizationMeta) -> np.ndarray:
    """
    将 normalized 坐标点云恢复到世界坐标系。

    输入:
        points_norm: (N, 3) normalized 坐标点云
        meta: SceneNormalizationMeta 标准化元数据
    输出:
        (N, 3) 世界坐标点云
    """
    return transform_points(np.asarray(points_norm, dtype=np.float64), meta.T_norm_to_world)


def normalize_box_from_aligned(box_aligned: np.ndarray, scene_center: np.ndarray, scene_scale: float) -> np.ndarray:
    """
    将 camera 坐标系下的 7D box 转为 normalized box。

    作用:
        仅对中心和平移尺度相关的尺寸项做平移/缩放，yaw 保持不变。

    输入:
        box_aligned: (..., 7) box，格式为 (cx, cy, cz, l, w, h, yaw)
        scene_center: (3,) camera 场景中心
        scene_scale: float 单标量尺度
    输出:
        (..., 7) normalized box
    """
    box_aligned = np.asarray(box_aligned, dtype=np.float64)
    normalized = np.array(box_aligned, dtype=np.float64, copy=True)
    normalized[..., :3] = (normalized[..., :3] - np.asarray(scene_center, dtype=np.float64)) / float(scene_scale)
    normalized[..., 3:6] = normalized[..., 3:6] / float(scene_scale)
    return normalized


def denormalize_box_to_aligned(box_norm: np.ndarray, scene_center: np.ndarray, scene_scale: float) -> np.ndarray:
    """
    将 normalized 坐标系下的 7D box 恢复到 camera 坐标系。

    作用:
        仅对中心和平移尺度相关的尺寸项做反归一化，yaw 保持不变。

    输入:
        box_norm: (..., 7) normalized box，格式为 (cx, cy, cz, l, w, h, yaw)
        scene_center: (3,) camera 场景中心
        scene_scale: float 单标量尺度
    输出:
        (..., 7) camera 坐标 box
    """
    box_norm = np.asarray(box_norm, dtype=np.float64)
    denormalized = np.array(box_norm, dtype=np.float64, copy=True)
    denormalized[..., :3] = denormalized[..., :3] * float(scene_scale) + np.asarray(scene_center, dtype=np.float64)
    denormalized[..., 3:6] = denormalized[..., 3:6] * float(scene_scale)
    return denormalized


def build_camera_aligned_scene_normalizer(
        points_world: np.ndarray,
        E_w2c: np.ndarray,
        center_mode: str = "bbox",
        scale_eps: float = 1e-6) -> SceneNormalizationMeta:
    """
    兼容旧接口名称，等价于 build_camera_scene_normalizer。

    输入:
        points_world: (N, 3) 世界坐标点云
        E_w2c: (4, 4) world -> camera 变换矩阵
        center_mode: str 场景中心定义
        scale_eps: float 尺度下限
    输出:
        SceneNormalizationMeta 标准化元数据
    """
    return build_camera_scene_normalizer(
        points_world=points_world,
        E_w2c=E_w2c,
        center_mode=center_mode,
        scale_eps=scale_eps,
    )


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
    分析物体姿态是否为可直接保留的轴对齐稳定姿态。

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
            - is_axis_aligned: 是否存在局部轴与世界竖直方向足够对齐
            - is_flat: 是否为平放姿态
            - is_upright: 是否为竖立姿态
            - is_reasonable: 是否为合理姿态（轴对齐稳定姿态）

    判定规则:
        - 先找到最接近世界竖直方向的局部轴
        - 若该轴与竖直方向夹角足够小，则视为轴对齐稳定姿态，可保留原始 roll/pitch
        - 若该轴恰好是 canonical 最短轴，则额外标记为平放
        - 若该轴恰好是 canonical 最长轴，则额外标记为竖立
        - 若没有任何局部轴足够接近竖直方向，则视为倾斜/不合理姿态

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
    axis_aligned_threshold = min(flat_alignment_threshold, upright_alignment_threshold)

    is_flat = (vertical_axis_index == flat_axis_index and
               vertical_axis_alignment >= flat_alignment_threshold)
    is_upright = (vertical_axis_index == upright_axis_index and
                  vertical_axis_alignment >= upright_alignment_threshold)
    is_axis_aligned = vertical_axis_alignment >= axis_aligned_threshold

    return {
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "vertical_axis_index": vertical_axis_index,
        "vertical_axis_alignment": vertical_axis_alignment,
        "flat_axis_index": flat_axis_index,
        "upright_axis_index": upright_axis_index,
        "is_axis_aligned": is_axis_aligned,
        "is_flat": is_flat,
        "is_upright": is_upright,
        "is_reasonable": is_axis_aligned
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
