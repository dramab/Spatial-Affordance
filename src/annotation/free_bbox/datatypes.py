"""
src/annotation/free_bbox/datatypes.py
--------------------------------------
通用数据类定义，作为放置规划 pipeline 的标准数据接口。

所有通用模块通过这些数据类通信，不包含任何数据集特定的字段或转换。
数据集适配器负责将原始数据转换为这些格式。

用法:
    from src.annotation.free_bbox.datatypes import SceneData, CameraParams, ObjectInfo, PlacementConfig
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class CameraParams:
    """
    与数据集无关的相机参数。

    所有长度单位与场景工作单位一致（由 SceneData.unit 指定）。

    属性:
        fx, fy, cx, cy: 相机内参
        E_c2w: (4,4) camera→world 变换矩阵
        img_w, img_h: 图像宽高（像素）
    """
    fx: float
    fy: float
    cx: float
    cy: float
    E_c2w: np.ndarray   # (4,4) camera→world
    img_w: int
    img_h: int

    @property
    def K(self) -> np.ndarray:
        """3x3 内参矩阵。"""
        return np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0],
        ], dtype=np.float64)

    @property
    def E_w2c(self) -> np.ndarray:
        """(4,4) world→camera 变换矩阵（E_c2w 的逆）。"""
        return np.linalg.inv(self.E_c2w)


@dataclass
class ObjectInfo:
    """
    单个物体的通用信息。

    bbox3d_canonical 是核心尺度信息，可从 mesh、CAD 尺寸表或手动标注获得。
    pose_world 统一为 object→world 变换，adapter 负责从数据集特定格式转换。

    属性:
        obj_id: 物体唯一标识（如 "obj_0"）
        class_name: 类别名（如 "AlphabetSoup"）
        bbox3d_canonical: (6,) [min_x, min_y, min_z, max_x, max_y, max_z] 物体规范坐标系 AABB
        pose_world: (4,4) object→world 变换矩阵
    """
    obj_id: str
    class_name: str
    bbox3d_canonical: np.ndarray   # (6,) AABB in object canonical frame
    pose_world: np.ndarray         # (4,4) object→world


@dataclass
class SceneData:
    """
    场景数据，PlacementPipeline 的统一输入。

    depth 已由 adapter 转换为场景工作单位，通用代码不做单位转换。

    属性:
        scene_id: 场景标识
        frame_id: 帧标识
        rgb: (H,W,3) uint8 RGB 图像
        depth: (H,W) float 深度图，已转换为场景工作单位
        camera: 相机参数
        objects: 场景中所有物体信息列表
        unit: 场景工作单位标识（默认 "cm"）
    """
    scene_id: str
    frame_id: str
    rgb: np.ndarray              # (H,W,3) uint8
    depth: np.ndarray            # (H,W) float, 场景工作单位
    camera: CameraParams
    objects: list
    unit: str = "cm"


@dataclass
class PlacementConfig:
    """
    放置规划的可配置参数。

    所有长度参数的单位与 SceneData.unit 一致。

    属性:
        voxel_size: 体素边长
        pixel_stride: 深度图采样步长（每 N×N 像素块取 1 个）
        grid_padding: 栅格边界 padding
        safety_margin: 碰撞安全边距
        yaw_steps: yaw 旋转离散步数（360° / yaw_steps = 每步角度）
        min_surface_area: 最小支撑面面积（单位²）
        min_support_ratio: 最小支撑比（0~1）
        occlusion_threshold: 遮挡阈值（OBB 角点被遮挡的比例上限）
        dbscan_eps: DBSCAN 聚类半径
        dbscan_min_samples: DBSCAN 最小样本数
        world_up: 世界坐标系上方向 (3,)
        vis_margin_px: 可视化边距（像素）
    """
    voxel_size: float = 1.0
    pixel_stride: int = 4
    grid_padding: float = 10.0
    safety_margin: float = 0.5
    yaw_steps: int = 24
    min_surface_area: float = 50.0
    min_support_ratio: float = 1.0
    occlusion_threshold: float = 0.3
    dbscan_eps: float = 5.0
    dbscan_min_samples: int = 1
    world_up: tuple = (0.0, 0.0, 1.0)
    vis_margin_px: int = 30
    stability_chunk_size: int = 2000


@dataclass
class PlacementResult:
    """
    单个物体的放置规划结果。

    属性:
        obj_id: 物体标识
        class_name: 类别名
        original_aabb_world: (6,) 原始位置的世界坐标 AABB
        placements: 放置候选列表，每个元素为 dict:
            {
                "center_world": [x, y, z],
                "aabb_world": [min_x, min_y, min_z, max_x, max_y, max_z],
                "yaw_rad": float,
                "transform_world": (4,4) list,
                "free_space_score": float,
            }
        num_raw_candidates: 过滤前的候选总数
        num_after_stability: 稳定性过滤后数量
        num_after_visibility: 可见性过滤后数量
        num_after_occlusion: 遮挡过滤后数量
    """
    obj_id: str
    class_name: str
    original_aabb_world: np.ndarray
    placements: list
    num_raw_candidates: int = 0
    num_after_stability: int = 0
    num_after_visibility: int = 0
    num_after_occlusion: int = 0
