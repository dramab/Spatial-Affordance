"""
src/annotation/free_bbox/io_utils.py
--------------------------------------
I/O 工具：PLY 点云导出、占据栅格导出、JSON 标注读写。

用法:
    from src.annotation.free_bbox.io_utils import (
        save_ply, save_occupancy_ply,
        save_placement_annotations, save_placement_samples,
    )
"""

import json
import os
import numpy as np

from src.annotation.bbox3d.bbox_utils import get_bbox_corners
from src.annotation.free_bbox.occupancy import FREE, OCCUPIED, UNKNOWN
from src.utils.coord_utils import transform_points


def save_ply(path, points, colors):
    """
    保存彩色点云为 ASCII PLY 文件。

    输入:
        path: str 输出文件路径
        points: (N, 3) float 点坐标
        colors: (N, 3) uint8 RGB 颜色
    """
    n = len(points)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    data = np.hstack([points.astype(np.float32), colors.astype(np.float32)])
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, data, fmt="%.4f %.4f %.4f %d %d %d")


def save_occupancy_ply(path, grid, grid_min, voxel_size,
                       states=None, max_pts=None):
    """
    将占据栅格中指定状态的体素导出为 PLY 点云。

    输入:
        path: str 输出文件路径
        grid: (Gx, Gy, Gz) uint8 占据栅格
        grid_min: (3,) 栅格最小角世界坐标
        voxel_size: float 体素边长
        states: list[int] 要导出的体素状态，默认 [OCCUPIED]
        max_pts: int 最大点数（随机采样），None 表示不限制
    """
    if states is None:
        states = [OCCUPIED]

    mask = np.zeros(grid.shape, dtype=bool)
    for s in states:
        mask |= (grid == s)

    idx = np.argwhere(mask)
    if max_pts is not None and len(idx) > max_pts:
        rng = np.random.default_rng(42)
        idx = idx[rng.choice(len(idx), max_pts, replace=False)]

    pts = grid_min + (idx + 0.5) * voxel_size

    # 颜色映射: FREE=绿, OCCUPIED=红, UNKNOWN=灰
    color_map = {
        FREE:     [76, 175, 80],
        OCCUPIED: [244, 67, 54],
        UNKNOWN:  [158, 158, 158],
    }
    colors = np.array([color_map.get(grid[i, j, k], [128, 128, 128])
                        for i, j, k in idx], dtype=np.uint8)

    save_ply(path, pts, colors)


def save_json(path, data):
    """保存 JSON 数据。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def camera_to_json(camera):
    """
    用法: payload = camera_to_json(scene.camera)
    作用: 将 CameraParams 转成 placement 元数据可保存的 JSON 字段
    输入: camera: CameraParams，已按 SceneData.unit 标准化的相机参数
    输出: dict，包含内参、图像尺寸和 camera->world 外参
    """
    return {
        "fx": float(camera.fx),
        "fy": float(camera.fy),
        "cx": float(camera.cx),
        "cy": float(camera.cy),
        "img_w": int(camera.img_w),
        "img_h": int(camera.img_h),
        "E_c2w": np.asarray(camera.E_c2w, dtype=np.float64).tolist(),
    }


def object_info_to_json(obj):
    """
    用法: record = object_info_to_json(scene.objects[0])
    作用: 将 ObjectInfo 转成可保存的全场景物体几何记录
    输入: obj，包含 obj_id、class_name、bbox3d_canonical 与 pose_world
    输出: dict，包含规范 AABB、世界位姿、世界角点与世界 AABB
    """
    canonical_aabb = np.asarray(obj.bbox3d_canonical, dtype=np.float64)
    pose_world = np.asarray(obj.pose_world, dtype=np.float64)
    corners_world = transform_points(get_bbox_corners(canonical_aabb), pose_world)
    aabb_world = np.concatenate([corners_world.min(axis=0), corners_world.max(axis=0)])
    return {
        "object_id": str(obj.obj_id),
        "class_name": str(obj.class_name),
        "canonical_aabb_object": canonical_aabb.tolist(),
        "pose_world": pose_world.tolist(),
        "corners_world": corners_world.tolist(),
        "aabb_world": aabb_world.tolist(),
    }


def scene_objects_to_json(scene):
    """
    用法: payload = scene_objects_to_json(scene)
    作用: 构建当前帧所有物体的 benchmark 友好几何快照
    输入: scene: SceneData，包含 scene_id、frame_id、unit 与 objects
    输出: dict，可直接写入 scene_objects JSON
    """
    return {
        "schema_version": "placement_scene_objects/v1",
        "scene_id": str(scene.scene_id),
        "frame_id": str(scene.frame_id),
        "unit": str(getattr(scene, "unit", "cm")),
        "objects": [object_info_to_json(obj) for obj in scene.objects],
    }


def save_scene_objects(path, scene):
    """
    用法: save_scene_objects(path, scene)
    作用: 保存当前帧所有物体的几何快照
    输入: path: str；scene: SceneData
    输出: None
    """
    save_json(path, scene_objects_to_json(scene))


def save_placement_annotations(path, annotations):
    """保存按物体分组的 placement 主标注。"""
    save_json(path, annotations)


def save_placement_samples(path, samples):
    """保存按 placement 展平的样本标注。"""
    save_json(path, samples)


def save_placement_result(path, config_dict, object_results):
    """兼容旧接口：保存旧版 placement 结果。"""
    save_json(path, {
        "config": config_dict,
        "objects": object_results,
    })


def load_json(path):
    """加载 JSON 文件。"""
    with open(path, "r") as f:
        return json.load(f)


def load_placement_result(path):
    """兼容旧接口：加载 placement 结果 JSON。"""
    return load_json(path)


def save_grid_meta(path, voxel_params, grid_shape, voxel_counts=None,
                   extra=None):
    """
    保存栅格元数据为 JSON。

    输入:
        path: str 输出文件路径
        voxel_params: dict 体素参数
        grid_shape: tuple 栅格尺寸
        voxel_counts: dict 各状态体素计数（可选）
        extra: dict 额外信息（可选）
    """
    meta = {
        "voxel_params": voxel_params,
        "grid_shape": list(grid_shape),
    }
    if voxel_counts is not None:
        meta["voxel_counts"] = voxel_counts
    if extra is not None:
        meta.update(extra)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def _json_default(obj):
    """JSON 序列化 numpy 类型的 fallback。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
