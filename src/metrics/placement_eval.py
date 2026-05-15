"""
src/metrics/placement_eval.py
-----------------------------
放置预测结果评测工具函数。

提供四类可复用指标：
1. 预测 3D box 是否与上游 occupancy grid 中的 OCCUPIED 体素碰撞
2. 预测放置方向是否符合结构化指令关系
3. 预测 3D box 体积是否与目标物体体积一致
4. 预测移动前物体中心投影点是否落在目标物体原始 3D 框投影区域内

用法：
    from src.metrics.placement_eval import (
        evaluate_collision,
        evaluate_direction,
        evaluate_projected_object_center,
        evaluate_size_consistency,
        summarize_metric_records,
    )
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.annotation.bbox3d.bbox_utils import get_bbox_corners
from src.annotation.free_bbox.datatypes import ObjectInfo
from src.annotation.free_bbox.grid_ops import voxelize_obb
from src.annotation.free_bbox.occupancy import OCCUPIED, UNKNOWN
from src.utils.coord_utils import box7d_to_corners_world, project_world, rotation_z_3x3, transform_points


DEFAULT_COLLISION_RATIO_THRESHOLD = 0.003
DEFAULT_VOLUME_ERROR_RATIO_THRESHOLD = 0.1
DEFAULT_AXIS_SIZE_ERROR_SUM_THRESHOLD_CM = 5.0
DEFAULT_DIRECTION_CENTER_L2_THRESHOLD_CM = 10.0
DEFAULT_SUPPORT_IGNORE_LAYERS = 4


def _load_auto_label_module():
    """
    用法: auto_label = _load_auto_label_module()
    作用: 延迟加载 auto_label 中已有的几何关系函数，避免重复实现方向判定。
    输入: 无
    输出: tools.auto_label 模块
    """
    from tools import auto_label

    return auto_label


def as_box7d(box_value: Sequence[float], name: str = "box") -> np.ndarray:
    """
    用法: box = as_box7d(record["pred_box_world"], "pred_box_world")
    作用: 将输入校验并转换为 7D box 数组
    输入: box_value: 长度为 7 的数值序列；name: 报错时使用的字段名
    输出: ndarray(7,)，格式为 [cx, cy, cz, sx, sy, sz, yaw_degrees]
    """
    box = np.asarray(box_value, dtype=np.float64)
    if box.shape != (7,):
        raise ValueError(f"{name} must have shape (7,), got {box.shape}")
    if not np.isfinite(box).all():
        raise ValueError(f"{name} contains non-finite values")
    if np.any(box[3:6] <= 0.0):
        raise ValueError(f"{name} size values must be positive, got {box[3:6].tolist()}")
    return box


def as_center3d(center_value: Sequence[float], name: str = "center") -> np.ndarray:
    """
    用法: center = as_center3d(record["pred_object_center_world"], "pred_object_center_world")
    作用: 将输入校验并转换为 3D center 数组
    输入: center_value: 长度为 3 的数值序列；name: 报错时使用的字段名
    输出: ndarray(3,)，格式为 [cx, cy, cz]
    """
    center = np.asarray(center_value, dtype=np.float64)
    if center.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {center.shape}")
    if not np.isfinite(center).all():
        raise ValueError(f"{name} contains non-finite values")
    return center


def as_corners3d(corners_value: Sequence[Sequence[float]], name: str = "corners") -> np.ndarray:
    """
    用法: corners = as_corners3d(record["target_object"]["corners_world"], "target_corners_world")
    作用: 将输入校验并转换为 3D 角点数组
    输入: corners_value: (N, 3) 数值序列，至少包含 3 个点；name: 报错字段名
    输出: ndarray(N, 3)，世界坐标角点
    """
    corners = np.asarray(corners_value, dtype=np.float64)
    if corners.ndim != 2 or corners.shape[1] != 3 or corners.shape[0] < 3:
        raise ValueError(f"{name} must have shape (N,3) with N>=3, got {corners.shape}")
    if not np.isfinite(corners).all():
        raise ValueError(f"{name} contains non-finite values")
    return corners


def object_info_to_corners_world(obj: ObjectInfo) -> np.ndarray:
    """
    用法: corners = object_info_to_corners_world(scene.objects[0])
    作用: 将 ObjectInfo 的 canonical AABB 和 object->world 位姿转换为世界坐标角点
    输入: obj: ObjectInfo，包含 bbox3d_canonical 和 pose_world
    输出: ndarray(8,3)，世界坐标 3D box 角点
    """
    corners_object = get_bbox_corners(np.asarray(obj.bbox3d_canonical, dtype=np.float64))
    return transform_points(corners_object, np.asarray(obj.pose_world, dtype=np.float64))


def camera_to_projection_matrices(camera: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """
    用法: K, E_w2c = camera_to_projection_matrices(sample["camera"])
    作用: 从 benchmark camera 字段构造投影矩阵
    输入: camera: dict，包含 fx/fy/cx/cy/E_c2w
    输出: tuple，分别为 3x3 内参矩阵和 4x4 world->camera 外参矩阵
    """
    e_c2w = np.asarray(camera["E_c2w"], dtype=np.float64)
    if e_c2w.shape != (4, 4):
        raise ValueError(f"camera.E_c2w must have shape (4,4), got {e_c2w.shape}")
    return (
        np.array(
            [
                [float(camera["fx"]), 0.0, float(camera["cx"])],
                [0.0, float(camera["fy"]), float(camera["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        np.linalg.inv(e_c2w),
    )


def polygon_area_2d(points_uv: np.ndarray) -> float:
    """
    用法: area = polygon_area_2d(hull_uv)
    作用: 计算二维多边形面积
    输入: points_uv: (N, 2) 按边界顺序排列的顶点
    输出: float，顶点不足 3 个时返回 0
    """
    points_uv = np.asarray(points_uv, dtype=np.float64)
    if points_uv.shape[0] < 3:
        return 0.0
    x = points_uv[:, 0]
    y = points_uv[:, 1]
    return float(abs(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))))


def _cross_2d(origin: np.ndarray, point_a: np.ndarray, point_b: np.ndarray) -> float:
    """
    用法: value = _cross_2d(origin, point_a, point_b)
    作用: 计算二维向量 origin->point_a 与 origin->point_b 的叉积
    输入: origin/point_a/point_b: (2,) 二维点
    输出: float，叉积值
    """
    return float(
        (point_a[0] - origin[0]) * (point_b[1] - origin[1])
        - (point_a[1] - origin[1]) * (point_b[0] - origin[0])
    )


def convex_hull_2d(points_uv: np.ndarray) -> np.ndarray:
    """
    用法: hull = convex_hull_2d(projected_corners)
    作用: 计算二维点集凸包，用于近似目标物体 3D 框投影区域
    输入: points_uv: (N, 2) 像素坐标点
    输出: ndarray(M, 2)，按边界顺序排列的凸包顶点
    """
    points = np.unique(np.asarray(points_uv, dtype=np.float64), axis=0)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points_uv must have shape (N,2), got {points.shape}")
    if points.shape[0] <= 2:
        return points

    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    lower: list[np.ndarray] = []
    for point in points:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: list[np.ndarray] = []
    for point in reversed(points):
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def point_in_convex_polygon(point_uv: np.ndarray, polygon_uv: np.ndarray, eps: float = 1e-6) -> bool:
    """
    用法: inside = point_in_convex_polygon(point_uv, hull_uv)
    作用: 判断二维点是否在凸多边形内，边界点视为在内部
    输入: point_uv: (2,) 像素点；polygon_uv: (N, 2) 凸包顶点；eps: 数值容差
    输出: bool，True 表示点在凸包内部或边界上
    """
    point = np.asarray(point_uv, dtype=np.float64)
    polygon = np.asarray(polygon_uv, dtype=np.float64)
    if point.shape != (2,):
        raise ValueError(f"point_uv must have shape (2,), got {point.shape}")
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
        raise ValueError(f"polygon_uv must have shape (N,2) with N>=3, got {polygon.shape}")
    edges = np.roll(polygon, -1, axis=0) - polygon
    rel = point - polygon
    cross = edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0]
    return bool(np.all(cross >= -float(eps)) or np.all(cross <= float(eps)))


def evaluate_projected_object_center(
    pred_center_world: Sequence[float],
    target_corners_world: Sequence[Sequence[float]],
    camera: Mapping[str, Any],
) -> dict[str, Any]:
    """
    用法: result = evaluate_projected_object_center(pred_center, target_corners, camera)
    作用: 判断预测物体中心投影点是否落在目标物体原始 3D 框的图像投影区域内
    输入:
        pred_center_world: 长度为 3 的预测世界坐标中心
        target_corners_world: (N, 3) 目标物体原始 3D 框世界角点
        camera: dict，包含 fx/fy/cx/cy/E_c2w
    输出: dict，包含 center_match、pred_center_uv、target_projected_hull_uv 等字段
    """
    pred_center = as_center3d(pred_center_world, "pred_center_world")
    target_corners = as_corners3d(target_corners_world, "target_corners_world")
    k_matrix, e_w2c = camera_to_projection_matrices(camera)

    pred_uv, pred_z = project_world(pred_center.reshape(1, 3), k_matrix, e_w2c)
    target_uv, target_z = project_world(target_corners, k_matrix, e_w2c)
    target_valid = (
        (target_z > 0.0)
        & np.isfinite(target_z)
        & np.isfinite(target_uv[:, 0])
        & np.isfinite(target_uv[:, 1])
    )
    target_visible_count = int(np.count_nonzero(target_valid))
    if target_visible_count < 3:
        return {
            "evaluated": False,
            "reason": "target object projected box has fewer than 3 valid corners in front of camera",
            "center_match": False,
            "target_corner_count": int(target_corners.shape[0]),
            "target_visible_corner_count": target_visible_count,
        }

    hull_uv = convex_hull_2d(target_uv[target_valid])
    hull_area_px2 = polygon_area_2d(hull_uv)
    if hull_uv.shape[0] < 3 or hull_area_px2 <= 1e-8:
        return {
            "evaluated": False,
            "reason": "target object projected box is degenerate",
            "center_match": False,
            "target_corner_count": int(target_corners.shape[0]),
            "target_visible_corner_count": target_visible_count,
            "target_projected_hull_uv": hull_uv.tolist(),
            "target_projected_hull_area_px2": hull_area_px2,
        }

    pred_uv_value = pred_uv[0]
    pred_depth = float(pred_z[0])
    pred_projected = bool(
        pred_depth > 0.0
        and np.isfinite(pred_depth)
        and np.isfinite(pred_uv_value).all()
    )
    center_inside = bool(
        pred_projected and point_in_convex_polygon(pred_uv_value, hull_uv)
    )
    return {
        "evaluated": True,
        "center_match": center_inside,
        "projected_center_in_target_box": center_inside,
        "pred_center_projected": pred_projected,
        "pred_center_uv": pred_uv_value.tolist() if np.isfinite(pred_uv_value).all() else None,
        "pred_center_depth": pred_depth if np.isfinite(pred_depth) else None,
        "target_projected_hull_uv": hull_uv.tolist(),
        "target_projected_hull_area_px2": hull_area_px2,
        "target_corner_count": int(target_corners.shape[0]),
        "target_visible_corner_count": target_visible_count,
    }


def box7d_to_occupancy_voxels(
    box_world: Sequence[float],
    voxel_params: Mapping[str, Any],
    grid_shape: Sequence[int],
) -> np.ndarray:
    """
    用法: voxels = box7d_to_occupancy_voxels(pred_box, voxel_params, grid.shape)
    作用: 将世界坐标 7D yaw-only box 体素化到 occupancy grid 索引空间
    输入:
        box_world: 长度 7 的世界坐标 box，[cx, cy, cz, sx, sy, sz, yaw_degrees]
        voxel_params: dict，包含 origin 和 voxel_size
        grid_shape: occupancy grid 形状
    输出: ndarray(M,3)，落在 grid 边界内的体素索引
    """
    box = as_box7d(box_world, "box_world")
    half_size = box[3:6] * 0.5
    canonical_bbox = np.concatenate([-half_size, half_size]).astype(np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_z_3x3(np.deg2rad(float(box[6])))
    transform[:3, 3] = box[:3]
    return voxelize_obb(
        canonical_bbox,
        transform,
        dict(voxel_params),
        np.asarray(grid_shape, dtype=int),
    )


def infer_support_layers_from_target_box(
    target_box_world: Sequence[float],
    voxel_params: Mapping[str, Any],
    grid_shape: Sequence[int],
    support_ignore_layers: int = DEFAULT_SUPPORT_IGNORE_LAYERS,
) -> list[int]:
    """
    用法: layers = infer_support_layers_from_target_box(gt_box, voxel_params, grid.shape, 2)
    作用: 由 GT 放置框最低体素层反推需要忽略的桌面支撑层
    输入:
        target_box_world: 长度 7 的 GT 世界坐标 box
        voxel_params: dict，包含 origin 和 voxel_size
        grid_shape: occupancy grid 形状
        support_ignore_layers: 从 GT 最低体素层向下忽略的层数
    输出: list[int]，位于 grid 范围内的 Z 层索引
    """
    layer_count = max(0, int(support_ignore_layers))
    if layer_count == 0:
        return []
    target_voxels = box7d_to_occupancy_voxels(
        target_box_world,
        voxel_params=voxel_params,
        grid_shape=grid_shape,
    )
    if len(target_voxels) == 0:
        return []
    grid_z = int(np.asarray(grid_shape, dtype=int)[2])
    landing_z = int(target_voxels[:, 2].min())
    layers = [landing_z - offset for offset in range(1, layer_count + 1)]
    return [layer for layer in layers if 0 <= layer < grid_z]


def evaluate_collision(
    pred_box_world: Sequence[float],
    occupancy_grid: np.ndarray,
    voxel_params: Mapping[str, Any],
    collision_ratio_threshold: float = DEFAULT_COLLISION_RATIO_THRESHOLD,
    target_box_world: Sequence[float] | None = None,
    support_ignore_layers: int = 0,
) -> dict[str, Any]:
    """
    用法: result = evaluate_collision(pred_box, grid, voxel_params, target_box_world=gt_box)
    作用: 使用上游 occupancy grid 评估预测 box 是否碰撞 OCCUPIED 体素
    输入:
        pred_box_world: 长度 7 的预测世界坐标 box
        occupancy_grid: ndarray(Gx,Gy,Gz)，FREE=0、OCCUPIED=1、UNKNOWN=2
        voxel_params: dict，包含 origin 和 voxel_size
        collision_ratio_threshold: 允许的最大 OCCUPIED 体素占预测体素比例
        target_box_world: 可选 GT box，用于推断桌面支撑层
        support_ignore_layers: 从 GT 最低体素层向下忽略的支撑层数
    输出: dict，包含 collision_free、occupied_collision_ratio、unknown_overlap_ratio 等字段
    """
    grid = np.asarray(occupancy_grid)
    if grid.ndim != 3:
        raise ValueError(f"occupancy_grid must have shape (Gx,Gy,Gz), got {grid.shape}")
    threshold = float(collision_ratio_threshold)
    pred_voxels = box7d_to_occupancy_voxels(
        pred_box_world,
        voxel_params=voxel_params,
        grid_shape=grid.shape,
    )
    pred_voxel_count = int(len(pred_voxels))
    if pred_voxel_count == 0:
        return {
            "evaluated": False,
            "reason": "predicted box has no voxels inside occupancy grid",
            "collision_free": False,
            "pred_voxel_count": 0,
            "occupied_voxel_count": 0,
            "occupied_collision_ratio": None,
            "unknown_voxel_count": 0,
            "unknown_overlap_ratio": None,
            "collision_ratio_threshold": threshold,
            "support_ignore_layers": int(support_ignore_layers),
            "ignored_support_layers": [],
            "ignored_support_occupied_count": 0,
            "collision_pass_rule": "occupied_collision_ratio <= collision_ratio_threshold",
        }

    states = grid[pred_voxels[:, 0], pred_voxels[:, 1], pred_voxels[:, 2]]
    occupied_mask = states == OCCUPIED
    ignored_layers: list[int] = []
    ignored_support_occupied_count = 0
    if target_box_world is not None and int(support_ignore_layers) > 0:
        ignored_layers = infer_support_layers_from_target_box(
            target_box_world,
            voxel_params=voxel_params,
            grid_shape=grid.shape,
            support_ignore_layers=int(support_ignore_layers),
        )
        if ignored_layers:
            support_mask = np.isin(pred_voxels[:, 2], np.asarray(ignored_layers, dtype=int))
            ignored_support_occupied_count = int(np.count_nonzero(occupied_mask & support_mask))
            occupied_mask = occupied_mask & ~support_mask

    occupied_count = int(np.count_nonzero(occupied_mask))
    unknown_count = int(np.count_nonzero(states == UNKNOWN))
    occupied_ratio = float(occupied_count / max(pred_voxel_count, 1))
    unknown_ratio = float(unknown_count / max(pred_voxel_count, 1))
    collision_free = bool(occupied_ratio <= threshold)

    return {
        "evaluated": True,
        "collision_free": collision_free,
        "pred_voxel_count": pred_voxel_count,
        "occupied_voxel_count": occupied_count,
        "occupied_collision_ratio": occupied_ratio,
        "unknown_voxel_count": unknown_count,
        "unknown_overlap_ratio": unknown_ratio,
        "collision_ratio_threshold": threshold,
        "support_ignore_layers": int(support_ignore_layers),
        "ignored_support_layers": ignored_layers,
        "ignored_support_occupied_count": ignored_support_occupied_count,
        "collision_pass_rule": "occupied_collision_ratio <= collision_ratio_threshold",
    }


def evaluate_size_consistency(
    pred_box_world: Sequence[float],
    target_box_world: Sequence[float],
    volume_error_ratio_threshold: float = DEFAULT_VOLUME_ERROR_RATIO_THRESHOLD,
    axis_size_error_sum_threshold_cm: float = DEFAULT_AXIS_SIZE_ERROR_SUM_THRESHOLD_CM,
) -> dict[str, Any]:
    """
    用法: result = evaluate_size_consistency(pred_box, gt_box)
    作用: 评估预测 box 体积和三轴尺寸绝对误差之和是否与目标 box 一致
    输入:
        pred_box_world: 长度 7 的预测世界坐标 box
        target_box_world: 长度 7 的 GT 世界坐标 box
        volume_error_ratio_threshold: 预测体积相对 GT 体积误差阈值
        axis_size_error_sum_threshold_cm: 三轴尺寸绝对误差之和阈值，单位 cm
    输出: dict，包含 size_consistent、pred_volume_cm3、target_volume_cm3、
         volume_error_ratio、axis_size_error_sum_cm 与对应阈值
    """
    pred_box = as_box7d(pred_box_world, "pred_box_world")
    target_box = as_box7d(target_box_world, "target_box_world")
    pred_size = pred_box[3:6]
    target_size = target_box[3:6]
    pred_volume_cm3 = float(np.prod(pred_size))
    target_volume_cm3 = float(np.prod(target_size))
    volume_error_cm3 = float(abs(pred_volume_cm3 - target_volume_cm3))
    volume_error_ratio = float(volume_error_cm3 / target_volume_cm3)
    axis_size_errors_cm = np.abs(pred_size - target_size)
    axis_size_error_sum_cm = float(np.sum(axis_size_errors_cm))
    volume_pass = bool(volume_error_ratio <= float(volume_error_ratio_threshold))
    axis_size_pass = bool(axis_size_error_sum_cm <= float(axis_size_error_sum_threshold_cm))
    size_consistent = bool(volume_pass and axis_size_pass)
    return {
        "evaluated": True,
        "size_consistent": size_consistent,
        "volume_pass": volume_pass,
        "axis_size_pass": axis_size_pass,
        "pred_volume_cm3": pred_volume_cm3,
        "target_volume_cm3": target_volume_cm3,
        "volume_error_cm3": volume_error_cm3,
        "volume_error_ratio": volume_error_ratio,
        "volume_error_ratio_threshold": float(volume_error_ratio_threshold),
        "axis_size_errors_cm": axis_size_errors_cm.tolist(),
        "axis_size_error_sum_cm": axis_size_error_sum_cm,
        "axis_size_error_sum_threshold_cm": float(axis_size_error_sum_threshold_cm),
        "size_pass_rule": (
            "volume_error_ratio <= volume_error_ratio_threshold and "
            "axis_size_error_sum_cm <= axis_size_error_sum_threshold_cm"
        ),
    }


def evaluate_direction(
    pred_box_world: Sequence[float],
    reference_corners_world: np.ndarray,
    camera: Mapping[str, Any],
    expected_relation: str,
    target_box_world: Sequence[float] | None = None,
    center_l2_threshold_cm: float = DEFAULT_DIRECTION_CENTER_L2_THRESHOLD_CM,
) -> dict[str, Any]:
    """
    用法: result = evaluate_direction(pred_box, ref_corners, camera, "the right of")
    作用: 使用 auto_label 同源空间关系逻辑评估预测位置方向是否符合结构化指令
    输入:
        pred_box_world: 长度 7 的预测世界坐标 box
        reference_corners_world: ndarray(N,3)，指令目标参考物世界坐标角点
        camera: dict，包含 fx/fy/cx/cy/E_c2w
        expected_relation: str，auto_label 生成指令时保存的目标关系
        target_box_world: 可选 GT box，中心足够接近时直接判为方向正确
        center_l2_threshold_cm: 中心点 L2 距离直通阈值，单位 cm
    输出: dict，包含 direction_correct、pred_relation、expected_relation、center_l2_error_cm
    """
    pred_box = as_box7d(pred_box_world, "pred_box_world")
    expected_relation = str(expected_relation)
    center_l2_error_cm: float | None = None
    if target_box_world is not None:
        target_box = as_box7d(target_box_world, "target_box_world")
        center_l2_error_cm = float(np.linalg.norm(pred_box[:3] - target_box[:3]))
        center_match = bool(center_l2_error_cm <= float(center_l2_threshold_cm))
        if center_match:
            return {
                "evaluated": True,
                "direction_correct": True,
                "pred_relation": "center_match",
                "expected_relation": expected_relation,
                "center_match": True,
                "center_l2_error_cm": center_l2_error_cm,
                "center_l2_threshold_cm": float(center_l2_threshold_cm),
            }

    auto_label = _load_auto_label_module()
    pred_corners = box7d_to_corners_world(pred_box)
    e_c2w = np.asarray(camera["E_c2w"], dtype=np.float64)
    k_matrix = np.array(
        [
            [float(camera["fx"]), 0.0, float(camera["cx"])],
            [0.0, float(camera["fy"]), float(camera["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pred_relation = auto_label.describe_spatial_relation(
        pred_corners,
        np.asarray(reference_corners_world, dtype=np.float64),
        np.linalg.inv(e_c2w),
        k_matrix,
    )
    result = {
        "evaluated": True,
        "direction_correct": pred_relation == expected_relation,
        "pred_relation": pred_relation,
        "expected_relation": expected_relation,
        "center_match": False,
        "center_l2_threshold_cm": float(center_l2_threshold_cm),
    }
    if center_l2_error_cm is not None:
        result["center_l2_error_cm"] = center_l2_error_cm
    return result


def merge_sample_metric_status(
    collision: Mapping[str, Any] | None,
    direction: Mapping[str, Any] | None,
    size: Mapping[str, Any] | None,
    object_center: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    用法: status = merge_sample_metric_status(collision, direction, size)
    作用: 合并三类 metric 的单样本通过状态
    输入: collision/direction/size/object_center: 单项 metric 结果，允许 None 表示未评估
    输出: dict，包含 placement_success 与各项是否参与评估
    """
    collision_evaluated = bool(collision and collision.get("evaluated"))
    direction_evaluated = bool(direction and direction.get("evaluated"))
    size_evaluated = bool(size and size.get("evaluated"))
    object_center_evaluated = bool(object_center and object_center.get("evaluated"))
    full_metric_evaluated = collision_evaluated and direction_evaluated and size_evaluated
    placement_success = (
        bool(collision and collision.get("collision_free"))
        and bool(direction and direction.get("direction_correct"))
        and bool(size and size.get("size_consistent"))
    )
    object_center_success = bool(object_center and object_center.get("center_match"))
    overall_metric_evaluated = full_metric_evaluated and object_center_evaluated
    return {
        "collision_evaluated": collision_evaluated,
        "direction_evaluated": direction_evaluated,
        "size_evaluated": size_evaluated,
        "object_center_evaluated": object_center_evaluated,
        "full_metric_evaluated": full_metric_evaluated,
        "overall_metric_evaluated": overall_metric_evaluated,
        "placement_success": bool(placement_success and full_metric_evaluated),
        "object_center_success": bool(object_center_success and object_center_evaluated),
        "overall_success": bool(placement_success and object_center_success and overall_metric_evaluated),
    }


def _safe_rate(values: Sequence[bool]) -> float | None:
    """
    用法: rate = _safe_rate([True, False])
    作用: 计算布尔列表通过率，空列表返回 None
    输入: values: bool 序列
    输出: float | None，通过率
    """
    if not values:
        return None
    return float(sum(1 for value in values if value) / len(values))


def _safe_mean(values: Sequence[float]) -> float | None:
    """
    用法: value = _safe_mean([1.0, 2.0])
    作用: 计算均值，空列表返回 None
    输入: values: float 序列
    输出: float | None
    """
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_median(values: Sequence[float]) -> float | None:
    """
    用法: value = _safe_median([1.0, 2.0])
    作用: 计算中位数，空列表返回 None
    输入: values: float 序列
    输出: float | None
    """
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def summarize_metric_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """
    用法: summary = summarize_metric_records(per_sample_records)
    作用: 汇总一组 per-sample metric 记录的覆盖率、通过率和误差统计
    输入: records: evaluate CLI 生成的单样本评测记录列表
    输出: dict，summary JSON 可序列化结果
    """
    sample_count = len(records)
    collision_items = [item for item in records if item.get("status", {}).get("collision_evaluated")]
    direction_items = [item for item in records if item.get("status", {}).get("direction_evaluated")]
    size_items = [item for item in records if item.get("status", {}).get("size_evaluated")]
    object_center_items = [item for item in records if item.get("status", {}).get("object_center_evaluated")]
    full_items = [item for item in records if item.get("status", {}).get("full_metric_evaluated")]
    overall_items = [item for item in records if item.get("status", {}).get("overall_metric_evaluated")]

    collision_free_values = [
        bool(item.get("collision", {}).get("collision_free"))
        for item in collision_items
    ]
    direction_correct_values = [
        bool(item.get("direction", {}).get("direction_correct"))
        for item in direction_items
    ]
    size_consistent_values = [
        bool(item.get("size", {}).get("size_consistent"))
        for item in size_items
    ]
    placement_success_values = [
        bool(item.get("status", {}).get("placement_success"))
        for item in full_items
    ]
    object_center_match_values = [
        bool(item.get("object_center", {}).get("center_match"))
        for item in object_center_items
    ]
    overall_success_values = [
        bool(item.get("status", {}).get("overall_success"))
        for item in overall_items
    ]
    occupied_collision_ratios = [
        float(item.get("collision", {}).get("occupied_collision_ratio", 0.0))
        for item in collision_items
        if item.get("collision", {}).get("occupied_collision_ratio") is not None
    ]
    unknown_overlap_ratios = [
        float(item.get("collision", {}).get("unknown_overlap_ratio", 0.0))
        for item in collision_items
        if item.get("collision", {}).get("unknown_overlap_ratio") is not None
    ]
    volume_errors_cm3 = [
        float(item.get("size", {}).get("volume_error_cm3", 0.0))
        for item in size_items
        if item.get("size", {}).get("volume_error_cm3") is not None
    ]
    volume_error_ratios = [
        float(item.get("size", {}).get("volume_error_ratio", 0.0))
        for item in size_items
        if item.get("size", {}).get("volume_error_ratio") is not None
    ]
    axis_size_error_sums_cm = [
        float(item.get("size", {}).get("axis_size_error_sum_cm", 0.0))
        for item in size_items
        if item.get("size", {}).get("axis_size_error_sum_cm") is not None
    ]
    return {
        "sample_count": sample_count,
        "full_metric_count": len(full_items),
        "overall_metric_count": len(overall_items),
        "collision_metric_count": len(collision_items),
        "direction_metric_count": len(direction_items),
        "size_metric_count": len(size_items),
        "object_center_metric_count": len(object_center_items),
        "full_metric_coverage": None if sample_count == 0 else float(len(full_items) / sample_count),
        "overall_metric_coverage": None if sample_count == 0 else float(len(overall_items) / sample_count),
        "collision_coverage": None if sample_count == 0 else float(len(collision_items) / sample_count),
        "direction_coverage": None if sample_count == 0 else float(len(direction_items) / sample_count),
        "size_coverage": None if sample_count == 0 else float(len(size_items) / sample_count),
        "object_center_coverage": None if sample_count == 0 else float(len(object_center_items) / sample_count),
        "placement_success_rate": _safe_rate(placement_success_values),
        "overall_success_rate": _safe_rate(overall_success_values),
        "collision_free_rate": _safe_rate(collision_free_values),
        "direction_correct_rate": _safe_rate(direction_correct_values),
        "size_consistent_rate": _safe_rate(size_consistent_values),
        "object_center_match_rate": _safe_rate(object_center_match_values),
        "mean_occupied_collision_ratio": _safe_mean(occupied_collision_ratios),
        "median_occupied_collision_ratio": _safe_median(occupied_collision_ratios),
        "mean_unknown_overlap_ratio": _safe_mean(unknown_overlap_ratios),
        "median_unknown_overlap_ratio": _safe_median(unknown_overlap_ratios),
        "mean_volume_error_cm3": _safe_mean(volume_errors_cm3),
        "median_volume_error_cm3": _safe_median(volume_errors_cm3),
        "mean_volume_error_ratio": _safe_mean(volume_error_ratios),
        "median_volume_error_ratio": _safe_median(volume_error_ratios),
        "mean_axis_size_error_sum_cm": _safe_mean(axis_size_error_sums_cm),
        "median_axis_size_error_sum_cm": _safe_median(axis_size_error_sums_cm),
    }


def summarize_by_source(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """
    用法: by_source = summarize_by_source(per_sample_records)
    作用: 按 source_name 分组汇总 metric 结果
    输入: records: 单样本评测记录列表
    输出: dict，source_name -> summary
    """
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in records:
        source_name = str(item.get("source_name", "unknown"))
        grouped.setdefault(source_name, []).append(item)
    return {
        source_name: summarize_metric_records(source_records)
        for source_name, source_records in sorted(grouped.items())
    }
