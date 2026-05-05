"""
src/metrics/placement_eval.py
-----------------------------
放置预测结果评测工具函数。

提供三类可复用指标：
1. 预测 3D box 是否与上游 occupancy grid 中的 OCCUPIED 体素碰撞
2. 预测放置方向是否符合结构化指令关系
3. 预测 3D box 体积是否与目标物体体积一致

用法：
    from src.metrics.placement_eval import (
        evaluate_collision,
        evaluate_direction,
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
from src.utils.coord_utils import box7d_to_corners_world, rotation_z_3x3, transform_points


DEFAULT_COLLISION_RATIO_THRESHOLD = 0.003
DEFAULT_VOLUME_ERROR_RATIO_THRESHOLD = 0.1
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


def object_info_to_corners_world(obj: ObjectInfo) -> np.ndarray:
    """
    用法: corners = object_info_to_corners_world(scene.objects[0])
    作用: 将 ObjectInfo 的 canonical AABB 和 object->world 位姿转换为世界坐标角点
    输入: obj: ObjectInfo，包含 bbox3d_canonical 和 pose_world
    输出: ndarray(8,3)，世界坐标 3D box 角点
    """
    corners_object = get_bbox_corners(np.asarray(obj.bbox3d_canonical, dtype=np.float64))
    return transform_points(corners_object, np.asarray(obj.pose_world, dtype=np.float64))


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
) -> dict[str, Any]:
    """
    用法: result = evaluate_size_consistency(pred_box, gt_box)
    作用: 评估预测 box 体积是否与目标 box 体积一致
    输入:
        pred_box_world: 长度 7 的预测世界坐标 box
        target_box_world: 长度 7 的 GT 世界坐标 box
        volume_error_ratio_threshold: 预测体积相对 GT 体积误差阈值
    输出: dict，包含 size_consistent、pred_volume_cm3、target_volume_cm3、
         volume_error_cm3、volume_error_ratio、volume_error_ratio_threshold
    """
    pred_box = as_box7d(pred_box_world, "pred_box_world")
    target_box = as_box7d(target_box_world, "target_box_world")
    pred_volume_cm3 = float(np.prod(pred_box[3:6]))
    target_volume_cm3 = float(np.prod(target_box[3:6]))
    volume_error_cm3 = float(abs(pred_volume_cm3 - target_volume_cm3))
    volume_error_ratio = float(volume_error_cm3 / target_volume_cm3)
    size_consistent = bool(volume_error_ratio <= float(volume_error_ratio_threshold))
    return {
        "evaluated": True,
        "size_consistent": size_consistent,
        "pred_volume_cm3": pred_volume_cm3,
        "target_volume_cm3": target_volume_cm3,
        "volume_error_cm3": volume_error_cm3,
        "volume_error_ratio": volume_error_ratio,
        "volume_error_ratio_threshold": float(volume_error_ratio_threshold),
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
) -> dict[str, Any]:
    """
    用法: status = merge_sample_metric_status(collision, direction, size)
    作用: 合并三类 metric 的单样本通过状态
    输入: collision/direction/size: 单项 metric 结果，允许 None 表示未评估
    输出: dict，包含 placement_success 与各项是否参与评估
    """
    collision_evaluated = bool(collision and collision.get("evaluated"))
    direction_evaluated = bool(direction and direction.get("evaluated"))
    size_evaluated = bool(size and size.get("evaluated"))
    full_metric_evaluated = collision_evaluated and direction_evaluated and size_evaluated
    placement_success = (
        bool(collision and collision.get("collision_free"))
        and bool(direction and direction.get("direction_correct"))
        and bool(size and size.get("size_consistent"))
    )
    return {
        "collision_evaluated": collision_evaluated,
        "direction_evaluated": direction_evaluated,
        "size_evaluated": size_evaluated,
        "full_metric_evaluated": full_metric_evaluated,
        "placement_success": bool(placement_success and full_metric_evaluated),
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
    full_items = [item for item in records if item.get("status", {}).get("full_metric_evaluated")]

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

    return {
        "sample_count": sample_count,
        "full_metric_count": len(full_items),
        "collision_metric_count": len(collision_items),
        "direction_metric_count": len(direction_items),
        "size_metric_count": len(size_items),
        "full_metric_coverage": None if sample_count == 0 else float(len(full_items) / sample_count),
        "collision_coverage": None if sample_count == 0 else float(len(collision_items) / sample_count),
        "direction_coverage": None if sample_count == 0 else float(len(direction_items) / sample_count),
        "size_coverage": None if sample_count == 0 else float(len(size_items) / sample_count),
        "placement_success_rate": _safe_rate(placement_success_values),
        "collision_free_rate": _safe_rate(collision_free_values),
        "direction_correct_rate": _safe_rate(direction_correct_values),
        "size_consistent_rate": _safe_rate(size_consistent_values),
        "mean_occupied_collision_ratio": _safe_mean(occupied_collision_ratios),
        "median_occupied_collision_ratio": _safe_median(occupied_collision_ratios),
        "mean_unknown_overlap_ratio": _safe_mean(unknown_overlap_ratios),
        "median_unknown_overlap_ratio": _safe_median(unknown_overlap_ratios),
        "mean_volume_error_cm3": _safe_mean(volume_errors_cm3),
        "median_volume_error_cm3": _safe_median(volume_errors_cm3),
        "mean_volume_error_ratio": _safe_mean(volume_error_ratios),
        "median_volume_error_ratio": _safe_median(volume_error_ratios),
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
