"""
tests/test_placement_eval.py
----------------------------
职责：测试 placement prediction metric 的核心几何计算。

测试内容：
- test_evaluate_collision_uses_occupancy_without_target_removal：
  验证碰撞评测直接使用 occupancy grid，不移除目标物体原位置
- test_evaluate_collision_reports_unknown_overlap：
  验证 UNKNOWN 体素只记录覆盖比例，不按碰撞失败处理
- test_evaluate_size_consistency_thresholds：
  验证尺寸一致性阈值
- test_evaluate_direction_matches_auto_label_relation：
  验证方向 metric 复用 auto_label 空间关系逻辑
- test_summarize_metric_records_reports_rates：
  验证 summary 覆盖率和通过率统计

用法：
    pytest tests/test_placement_eval.py -v
"""

from __future__ import annotations

import numpy as np

from src.annotation.free_bbox.datatypes import ObjectInfo
from src.annotation.free_bbox.occupancy import OCCUPIED, UNKNOWN
from src.metrics.placement_eval import (
    evaluate_collision,
    evaluate_direction,
    evaluate_size_consistency,
    merge_sample_metric_status,
    object_info_to_corners_world,
    summarize_metric_records,
)


def _make_grid(shape: tuple[int, int, int] = (6, 6, 6)) -> tuple[np.ndarray, dict[str, object]]:
    """
    用法: grid, voxel_params = _make_grid()
    作用: 构造测试用 FREE occupancy grid 和体素参数
    输入: shape: grid 三维形状
    输出: tuple(grid, voxel_params)
    """
    grid = np.zeros(shape, dtype=np.uint8)
    voxel_params = {"origin": [0.0, 0.0, 0.0], "voxel_size": 1.0}
    return grid, voxel_params


def _make_object(obj_id: str, center: tuple[float, float, float]) -> ObjectInfo:
    """
    用法: obj = _make_object("obj_1", (0, 0, 0))
    作用: 构造测试用轴对齐 ObjectInfo
    输入: obj_id: 物体 ID；center: world 平移中心
    输出: ObjectInfo
    """
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(center, dtype=np.float64)
    return ObjectInfo(
        obj_id=obj_id,
        class_name="box",
        bbox3d_canonical=np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        pose_world=pose,
    )


def test_evaluate_collision_uses_occupancy_without_target_removal():
    """
    作用：验证碰撞评测直接使用 occupancy grid，因此目标物体原位置若为 OCCUPIED 也会计入碰撞。

    输入：
        无，内部构造预测框和 occupancy grid
    输出：
        无，通过断言验证结果
    """
    pred_box = [2.5, 2.5, 2.5, 2.0, 2.0, 2.0, 0.0]
    grid, voxel_params = _make_grid()
    grid[2, 2, 2] = OCCUPIED

    result = evaluate_collision(pred_box, grid, voxel_params)

    assert result["evaluated"] is True
    assert result["collision_free"] is False
    assert result["pred_voxel_count"] > 0
    assert result["occupied_voxel_count"] == 1
    assert result["occupied_collision_ratio"] > result["collision_ratio_threshold"]


def test_evaluate_collision_reports_unknown_overlap():
    """
    作用：验证 UNKNOWN 体素覆盖会被记录，但不会作为 OCCUPIED 碰撞失败。

    输入：
        无，内部构造含 UNKNOWN 体素的 occupancy grid
    输出：
        无，通过断言验证结果
    """
    pred_box = [2.5, 2.5, 2.5, 2.0, 2.0, 2.0, 0.0]
    grid, voxel_params = _make_grid()
    grid[2, 2, 2] = UNKNOWN

    result = evaluate_collision(pred_box, grid, voxel_params)

    assert result["evaluated"] is True
    assert result["collision_free"] is True
    assert result["occupied_voxel_count"] == 0
    assert result["unknown_voxel_count"] == 1
    assert result["unknown_overlap_ratio"] > 0.0


def test_evaluate_size_consistency_thresholds():
    """
    作用：验证尺寸一致性同时受平均相对误差和单轴相对误差约束。

    输入：
        无，内部构造预测和 GT box
    输出：
        无，通过断言验证结果
    """
    gt_box = [0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 0.0]
    good_pred = [0.0, 0.0, 0.0, 10.5, 19.0, 31.0, 0.0]
    bad_pred = [0.0, 0.0, 0.0, 10.0, 20.0, 36.0, 0.0]

    good = evaluate_size_consistency(good_pred, gt_box)
    bad = evaluate_size_consistency(bad_pred, gt_box)

    assert good["size_consistent"] is True
    assert bad["size_consistent"] is False
    assert bad["max_axis_relative_size_error"] > bad["max_axis_relative_threshold"]


def test_evaluate_direction_matches_auto_label_relation():
    """
    作用：验证方向 metric 会按 auto_label 关系逻辑判断预测位置。

    输入：
        无，内部构造参考物、预测 box 和相机
    输出：
        无，通过断言验证方向关系
    """
    ref_obj = _make_object("ref", (0.0, 0.0, 10.0))
    pred_box = [2.0, 0.0, 10.0, 0.4, 0.4, 0.4, 0.0]
    camera = {
        "fx": 100.0,
        "fy": 100.0,
        "cx": 0.0,
        "cy": 0.0,
        "E_c2w": np.eye(4, dtype=np.float64).tolist(),
    }

    result = evaluate_direction(
        pred_box_world=pred_box,
        reference_corners_world=object_info_to_corners_world(ref_obj),
        camera=camera,
        expected_relation="the right of",
    )

    assert result["evaluated"] is True
    assert result["pred_relation"] == "the right of"
    assert result["direction_correct"] is True


def test_summarize_metric_records_reports_rates():
    """
    作用：验证 summary 会正确统计覆盖率和通过率。

    输入：
        无，内部构造两条 per-sample metric 记录
    输出：
        无，通过断言验证 summary 字段
    """
    first_collision = {
        "evaluated": True,
        "collision_free": True,
        "occupied_collision_ratio": 0.0,
        "unknown_overlap_ratio": 0.0,
    }
    first_direction = {"evaluated": True, "direction_correct": True}
    first_size = {"evaluated": True, "size_consistent": True, "mean_relative_size_error": 0.05, "max_axis_relative_size_error": 0.08}
    second_collision = {
        "evaluated": True,
        "collision_free": False,
        "occupied_collision_ratio": 0.2,
        "unknown_overlap_ratio": 0.1,
    }
    second_direction = {"evaluated": True, "direction_correct": False}
    second_size = {"evaluated": True, "size_consistent": True, "mean_relative_size_error": 0.02, "max_axis_relative_size_error": 0.04}
    records = [
        {
            "source_name": "demo",
            "collision": first_collision,
            "direction": first_direction,
            "size": first_size,
            "status": merge_sample_metric_status(first_collision, first_direction, first_size),
        },
        {
            "source_name": "demo",
            "collision": second_collision,
            "direction": second_direction,
            "size": second_size,
            "status": merge_sample_metric_status(second_collision, second_direction, second_size),
        },
    ]

    summary = summarize_metric_records(records)

    assert summary["sample_count"] == 2
    assert summary["full_metric_coverage"] == 1.0
    assert summary["placement_success_rate"] == 0.5
    assert summary["collision_free_rate"] == 0.5
    assert summary["direction_correct_rate"] == 0.5
    assert summary["size_consistent_rate"] == 1.0
    assert np.isclose(summary["mean_occupied_collision_ratio"], 0.1)
    assert np.isclose(summary["mean_unknown_overlap_ratio"], 0.05)
