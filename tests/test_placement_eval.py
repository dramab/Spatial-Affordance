"""
tests/test_placement_eval.py
----------------------------
职责：测试 placement prediction metric 的核心几何计算。

测试内容：
- test_evaluate_collision_ignores_two_support_layers_from_gt：
  验证碰撞评测可按 GT 最低体素层忽略两层桌面 occupied
- test_evaluate_collision_uses_ratio_threshold_after_support_ignore：
  验证忽略支撑层后按 occupied collision ratio 阈值判定碰撞
- test_evaluate_collision_reports_unknown_overlap：
  验证 UNKNOWN 体素只记录覆盖比例，不按碰撞失败处理
- test_evaluate_size_consistency_thresholds：
  验证体积相对误差阈值
- test_evaluate_center_alignment_uses_l2_threshold：
  验证中心点评测复用 L2 阈值逻辑
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
    evaluate_center_alignment,
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


def test_evaluate_collision_ignores_two_support_layers_from_gt():
    """
    作用：验证根据 GT 最低体素层向下忽略两层桌面 occupied。

    输入：
        无，内部构造预测框和 occupancy grid
    输出：
        无，通过断言验证结果
    """
    pred_box = [2.5, 2.5, 2.5, 2.0, 2.0, 5.0, 0.0]
    gt_box = [2.5, 2.5, 4.0, 2.0, 2.0, 4.0, 0.0]
    grid, voxel_params = _make_grid()
    grid[2, 2, 0] = OCCUPIED
    grid[2, 2, 1] = OCCUPIED

    result = evaluate_collision(
        pred_box,
        grid,
        voxel_params,
        target_box_world=gt_box,
        support_ignore_layers=2,
    )

    assert result["evaluated"] is True
    assert result["collision_free"] is True
    assert result["pred_voxel_count"] > 0
    assert result["occupied_voxel_count"] == 0
    assert result["ignored_support_layers"] == [1, 0]
    assert result["ignored_support_occupied_count"] == 2
    assert result["occupied_collision_ratio"] == 0.0


def test_evaluate_collision_uses_ratio_threshold_after_support_ignore():
    """
    作用：验证桌面支撑层之外的 OCCUPIED 会按碰撞比例阈值决定是否失败。

    输入：
        无，内部构造预测框、GT 框和 occupancy grid
    输出：
        无，通过断言验证结果
    """
    pred_box = [2.5, 2.5, 2.5, 2.0, 2.0, 5.0, 0.0]
    gt_box = [2.5, 2.5, 4.0, 2.0, 2.0, 4.0, 0.0]
    grid, voxel_params = _make_grid()
    grid[2, 2, 0] = OCCUPIED
    grid[2, 2, 1] = OCCUPIED
    grid[2, 2, 2] = OCCUPIED

    pass_result = evaluate_collision(
        pred_box,
        grid,
        voxel_params,
        collision_ratio_threshold=0.1,
        target_box_world=gt_box,
        support_ignore_layers=2,
    )
    fail_result = evaluate_collision(
        pred_box,
        grid,
        voxel_params,
        collision_ratio_threshold=0.01,
        target_box_world=gt_box,
        support_ignore_layers=2,
    )

    assert pass_result["evaluated"] is True
    assert pass_result["collision_free"] is True
    assert pass_result["occupied_voxel_count"] == 1
    assert pass_result["ignored_support_occupied_count"] == 2
    assert pass_result["occupied_collision_ratio"] > 0.01
    assert pass_result["occupied_collision_ratio"] <= 0.1

    assert fail_result["evaluated"] is True
    assert fail_result["collision_free"] is False
    assert fail_result["occupied_voxel_count"] == 1
    assert fail_result["ignored_support_occupied_count"] == 2
    assert fail_result["occupied_collision_ratio"] == pass_result["occupied_collision_ratio"]


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
    作用：验证尺寸一致性按体积相对误差阈值判断。

    输入：
        无，内部构造预测和 GT box
    输出：
        无，通过断言验证结果
    """
    gt_box = [0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 0.0]
    good_pred = [0.0, 0.0, 0.0, 11.0, 20.0, 30.0, 0.0]
    bad_pred = [0.0, 0.0, 0.0, 13.0, 20.0, 30.0, 0.0]

    good = evaluate_size_consistency(good_pred, gt_box, volume_error_ratio_threshold=0.15)
    bad = evaluate_size_consistency(bad_pred, gt_box, volume_error_ratio_threshold=0.15)

    assert good["size_consistent"] is True
    assert bad["size_consistent"] is False
    assert np.isclose(good["pred_volume_cm3"], 6600.0)
    assert np.isclose(good["target_volume_cm3"], 6000.0)
    assert np.isclose(good["volume_error_ratio"], 0.1)
    assert bad["volume_error_ratio"] > bad["volume_error_ratio_threshold"]


def test_evaluate_center_alignment_uses_l2_threshold():
    """
    作用：验证移动前物体中心评测使用 L2 距离阈值。

    输入：
        无，内部构造预测中心和 GT 中心
    输出：
        无，通过断言验证中心匹配结果
    """
    good = evaluate_center_alignment(
        pred_center_world=[1.0, 2.0, 3.0],
        target_center_world=[1.0, 2.0, 3.5],
        center_l2_threshold_cm=1.0,
    )
    bad = evaluate_center_alignment(
        pred_center_world=[1.0, 2.0, 3.0],
        target_center_world=[1.0, 2.0, 5.0],
        center_l2_threshold_cm=1.0,
    )

    assert good["evaluated"] is True
    assert good["center_match"] is True
    assert np.isclose(good["center_l2_error_cm"], 0.5)
    assert bad["center_match"] is False


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


def test_evaluate_direction_accepts_close_center_before_relation():
    """
    作用：验证预测中心与 GT 中心足够接近时，方向 metric 直接判定正确。

    输入：
        无，内部构造中心接近但期望方向不同的预测
    输出：
        无，通过断言验证中心直通逻辑
    """
    ref_obj = _make_object("ref", (0.0, 0.0, 10.0))
    pred_box = [0.5, 0.5, 10.5, 0.4, 0.4, 0.4, 0.0]
    gt_box = [0.0, 0.0, 10.0, 0.4, 0.4, 0.4, 0.0]
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
        expected_relation="behind",
        target_box_world=gt_box,
        center_l2_threshold_cm=1.0,
    )

    assert result["evaluated"] is True
    assert result["direction_correct"] is True
    assert result["center_match"] is True
    assert result["pred_relation"] == "center_match"


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
    first_size = {
        "evaluated": True,
        "size_consistent": True,
        "volume_error_cm3": 300.0,
        "volume_error_ratio": 0.05,
    }
    first_object_center = {
        "evaluated": True,
        "center_match": True,
        "center_l2_error_cm": 1.0,
    }
    second_collision = {
        "evaluated": True,
        "collision_free": False,
        "occupied_collision_ratio": 0.2,
        "unknown_overlap_ratio": 0.1,
    }
    second_direction = {"evaluated": True, "direction_correct": False}
    second_size = {
        "evaluated": True,
        "size_consistent": True,
        "volume_error_cm3": 120.0,
        "volume_error_ratio": 0.02,
    }
    second_object_center = {
        "evaluated": True,
        "center_match": False,
        "center_l2_error_cm": 3.0,
    }
    records = [
        {
            "source_name": "demo",
            "collision": first_collision,
            "direction": first_direction,
            "size": first_size,
            "object_center": first_object_center,
            "status": merge_sample_metric_status(
                first_collision,
                first_direction,
                first_size,
                first_object_center,
            ),
        },
        {
            "source_name": "demo",
            "collision": second_collision,
            "direction": second_direction,
            "size": second_size,
            "object_center": second_object_center,
            "status": merge_sample_metric_status(
                second_collision,
                second_direction,
                second_size,
                second_object_center,
            ),
        },
    ]

    summary = summarize_metric_records(records)

    assert summary["sample_count"] == 2
    assert summary["full_metric_coverage"] == 1.0
    assert summary["overall_metric_coverage"] == 1.0
    assert summary["placement_success_rate"] == 0.5
    assert summary["overall_success_rate"] == 0.5
    assert summary["collision_free_rate"] == 0.5
    assert summary["direction_correct_rate"] == 0.5
    assert summary["size_consistent_rate"] == 1.0
    assert summary["object_center_match_rate"] == 0.5
    assert np.isclose(summary["mean_occupied_collision_ratio"], 0.1)
    assert np.isclose(summary["mean_unknown_overlap_ratio"], 0.05)
    assert np.isclose(summary["mean_volume_error_cm3"], 210.0)
    assert np.isclose(summary["mean_volume_error_ratio"], 0.035)
    assert np.isclose(summary["mean_object_center_l2_error_cm"], 2.0)
