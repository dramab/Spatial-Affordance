"""
tests/test_coord_utils.py
-------------------------
职责：测试姿态角解析与放置姿态合理性判定。

测试内容：
- test_analyze_pose_orientation_flat_pose：验证平放姿态会被识别为合理
- test_analyze_pose_orientation_upright_pose：验证真正竖立姿态会被识别为合理
- test_analyze_pose_orientation_rejects_tilted_remote_pose：验证斜插遥控器这类姿态不会被误判为竖立
- test_analyze_pose_orientation_accepts_axis_aligned_middle_axis_pose：
  验证中间尺度轴竖直的姿态也会被识别为可保留姿态

用法：
    pytest tests/test_coord_utils.py -v
"""

import numpy as np

from src.utils.coord_utils import (
    analyze_pose_orientation,
    rotation_matrix_from_euler_zyx,
)


def _make_pose(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """
    构造仅用于测试的 4x4 位姿矩阵。

    输入:
        roll_deg: float roll 角度（度）
        pitch_deg: float pitch 角度（度）
        yaw_deg: float yaw 角度（度）
    输出:
        (4, 4) ndarray object→world 位姿矩阵
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation_matrix_from_euler_zyx(
        np.deg2rad(roll_deg),
        np.deg2rad(pitch_deg),
        np.deg2rad(yaw_deg),
    )
    return T


def _make_bbox(size_x: float, size_y: float, size_z: float) -> np.ndarray:
    """
    构造仅用于测试的 canonical AABB。

    输入:
        size_x: float X 轴尺寸
        size_y: float Y 轴尺寸
        size_z: float Z 轴尺寸
    输出:
        (6,) ndarray [min_x, min_y, min_z, max_x, max_y, max_z]
    """
    half = np.array([size_x, size_y, size_z], dtype=np.float64) / 2.0
    return np.array([-half[0], -half[1], -half[2], half[0], half[1], half[2]],
                    dtype=np.float64)


def test_analyze_pose_orientation_flat_pose():
    """
    验证平放姿态会被识别为合理姿态。

    输入:
        无，内部构造 roll/pitch 接近 0° 的位姿
    输出:
        无，通过断言验证结果
    """
    pose = _make_pose(2.0, -3.0, 45.0)
    bbox = _make_bbox(12.0, 8.0, 2.0)
    info = analyze_pose_orientation(
        pose,
        bbox,
        flat_threshold_deg=5.0,
        upright_threshold_deg=5.0,
    )

    assert bool(info["is_flat"]) is True
    assert bool(info["is_upright"]) is False
    assert bool(info["is_reasonable"]) is True


def test_analyze_pose_orientation_upright_pose():
    """
    验证真正竖立姿态会被识别为合理姿态。

    输入:
        无，内部构造一个 roll 接近 90° 且 pitch 接近 0° 的位姿
    输出:
        无，通过断言验证结果
    """
    pose = _make_pose(90.0, 2.0, 30.0)
    bbox = _make_bbox(2.0, 12.0, 2.0)
    info = analyze_pose_orientation(
        pose,
        bbox,
        flat_threshold_deg=5.0,
        upright_threshold_deg=5.0,
    )

    assert bool(info["is_flat"]) is False
    assert bool(info["is_upright"]) is True
    assert bool(info["is_reasonable"]) is True


def test_analyze_pose_orientation_rejects_tilted_remote_pose():
    """
    验证斜插遥控器这类姿态不会被误判为竖立姿态。

    输入:
        无，内部使用样本 remote-grey 的原始旋转矩阵
    输出:
        无，通过断言验证结果
    """
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.array([
        [-0.3736369689557053, 0.6121312233971954, 0.6969151890810263],
        [0.36354947409558036, -0.594577755994652, 0.7171534507768815],
        [0.8533622884299574, 0.5213181921040281, -0.000383753785208301],
    ], dtype=np.float64)
    bbox = _make_bbox(22.0, 3.0, 5.0)

    info = analyze_pose_orientation(
        pose,
        bbox,
        flat_threshold_deg=5.0,
        upright_threshold_deg=5.0,
    )

    assert bool(info["is_flat"]) is False
    assert bool(info["is_upright"]) is False
    assert bool(info["is_reasonable"]) is False


def test_analyze_pose_orientation_accepts_flat_bottle_sample():
    """
    验证视觉上平放的 bottle-eres_inox 不会因欧拉角分解而误判为倾斜。

    输入:
        无，内部使用样本 bottle-eres_inox 的原始旋转矩阵与 canonical AABB
    输出:
        无，通过断言验证结果
    """
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.array([
        [0.012359447067249974, 0.6779096292103588, 0.7350413448861643],
        [0.01462940798799423, -0.7351414059370607, 0.677755924871779],
        [0.9998165954264353, 0.002376531244415877, -0.019003357838330986],
    ], dtype=np.float64)
    bbox = np.array([
        -1.7522500827908516, -8.714800328016281, -4.072250053286552,
        1.7522500827908516, 8.714800328016281, 4.072250053286552,
    ], dtype=np.float64)

    info = analyze_pose_orientation(
        pose,
        bbox,
        flat_threshold_deg=15.0,
        upright_threshold_deg=15.0,
    )

    assert int(info["flat_axis_index"]) == 0
    assert int(info["vertical_axis_index"]) == 0
    assert bool(info["is_axis_aligned"]) is True
    assert bool(info["is_flat"]) is True
    assert bool(info["is_upright"]) is False
    assert bool(info["is_reasonable"]) is True


def test_analyze_pose_orientation_accepts_axis_aligned_middle_axis_pose():
    """
    验证当中间尺度轴竖直时，姿态仍会被识别为可保留姿态。

    输入:
        无，内部使用样本 cup-green_actys 的原始旋转矩阵与 canonical AABB
    输出:
        无，通过断言验证结果
    """
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.array([
        [-0.36858265550763186, -0.01955667608560278, 0.9293892416417485],
        [0.9291522909979096, 0.02310052084795846, 0.36897477701041403],
        [-0.028685295750162727, 0.9995418462260788, 0.00965667907149146],
    ], dtype=np.float64)
    bbox = np.array([
        -5.507149919867516, -4.461599886417389, -4.048449918627739,
        5.507149919867516, 4.461599886417389, 4.048449918627739,
    ], dtype=np.float64)

    info = analyze_pose_orientation(
        pose,
        bbox,
        flat_threshold_deg=5.0,
        upright_threshold_deg=5.0,
    )

    assert int(info["vertical_axis_index"]) == 1
    assert int(info["flat_axis_index"]) == 2
    assert int(info["upright_axis_index"]) == 0
    assert bool(info["is_axis_aligned"]) is True
    assert bool(info["is_flat"]) is False
    assert bool(info["is_upright"]) is False
    assert bool(info["is_reasonable"]) is True
