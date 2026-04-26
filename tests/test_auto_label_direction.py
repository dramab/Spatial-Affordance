import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.free_bbox.grid_ops import _get_bbox_corners
from src.utils.coord_utils import transform_points
from tools.auto_label import (
    describe_angle_relation,
    describe_horizontal_relation_by_depth,
    describe_horizontal_relation_by_pixel_angle,
    describe_spatial_relation,
    describe_vertical_relation,
    footprint_overlap_ratio,
    project_box_center_to_pixel,
)


def make_pose(center):
    """
    用法: pose = make_pose(center)
    作用: 构造仅包含平移的 object->world 测试变换。
    输入: center 为 3D 平移坐标。
    输出: (4,4) 齐次变换矩阵。
    """
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(center, dtype=np.float64)
    return pose


def make_box_corners(center, bbox=None):
    """
    用法: corners = make_box_corners(center, bbox)
    作用: 构造 canonical box 经过 world 变换后的真实角点。
    输入: center 为 world 中心平移；bbox 为可选 canonical AABB。
    输出: (8,3) 世界坐标 box 角点。
    """
    if bbox is None:
        bbox = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    return transform_points(_get_bbox_corners(np.asarray(bbox, dtype=np.float64)), make_pose(center))


def make_camera():
    """
    用法: E_w2c, K = make_camera()
    作用: 构造用于像素方向测试的简单针孔相机。
    输入: 无。
    输出: world->camera 外参和 3x3 内参。
    """
    E_w2c = np.eye(4, dtype=np.float64)
    K = np.array(
        [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return E_w2c, K


@pytest.mark.parametrize(
    ("target_center", "relation"),
    [
        ([0.0, 0.0, 2.0], "the top of"),
        ([0.0, 0.0, -2.0], "below"),
    ],
)
def test_describe_vertical_relation_uses_real_world_boxes(target_center, relation):
    """
    用法: pytest tests/test_auto_label_direction.py
    作用: 验证上下关系来自 canonical box 角点经 transform 后的真实 world box。
    输入: 目标物中心和期望上下关系。
    输出: 断言方向关系正确。
    """
    target_corners = make_box_corners(target_center)
    ref_corners = make_box_corners([0.0, 0.0, 0.0])

    assert describe_vertical_relation(target_corners, ref_corners) == relation


def test_describe_vertical_relation_requires_xy_footprint_overlap():
    """
    用法: pytest tests/test_auto_label_direction.py
    作用: 验证没有 XY 足迹重叠时不会把高度差误判为上下关系。
    输入: 横向分离但 Z 方向位于上方的两个 box。
    输出: 断言上下关系为 None。
    """
    target_corners = make_box_corners([5.0, 0.0, 2.0])
    ref_corners = make_box_corners([0.0, 0.0, 0.0])

    assert footprint_overlap_ratio(target_corners, ref_corners) == 0.0
    assert describe_vertical_relation(target_corners, ref_corners) is None


def test_describe_vertical_relation_allows_adaptive_contact_tolerance():
    """
    用法: pytest tests/test_auto_label_direction.py
    作用: 验证轻微 Z 向穿插仍可通过自适应容差判为上下关系。
    输入: 目标物底面略低于参照物顶面的两个重叠 box。
    输出: 断言目标物仍被判为在参照物上方。
    """
    target_corners = make_box_corners([0.0, 0.0, 1.5])
    ref_corners = make_box_corners([0.0, 0.0, 0.0])

    assert describe_vertical_relation(target_corners, ref_corners) == "the top of"


def test_describe_vertical_relation_allows_bounded_bbox_penetration():
    """
    用法: pytest tests/test_auto_label_direction.py
    作用: 验证支撑关系允许有限 bbox 穿插，适配 bowl 等非实体顶面包围盒。
    输入: 目标底面低于参照物顶面约 2cm，但中心高度明显更高。
    输出: 断言目标物仍被判为在参照物上方。
    """
    bbox = np.array([-3.0, -3.0, -3.0, 3.0, 3.0, 3.0], dtype=np.float64)
    target_corners = make_box_corners([0.0, 0.0, 4.0], bbox=bbox)
    ref_corners = make_box_corners([0.0, 0.0, 0.0], bbox=bbox)

    assert describe_vertical_relation(target_corners, ref_corners) == "the top of"


def test_describe_vertical_relation_rejects_excessive_bbox_penetration():
    """
    用法: pytest tests/test_auto_label_direction.py
    作用: 验证穿插超过上限时不会误判为上下支撑关系。
    输入: 目标底面低于参照物顶面的距离超过允许穿插上限。
    输出: 断言上下关系为 None。
    """
    bbox = np.array([-3.0, -3.0, -3.0, 3.0, 3.0, 3.0], dtype=np.float64)
    target_corners = make_box_corners([0.0, 0.0, 3.6], bbox=bbox)
    ref_corners = make_box_corners([0.0, 0.0, 0.0], bbox=bbox)

    assert describe_vertical_relation(target_corners, ref_corners) is None


@pytest.mark.parametrize(
    ("target_center", "relation"),
    [
        ([2.0, 0.0, 10.0], "the right of"),
        ([2.0, 2.0, 10.0], "the front right of"),
        ([0.0, 2.0, 10.0], "in front of"),
        ([-2.0, 2.0, 10.0], "the front left of"),
        ([-2.0, 0.0, 10.0], "the left of"),
        ([-2.0, -2.0, 10.0], "the back left of"),
        ([0.0, -2.0, 10.0], "behind"),
        ([2.0, -2.0, 10.0], "the back right of"),
    ],
)
def test_describe_horizontal_relation_uses_projected_center_angle(target_center, relation):
    """
    用法: pytest tests/test_auto_label_direction.py
    作用: 验证水平 8 方向由两个 box 中心投影到像素后的向量角度决定。
    输入: 目标中心和期望方向。
    输出: 断言方向关系正确。
    """
    E_w2c, K = make_camera()
    target_corners = make_box_corners(target_center, bbox=[-0.2, -0.2, -0.2, 0.2, 0.2, 0.2])
    ref_corners = make_box_corners([0.0, 0.0, 10.0], bbox=[-0.2, -0.2, -0.2, 0.2, 0.2, 0.2])

    actual = describe_horizontal_relation_by_pixel_angle(target_corners, ref_corners, E_w2c, K)

    assert actual == relation


@pytest.mark.parametrize(
    ("angle_deg", "relation"),
    [
        (-15.0, "the right of"),
        (0.0, "the right of"),
        (15.0, "the right of"),
        (16.0, "the front right of"),
        (74.0, "the front right of"),
        (75.0, "in front of"),
        (90.0, "in front of"),
        (105.0, "in front of"),
        (106.0, "the front left of"),
        (164.0, "the front left of"),
        (165.0, "the left of"),
        (179.0, "the left of"),
        (-179.0, "the left of"),
        (-165.0, "the back left of"),
        (-164.0, "the back left of"),
        (-106.0, "the back left of"),
        (-105.0, "behind"),
        (-90.0, "behind"),
        (-75.0, "behind"),
        (-74.0, "the back right of"),
        (-16.0, "the back right of"),
    ],
)
def test_describe_angle_relation_uses_wide_combined_sectors(angle_deg, relation):
    """
    用法: pytest tests/test_auto_label_direction.py
    作用: 验证单轴方向为 30 度总宽、组合方向为 60 度总宽的非均匀扇区。
    输入: 像素向量角度和期望方向。
    输出: 断言角度映射关系正确。
    """
    assert describe_angle_relation(angle_deg) == relation


@pytest.mark.parametrize(
    ("target_center", "ref_center", "relation"),
    [
        ([5.0, 0.0, 25.0], [0.0, 0.0, 10.0], "the back right of"),
        ([5.0, 0.0, 21.0], [0.0, 0.0, 20.0], "the right of"),
        ([-5.0, 0.0, 10.0], [0.0, 0.0, 25.0], "the front left of"),
    ],
)
def test_describe_spatial_relation_uses_camera_depth_for_front_back(
        target_center, ref_center, relation):
    """
    用法: pytest tests/test_auto_label_direction.py
    作用: 验证主流程使用相机深度差判断前后，而不是使用像素竖直方向。
    输入: 横向分离的 target/ref 中心，避免触发上下关系。
    输出: 断言深度感知方向关系正确。
    """
    E_w2c, K = make_camera()
    target_corners = make_box_corners(target_center, bbox=[-0.2, -0.2, -0.2, 0.2, 0.2, 0.2])
    ref_corners = make_box_corners(ref_center, bbox=[-0.2, -0.2, -0.2, 0.2, 0.2, 0.2])

    actual = describe_spatial_relation(target_corners, ref_corners, E_w2c, K)

    assert actual == relation


def test_describe_horizontal_relation_by_depth_returns_near_for_small_offsets():
    """
    用法: pytest tests/test_auto_label_direction.py
    作用: 验证横向和深度差都不足阈值时返回 near。
    输入: 投影中心横向偏移和相机深度差均很小的两个 box。
    输出: 断言关系为 near。
    """
    E_w2c, K = make_camera()
    target_corners = make_box_corners([0.05, 0.0, 10.5], bbox=[-0.2, -0.2, -0.2, 0.2, 0.2, 0.2])
    ref_corners = make_box_corners([0.0, 0.0, 10.0], bbox=[-0.2, -0.2, -0.2, 0.2, 0.2, 0.2])

    actual = describe_horizontal_relation_by_depth(target_corners, ref_corners, E_w2c, K)

    assert actual == "near"


def test_describe_spatial_relation_prefers_vertical_before_pixel_angle():
    """
    用法: pytest tests/test_auto_label_direction.py
    作用: 验证上下关系优先于水平像素角度关系。
    输入: 目标物在参照物正上方且像素中心近似重合。
    输出: 断言最终关系为 the top of。
    """
    E_w2c, K = make_camera()
    target_corners = make_box_corners([0.0, 0.0, 2.0])
    ref_corners = make_box_corners([0.0, 0.0, 0.0])

    assert describe_spatial_relation(target_corners, ref_corners, E_w2c, K) == "the top of"


def test_project_box_center_to_pixel_uses_transformed_canonical_center():
    """
    用法: pytest tests/test_auto_label_direction.py
    作用: 验证像素投影使用 transform 后的 canonical box 真实中心。
    输入: 非原点 canonical box、平移矩阵和简单相机。
    输出: 断言中心像素坐标正确。
    """
    E_w2c, K = make_camera()
    bbox = np.array([0.0, 0.0, 0.0, 2.0, 4.0, 2.0], dtype=np.float64)
    corners = make_box_corners([9.0, 18.0, 9.0], bbox=bbox)

    center_uv = project_box_center_to_pixel(corners, E_w2c, K)

    np.testing.assert_allclose(center_uv, np.array([100.0, 200.0]))
