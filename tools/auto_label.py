#!/usr/bin/env python3
"""
tools/auto_label.py
-------------------
以渲染好的可视化图片为索引，自动为 placement 空位框样本生成自然语言标注。
结合 3D 物理测距、真实 world box 上下关系判断与 2D 像素中心角度，生成空间关系描述。

======================== 用法示例 ========================
python tools/auto_label.py \
    --image-dir /data/jiajun.xie/Spatial-Affordance/outputs/placement_rgb_bbox_vis \
    --outputs-base /data/jiajun.xie/Spatial-Affordance/outputs \
    --mapping configs/annotation/mappingv2.json
    --output-dir /data/jiajun.xie/Spatial-Affordance/outputs/auto_labels \
    --limit 50

python tools/auto_label.py \
    --image-dir /data/jiajun.xie/Spatial-Affordance/outputs/placement_rgb_bbox_vis \
    --outputs-base /data/jiajun.xie/Spatial-Affordance/outputs \
    --mapping configs/annotation/mappingv2.json
    --output-dir /data/jiajun.xie/Spatial-Affordance/outputs/auto_labels_selected \
    --sample-ids scene_0000_0000_obj_3_p000 scene_0000_0000_obj_8_p000

python tools/auto_label.py \
    --image-dir /data/jiajun.xie/Spatial-Affordance/outputs/placement_rgb_bbox_vis \
    --outputs-base /data/jiajun.xie/Spatial-Affordance/outputs \
    --output-dir /data/jiajun.xie/Spatial-Affordance/outputs/auto_labels_v2 \
    --mapping configs/annotation/mappingv2.json

======================== 参数说明 ========================
--image-dir:    渲染好的 RGB 图片目录（作为数据驱动的基准，图片名需符合 {source}__{id}.png 规范）
--outputs-base: JSON 等原始数据的根目录（脚本会去这里找 source_name 对应的 samples/ 和 categories/）
--output-dir:   all_labels.json 和 report.html 的统一输出目录
--mapping:      (可选) 类别名称映射文件路径，默认使用 configs/annotation/mapping.json
--limit:        (可选) 限制处理的图片数量，方便快速测试
--sample-ids:   (可选) 指定一个或多个 sample_id，仅生成这些样本
--sample-ids-file: (可选) 从文本文件读取 sample_id，每行一个，支持与 --sample-ids 同时使用
--overwrite:    (可选) 兼容旧调用，当前 JSON/HTML 输出会直接刷新

======================== 输出内容 ========================
1. all_labels.json，汇总所有成功标注的信息。
2. report.html，单文件离线网页，引用图片和文本，可通过 Web 服务查看。
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import yaml

# 请确保项目根目录正确
PROJECT_ROOT = Path("/data/jiajun.xie/Spatial-Affordance")
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.free_bbox.grid_ops import _get_bbox_corners
from src.utils.coord_utils import project_world, transform_points
from src.datasets.hope_adapter import HopeAdapter
from src.datasets.housecat6d_adapter import HouseCat6DAdapter
from src.datasets.ycb_video_adapter import YCBVideoAdapter
from src.datasets.scannet_adapter import ScanNetAdapter
from src.datasets.dopose_adapter import DoPoseAdapter
from src.annotation.free_bbox.datatypes import SceneData, ObjectInfo, CameraParams

# ===================== 全局配置与缓存 =====================
LABEL_TEMPLATE = "Move {object_name} located at {rel_original} {ref_a_name} to {rel_placement} {ref_b_name}."
MAPPING_PATH = '/data/jiajun.xie/Spatial-Affordance/configs/annotation/mapping.json'

# 上下关系判定阈值：XY 足迹重叠足够大时，才把 Z 方向差异解释为上下关系
VERTICAL_FOOTPRINT_OVERLAP_RATIO = 0.50
VERTICAL_TOLERANCE_RATIO = 0.10
MIN_VERTICAL_TOLERANCE = 1.0
VERTICAL_MAX_PENETRATION_RATIO = 0.35
MAX_VERTICAL_PENETRATION = 3.0
VERTICAL_CENTER_SEPARATION_RATIO = 0.20
MIN_VERTICAL_CENTER_SEPARATION = 0.5
HORIZONTAL_CENTER_EPS_PX = 1e-6
AXIS_DIRECTION_HALF_WIDTH_DEG = 15.0
DEPTH_DIRECTION_MIN_CM = 5.0
DEPTH_DIRECTION_EXTENT_RATIO = 0.20
LATERAL_DIRECTION_MIN_PX = 8.0
MIN_VISIBILITY_RATIO = 0.4
MAX_OCCLUSION_RATIO = 0.5
SMALL_IMAGE_AREA_THRESHOLD = 2500
LARGE_IMAGE_AREA_THRESHOLD = 5000
MAX_REFERENCE_CANDIDATES = 3

GLOBAL_MAPPING_CACHE = {}
# ===================== 1. 集成的Mapping与名称获取函数 =====================
def get_mapping(mapping_path: str = None):
    """
    用法: mapping = get_mapping(mapping_path)
    作用: 读取并缓存类别名到展示名的映射表。
    输入: mapping_path 为可选的映射文件路径，默认使用全局 MAPPING_PATH。
    输出: dict，类别名映射表；读取失败时为空 dict。
    """
    global GLOBAL_MAPPING_CACHE
    path = mapping_path or MAPPING_PATH
    if path not in GLOBAL_MAPPING_CACHE:
        try:
            with open(path, 'r', encoding="utf-8") as f:
                data = json.load(f)
                if 'mapping' in data:
                    data = data['mapping']
            GLOBAL_MAPPING_CACHE[path] = data
        except Exception as e:
            print(f"⚠️ 无法读取 Mapping 文件: {e}")
            GLOBAL_MAPPING_CACHE[path] = {}
    return GLOBAL_MAPPING_CACHE[path]

def get_target_object_name(sample_record: dict, source_dir: Path, mapping_data: dict) -> Tuple[str, bool]:
    """
    用法: name, found = get_target_object_name(sample_record, source_dir, mapping_data)
    作用: 从 sample record 中获取目标物体展示名。
    输入: sample_record 为 placement 样本记录；source_dir 保留兼容旧调用；mapping_data 为类别映射表。
    输出: (object_name, is_found_target)。
    """
    del source_dir
    target_class_name = sample_record.get('class_name')
    is_found_target = False
    if target_class_name:
        target_object_name = mapping_data.get(target_class_name, target_class_name)
        is_found_target = True
    else:
        target_object_name = "the object"
    return target_object_name, is_found_target

def get_reference_objects_with_names(
    scene_data: SceneData,
    mapping_data: dict,
) -> Tuple[List[ObjectInfo], List[str]]:
    """
    用法: refs, names = get_reference_objects_with_names(scene_data, mapping_data)
    作用: 从 scene_data.objects 中获取当前帧所有物体作为参照物。
    输入: 场景数据 scene_data，objects 已由 adapter 按帧加载；mapping_data 为类别映射表。
    输出: 参照物 ObjectInfo 列表及其展示名列表。
    """
    reference_objects = list(scene_data.objects)
    reference_names = [
        mapping_data.get(obj.class_name, obj.class_name)
        for obj in reference_objects
    ]
    return reference_objects, reference_names


# ===================== 2. 空间几何计算 (真实 box + 像素中心角度法) =====================
def get_camera_aabb(corners_world: np.ndarray, E_w2c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    用法: cam_min, cam_max = get_camera_aabb(corners_world, E_w2c)
    作用: 将世界坐标 box 角点变换到相机坐标系并返回 AABB。
    输入: corners_world 为 (N,3) 世界坐标角点；E_w2c 为 (4,4) world->camera 矩阵。
    输出: tuple[np.ndarray, np.ndarray]，相机坐标系下的最小点和最大点。
    """
    corners_homo = np.concatenate([corners_world, np.ones((corners_world.shape[0], 1))], axis=1)
    corners_cam = (E_w2c @ corners_homo.T).T[:, :3]
    return corners_cam.min(axis=0), corners_cam.max(axis=0)

def get_2d_bbox(corners_world: np.ndarray, E_w2c: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    用法: min_uv, max_uv = get_2d_bbox(corners_world, E_w2c, K)
    作用: 将世界坐标 box 角点投影到像素坐标并返回 2D AABB。
    输入: corners_world 为 (N,3) 世界角点；E_w2c 为 (4,4)；K 为 (3,3) 内参。
    输出: tuple[np.ndarray, np.ndarray]，像素坐标最小点和最大点。
    """
    corners_img, _ = project_world(corners_world, K, E_w2c)
    return corners_img.min(axis=0), corners_img.max(axis=0)

def center_distance(min1: np.ndarray, max1: np.ndarray, min2: np.ndarray, max2: np.ndarray) -> float:
    """
    用法: dist = center_distance(min1, max1, min2, max2)
    作用: 计算两个 AABB 中心点之间的欧氏距离。
    输入: 两个 AABB 的最小点和最大点。
    输出: float，中心点距离。
    """
    center1 = (min1 + max1) / 2.0
    center2 = (min2 + max2) / 2.0
    return float(np.linalg.norm(center1 - center2))

def get_world_aabb(corners_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    用法: world_min, world_max = get_world_aabb(corners_world)
    作用: 从真实世界坐标 box 角点计算 world AABB。
    输入: corners_world 为 (N,3) 世界坐标角点。
    输出: tuple[np.ndarray, np.ndarray]，世界坐标最小点和最大点。
    """
    corners_world = np.asarray(corners_world, dtype=np.float64)
    return corners_world.min(axis=0), corners_world.max(axis=0)

def polygon_area_xy(points_xy: np.ndarray) -> float:
    """
    用法: area = polygon_area_xy(points_xy)
    作用: 计算 XY 平面多边形面积。
    输入: points_xy 为按边界顺序排列的 (N,2) 顶点。
    输出: float，多边形面积；顶点不足 3 个时为 0。
    """
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if points_xy.shape[0] < 3:
        return 0.0
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    return float(abs(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))))

def cross_xy(origin: np.ndarray, point_a: np.ndarray, point_b: np.ndarray) -> float:
    """
    用法: value = cross_xy(origin, point_a, point_b)
    作用: 计算二维向量 origin->point_a 与 origin->point_b 的叉积。
    输入: 三个 (2,) XY 点。
    输出: float，正值表示 point_b 在 origin->point_a 左侧。
    """
    return float(
        (point_a[0] - origin[0]) * (point_b[1] - origin[1])
        - (point_a[1] - origin[1]) * (point_b[0] - origin[0])
    )

def convex_hull_xy(corners_world: np.ndarray) -> np.ndarray:
    """
    用法: hull = convex_hull_xy(corners_world)
    作用: 计算真实 box 角点投影到 XY 平面后的凸包足迹。
    输入: corners_world 为 (N,3) 世界坐标角点。
    输出: (M,2) 逆时针凸包顶点；退化时返回去重后的点。
    """
    points = np.unique(np.asarray(corners_world, dtype=np.float64)[:, :2], axis=0)
    if points.shape[0] <= 2:
        return points

    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    lower = []
    for point in points:
        while len(lower) >= 2 and cross_xy(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross_xy(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if hull.shape[0] >= 3 and polygon_signed_area_xy(hull) < 0.0:
        hull = hull[::-1]
    return hull

def polygon_signed_area_xy(points_xy: np.ndarray) -> float:
    """
    用法: signed_area = polygon_signed_area_xy(points_xy)
    作用: 计算 XY 多边形有向面积，用于确认顶点方向。
    输入: points_xy 为按边界顺序排列的 (N,2) 顶点。
    输出: float，逆时针为正，顺时针为负。
    """
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if points_xy.shape[0] < 3:
        return 0.0
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def line_intersection_xy(p1: np.ndarray, p2: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """
    用法: point = line_intersection_xy(p1, p2, e1, e2)
    作用: 计算线段 p1->p2 与裁剪边界 e1->e2 所在直线的交点。
    输入: 四个 (2,) XY 点。
    输出: (2,) 交点；近似平行时返回 p2 作为稳定兜底。
    """
    segment_vec = p2 - p1
    edge_vec = e2 - e1
    denom = segment_vec[0] * edge_vec[1] - segment_vec[1] * edge_vec[0]
    if abs(float(denom)) < 1e-12:
        return p2
    offset = e1 - p1
    t = (offset[0] * edge_vec[1] - offset[1] * edge_vec[0]) / denom
    return p1 + t * segment_vec

def polygon_intersection_area_xy(subject_xy: np.ndarray, clip_xy: np.ndarray) -> float:
    """
    用法: area = polygon_intersection_area_xy(subject_xy, clip_xy)
    作用: 用凸多边形裁剪计算两个 XY 足迹的交集面积。
    输入: 两个逆时针凸多边形顶点数组。
    输出: float，交集面积。
    """
    subject_xy = np.asarray(subject_xy, dtype=np.float64)
    clip_xy = np.asarray(clip_xy, dtype=np.float64)
    if subject_xy.shape[0] < 3 or clip_xy.shape[0] < 3:
        return 0.0

    output = subject_xy
    for edge_idx in range(clip_xy.shape[0]):
        edge_start = clip_xy[edge_idx]
        edge_end = clip_xy[(edge_idx + 1) % clip_xy.shape[0]]
        input_polygon = output
        output = []
        if len(input_polygon) == 0:
            break

        prev_point = input_polygon[-1]
        prev_inside = is_left_of_edge(prev_point, edge_start, edge_end)
        for curr_point in input_polygon:
            curr_inside = is_left_of_edge(curr_point, edge_start, edge_end)
            if curr_inside:
                if not prev_inside:
                    output.append(line_intersection_xy(prev_point, curr_point, edge_start, edge_end))
                output.append(curr_point)
            elif prev_inside:
                output.append(line_intersection_xy(prev_point, curr_point, edge_start, edge_end))
            prev_point = curr_point
            prev_inside = curr_inside
        output = np.asarray(output, dtype=np.float64)

    return polygon_area_xy(np.asarray(output, dtype=np.float64))

def is_left_of_edge(point_xy: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
    """
    用法: inside = is_left_of_edge(point_xy, edge_start, edge_end)
    作用: 判断点是否位于逆时针凸多边形边的内侧。
    输入: 待测点和边界起止点。
    输出: bool，True 表示在边左侧或边上。
    """
    edge_vec = edge_end - edge_start
    point_vec = point_xy - edge_start
    cross = edge_vec[0] * point_vec[1] - edge_vec[1] * point_vec[0]
    return bool(cross >= -1e-9)

def footprint_overlap_ratio(
    target_corners_world: np.ndarray,
    ref_corners_world: np.ndarray,
) -> float:
    """
    用法: ratio = footprint_overlap_ratio(target_corners_world, ref_corners_world)
    作用: 计算两个真实 box 在世界 XY 平面足迹上的重叠比例。
    输入: target/ref 的 (N,3) 世界坐标角点。
    输出: float，交集面积除以较小足迹面积，范围通常为 [0,1]。
    """
    target_hull = convex_hull_xy(target_corners_world)
    ref_hull = convex_hull_xy(ref_corners_world)
    inter_area = polygon_intersection_area_xy(target_hull, ref_hull)
    target_area = polygon_area_xy(target_hull)
    ref_area = polygon_area_xy(ref_hull)
    base_area = min(target_area, ref_area)
    if base_area <= 1e-12:
        return 0.0
    return float(inter_area / base_area)

def describe_vertical_relation(
    target_corners_world: np.ndarray,
    ref_corners_world: np.ndarray,
    overlap_threshold: float = VERTICAL_FOOTPRINT_OVERLAP_RATIO,
    tolerance_ratio: float = VERTICAL_TOLERANCE_RATIO,
    min_tolerance: float = MIN_VERTICAL_TOLERANCE,
    max_penetration_ratio: float = VERTICAL_MAX_PENETRATION_RATIO,
    max_penetration: float = MAX_VERTICAL_PENETRATION,
    center_separation_ratio: float = VERTICAL_CENTER_SEPARATION_RATIO,
    min_center_separation: float = MIN_VERTICAL_CENTER_SEPARATION,
) -> Optional[str]:
    """
    用法: relation = describe_vertical_relation(target_corners_world, ref_corners_world)
    作用: 基于真实 world box 判断上下关系，并允许有限 bbox 穿插。
    输入: target/ref 的 (N,3) 世界坐标角点、足迹重叠阈值、穿插容差和中心高度分离阈值。
    输出: "the top of"、"below" 或 None；None 表示不属于上下关系。
    """
    if footprint_overlap_ratio(target_corners_world, ref_corners_world) < overlap_threshold:
        return None

    t_min, t_max = get_world_aabb(target_corners_world)
    r_min, r_max = get_world_aabb(ref_corners_world)
    target_height = max(float(t_max[2] - t_min[2]), 0.0)
    ref_height = max(float(r_max[2] - r_min[2]), 0.0)
    min_height = min(target_height, ref_height)
    contact_tolerance = max(float(min_tolerance), float(tolerance_ratio) * min_height)
    penetration_limit = max(
        contact_tolerance,
        min(float(max_penetration), float(max_penetration_ratio) * min_height),
    )
    center_separation = max(
        float(min_center_separation),
        float(center_separation_ratio) * min_height,
    )

    target_center_z = float((t_min[2] + t_max[2]) * 0.5)
    ref_center_z = float((r_min[2] + r_max[2]) * 0.5)
    top_penetration = max(0.0, float(r_max[2] - t_min[2]))
    below_penetration = max(0.0, float(t_max[2] - r_min[2]))
    if target_center_z > ref_center_z + center_separation and top_penetration <= penetration_limit:
        return "the top of"
    if target_center_z < ref_center_z - center_separation and below_penetration <= penetration_limit:
        return "below"
    return None

def project_box_center_to_camera(corners_world: np.ndarray, E_w2c: np.ndarray) -> np.ndarray:
    """
    用法: center_cam = project_box_center_to_camera(corners_world, E_w2c)
    作用: 将真实 world box 的中心点变换到相机坐标系。
    输入: corners_world 为 (N,3) 世界坐标角点；E_w2c 为 (4,4) world->camera 矩阵。
    输出: np.ndarray，形状为 (3,) 的相机坐标中心 [x, y, z]。
    """
    center_world = np.asarray(corners_world, dtype=np.float64).mean(axis=0, dtype=np.float64)
    center_homo = np.append(center_world, 1.0)
    return (np.asarray(E_w2c, dtype=np.float64) @ center_homo)[:3]

def project_box_center_to_pixel(corners_world: np.ndarray, E_w2c: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    用法: center_uv = project_box_center_to_pixel(corners_world, E_w2c, K)
    作用: 将真实 world box 的中心点投影到像素坐标。
    输入: corners_world 为 (N,3) 世界坐标角点；E_w2c 为 (4,4)；K 为 (3,3)。
    输出: np.ndarray，形状为 (2,) 的像素坐标 [u, v]。
    """
    center_world = np.asarray(corners_world, dtype=np.float64).mean(axis=0, dtype=np.float64)
    center_uv, _ = project_world(center_world[None, :], K, E_w2c)
    return center_uv[0]

def normalize_angle_degrees(angle_deg: float) -> float:
    """
    用法: angle = normalize_angle_degrees(angle_deg)
    作用: 将角度归一化到 [-180, 180) 区间，便于方向扇区判断。
    输入: angle_deg 为任意角度值。
    输出: float，归一化后的角度。
    """
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0

def describe_angle_relation(
    angle_deg: float,
    axis_half_width_deg: float = AXIS_DIRECTION_HALF_WIDTH_DEG,
) -> str:
    """
    用法: relation = describe_angle_relation(angle_deg, axis_half_width_deg)
    作用: 用非均匀扇区将像素向量角度映射到 8 个水平方向。
    输入: angle_deg 为像素向量角度；axis_half_width_deg 为单轴方向半宽。
    输出: str，8 个水平方向之一。
    """
    axis_half_width_deg = float(axis_half_width_deg)
    if not (0.0 < axis_half_width_deg < 45.0):
        raise ValueError("axis_half_width_deg must be in (0, 45)")

    angle = normalize_angle_degrees(angle_deg)
    right_min = -axis_half_width_deg
    right_max = axis_half_width_deg
    front_min = 90.0 - axis_half_width_deg
    front_max = 90.0 + axis_half_width_deg
    back_min = -90.0 - axis_half_width_deg
    back_max = -90.0 + axis_half_width_deg
    left_start = 180.0 - axis_half_width_deg

    if right_min <= angle <= right_max:
        return "the right of"
    if front_min <= angle <= front_max:
        return "in front of"
    if back_min <= angle <= back_max:
        return "behind"
    if angle >= left_start or angle < -left_start:
        return "the left of"
    if right_max < angle < front_min:
        return "the front right of"
    if front_max < angle < left_start:
        return "the front left of"
    if -left_start <= angle < back_min:
        return "the back left of"
    return "the back right of"

def describe_horizontal_relation_by_pixel_angle(
    target_corners_world: np.ndarray,
    ref_corners_world: np.ndarray,
    E_w2c: np.ndarray,
    K: np.ndarray,
    center_eps_px: float = HORIZONTAL_CENTER_EPS_PX,
    axis_half_width_deg: float = AXIS_DIRECTION_HALF_WIDTH_DEG,
) -> str:
    """
    用法: relation = describe_horizontal_relation_by_pixel_angle(target_corners_world, ref_corners_world, E_w2c, K)
    作用: 用两个真实 box 中心的像素投影向量角度判定 8 个水平方向。
    输入: target/ref 的世界角点、world->camera 外参、相机内参、中心重合阈值和单轴扇区半宽。
    输出: str，8 个水平方向之一；中心几乎重合时返回 "near"。
    """
    target_uv = project_box_center_to_pixel(target_corners_world, E_w2c, K)
    ref_uv = project_box_center_to_pixel(ref_corners_world, E_w2c, K)
    delta_uv = target_uv - ref_uv
    if float(np.linalg.norm(delta_uv)) < float(center_eps_px):
        return "near"

    angle_deg = float(np.degrees(np.arctan2(delta_uv[1], delta_uv[0])))
    return describe_angle_relation(angle_deg, axis_half_width_deg=axis_half_width_deg)

def compute_depth_direction_threshold(
    target_corners_world: np.ndarray,
    ref_corners_world: np.ndarray,
    E_w2c: np.ndarray,
    min_depth_cm: float = DEPTH_DIRECTION_MIN_CM,
    extent_ratio: float = DEPTH_DIRECTION_EXTENT_RATIO,
) -> float:
    """
    用法: threshold = compute_depth_direction_threshold(target_corners_world, ref_corners_world, E_w2c)
    作用: 根据最小深度阈值和两个 box 的相机深度厚度生成前后方向阈值。
    输入: target/ref 世界角点、world->camera 外参、最小深度阈值和深度厚度比例。
    输出: float，判定 front/back 所需的相机深度差阈值。
    """
    t_min_c, t_max_c = get_camera_aabb(target_corners_world, E_w2c)
    r_min_c, r_max_c = get_camera_aabb(ref_corners_world, E_w2c)
    target_depth_extent = max(0.0, float(t_max_c[2] - t_min_c[2]))
    ref_depth_extent = max(0.0, float(r_max_c[2] - r_min_c[2]))
    adaptive_threshold = float(extent_ratio) * min(target_depth_extent, ref_depth_extent)
    return max(float(min_depth_cm), adaptive_threshold)

def describe_horizontal_relation_by_depth(
    target_corners_world: np.ndarray,
    ref_corners_world: np.ndarray,
    E_w2c: np.ndarray,
    K: np.ndarray,
    lateral_min_px: float = LATERAL_DIRECTION_MIN_PX,
    depth_min_cm: float = DEPTH_DIRECTION_MIN_CM,
    axis_half_width_deg: float = AXIS_DIRECTION_HALF_WIDTH_DEG,
) -> str:
    """
    用法: relation = describe_horizontal_relation_by_depth(target_corners_world, ref_corners_world, E_w2c, K)
    作用: 在相机横向-深度平面上用角度扇区判断 8 向水平关系。
    输入: target/ref 世界角点、world->camera 外参、相机内参、横向像素阈值、深度阈值和主方向半宽。
    输出: str，深度感知的水平方向关系；中心近似重合时返回 "near"。
    """
    target_uv = project_box_center_to_pixel(target_corners_world, E_w2c, K)
    ref_uv = project_box_center_to_pixel(ref_corners_world, E_w2c, K)
    target_cam = project_box_center_to_camera(target_corners_world, E_w2c)
    ref_cam = project_box_center_to_camera(ref_corners_world, E_w2c)

    delta_u = float(target_uv[0] - ref_uv[0])
    delta_v = float(target_uv[1] - ref_uv[1])
    delta_depth = float(target_cam[2] - ref_cam[2])
    if (
        abs(delta_u) <= float(lateral_min_px)
        and abs(delta_v) <= float(lateral_min_px)
        and abs(delta_depth) <= float(depth_min_cm)
    ):
        return "near"

    depth_threshold = compute_depth_direction_threshold(
        target_corners_world,
        ref_corners_world,
        E_w2c,
        min_depth_cm=depth_min_cm,
    )

    # 将像素偏移和深度偏移归一化，保留更强的前后证据再计算角度。
    direction_x = delta_u / float(lateral_min_px)
    image_direction_y = delta_v / float(lateral_min_px)
    depth_direction_y = -delta_depth / float(depth_threshold)
    if abs(depth_direction_y) > abs(image_direction_y):
        direction_y = depth_direction_y
    else:
        direction_y = image_direction_y
    angle_deg = float(np.degrees(np.arctan2(direction_y, direction_x)))
    return describe_angle_relation(angle_deg, axis_half_width_deg=axis_half_width_deg)

def describe_spatial_relation(
    target_corners_world: np.ndarray,
    ref_corners_world: np.ndarray,
    E_w2c: np.ndarray,
    K: np.ndarray,
) -> str:
    """
    用法: relation = describe_spatial_relation(target_corners_world, ref_corners_world, E_w2c, K)
    作用: 先用真实 world box 判断上下关系；不是上下时融合像素左右和相机深度判断水平关系。
    输入: target/ref 的世界角点、world->camera 外参和相机内参。
    输出: str，10 类方向关系之一，极端中心重合时为 "near"。
    """
    vertical_relation = describe_vertical_relation(target_corners_world, ref_corners_world)
    if vertical_relation is not None:
        return vertical_relation
    return describe_horizontal_relation_by_depth(target_corners_world, ref_corners_world, E_w2c, K)

def get_object_corners_world(obj: ObjectInfo) -> np.ndarray:
    """
    用法: corners = get_object_corners_world(obj)
    作用: 将 ObjectInfo 的 canonical AABB 通过 pose_world 转为真实 world box 角点。
    输入: obj 为带 bbox3d_canonical 和 pose_world 的物体信息。
    输出: (8,3) 世界坐标角点。
    """
    return transform_points(_get_bbox_corners(obj.bbox3d_canonical), obj.pose_world)

def infer_image_size_from_camera(camera: CameraParams) -> Tuple[int, int]:
    """
    用法: img_w, img_h = infer_image_size_from_camera(camera)
    作用: 依据相机参数推断当前可视化图片尺寸，保留旧脚本的尺寸兼容逻辑。
    输入: camera 为相机参数。
    输出: tuple[int, int]，图像宽高。
    """
    est_w = int(camera.K[0, 2] * 2)
    est_h = int(camera.K[1, 2] * 2)
    if abs(est_w - 640) < 100 or abs(est_h - 480) < 100:
        return 640, 480
    if abs(est_w - 1096) < 150 or abs(est_h - 852) < 150:
        return 1096, 852
    return max(640, est_w), max(480, est_h)

def build_reference_projection_info(
    reference_objects: List[ObjectInfo],
    E_w2c: np.ndarray,
    K: np.ndarray,
) -> Dict[str, dict]:
    """
    用法: info_map = build_reference_projection_info(reference_objects, E_w2c, K)
    作用: 预计算参照物的真实角点、2D bbox 和相机深度，供筛选与遮挡检测复用。
    输入: 参照物列表、world->camera 矩阵和相机内参。
    输出: dict，键为 obj_id，值包含 corners_world、min_2d、max_2d、depth 和 obj。
    """
    obj_info_map = {}
    for obj in reference_objects:
        corners_world = get_object_corners_world(obj)
        min_2d, max_2d = get_2d_bbox(corners_world, E_w2c, K)
        depth = get_camera_aabb(corners_world, E_w2c)[0][2]
        obj_info_map[obj.obj_id] = {
            "corners_world": corners_world,
            "min_2d": min_2d,
            "max_2d": max_2d,
            "depth": depth,
            "obj": obj,
        }
    return obj_info_map

def compute_projected_box_area(min_2d: np.ndarray, max_2d: np.ndarray) -> float:
    """
    用法: area = compute_projected_box_area(min_2d, max_2d)
    作用: 计算像素坐标 2D bbox 面积。
    输入: min_2d 和 max_2d 为像素 bbox 的最小/最大坐标。
    输出: float，非负面积。
    """
    box_w = max(0.0, float(max_2d[0] - min_2d[0]))
    box_h = max(0.0, float(max_2d[1] - min_2d[1]))
    return box_w * box_h

def compute_image_intersection_area(
    min_2d: np.ndarray,
    max_2d: np.ndarray,
    img_w: int,
    img_h: int,
) -> float:
    """
    用法: inter_area = compute_image_intersection_area(min_2d, max_2d, img_w, img_h)
    作用: 计算 2D bbox 与图像画幅的交集面积。
    输入: 像素 bbox 最小/最大坐标，以及图像宽高。
    输出: float，位于图像内的 bbox 面积。
    """
    inter_xmin = max(float(min_2d[0]), 0.0)
    inter_ymin = max(float(min_2d[1]), 0.0)
    inter_xmax = min(float(max_2d[0]), float(img_w))
    inter_ymax = min(float(max_2d[1]), float(img_h))
    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    return inter_w * inter_h

def passes_reference_visibility_filter(
    min_2d: np.ndarray,
    max_2d: np.ndarray,
    img_w: int,
    img_h: int,
) -> Tuple[bool, float]:
    """
    用法: is_visible, area = passes_reference_visibility_filter(min_2d, max_2d, img_w, img_h)
    作用: 检查候选参照物是否有足够大的画面内可见区域。
    输入: 候选参照物 2D bbox 和图像尺寸。
    输出: (bool, float)，是否通过过滤及完整投影 bbox 面积。
    """
    area = compute_projected_box_area(min_2d, max_2d)
    inter_area = compute_image_intersection_area(min_2d, max_2d, img_w, img_h)
    if inter_area <= 0.0:
        return False, area

    area_threshold = SMALL_IMAGE_AREA_THRESHOLD if img_w < 800 else LARGE_IMAGE_AREA_THRESHOLD
    if inter_area < area_threshold:
        return False, area

    visibility_ratio = inter_area / (area + 1e-6)
    return visibility_ratio >= MIN_VISIBILITY_RATIO, area

def compute_bbox_overlap_area(
    min_a: np.ndarray,
    max_a: np.ndarray,
    min_b: np.ndarray,
    max_b: np.ndarray,
) -> float:
    """
    用法: area = compute_bbox_overlap_area(min_a, max_a, min_b, max_b)
    作用: 计算两个像素 2D bbox 的重叠面积。
    输入: 两个 bbox 的最小/最大像素坐标。
    输出: float，重叠面积。
    """
    overlap_xmin = max(float(min_a[0]), float(min_b[0]))
    overlap_ymin = max(float(min_a[1]), float(min_b[1]))
    overlap_xmax = min(float(max_a[0]), float(max_b[0]))
    overlap_ymax = min(float(max_a[1]), float(max_b[1]))
    overlap_w = max(0.0, overlap_xmax - overlap_xmin)
    overlap_h = max(0.0, overlap_ymax - overlap_ymin)
    return overlap_w * overlap_h

def compute_occlusion_ratio(
    ref_id: str,
    ref_min_2d: np.ndarray,
    ref_max_2d: np.ndarray,
    ref_area: float,
    ref_depth: float,
    obj_info_map: Dict[str, dict],
    exclude_id: str = None,
) -> float:
    """
    用法: ratio = compute_occlusion_ratio(ref_id, ref_min_2d, ref_max_2d, ref_area, ref_depth, info_map, exclude_id)
    作用: 估计候选参照物被更靠近相机的其他参照物遮挡的比例。
    输入: 当前参照物 id、2D bbox、投影面积、深度、预计算信息和可选排除 id。
    输出: float，遮挡面积除以参照物投影面积。
    """
    occluded_area = 0.0
    for other_id, other_info in obj_info_map.items():
        if other_id == ref_id or other_id == exclude_id:
            continue
        if other_info["depth"] >= ref_depth:
            continue
        occluded_area += compute_bbox_overlap_area(
            ref_min_2d,
            ref_max_2d,
            other_info["min_2d"],
            other_info["max_2d"],
        )
    return occluded_area / (ref_area + 1e-6)

def find_nearest_reference(
    target_corners_world: np.ndarray,
    reference_objects: List[ObjectInfo],
    camera: CameraParams,
    exclude_id: str = None,  
) -> List[Tuple[Optional[ObjectInfo], str, float]]:
    """
    用法: candidates = find_nearest_reference(target_corners_world, reference_objects, camera, exclude_id)
    作用: 经过可见性、遮挡和距离过滤后，为目标 box 选出近邻参照物并计算空间关系。
    输入: 目标 world box 角点、参照物列表、相机参数和可选排除 obj_id。
    输出: list[(ObjectInfo|None, relation, distance)]，按参照物候选顺序返回。
    """
    if not reference_objects:
        return [(None, "near", float('inf'))]
    
    E_w2c = np.linalg.inv(np.asarray(camera.E_c2w, dtype=np.float64))
    K = camera.K
    img_w, img_h = infer_image_size_from_camera(camera)
    t_min_c, t_max_c = get_camera_aabb(target_corners_world, E_w2c)
    obj_info_map = build_reference_projection_info(reference_objects, E_w2c, K)

    valid_candidates = []
    for ref in reference_objects:
        if exclude_id is not None and ref.obj_id == exclude_id:
            continue

        ref_info = obj_info_map[ref.obj_id]
        ref_corners_world = ref_info["corners_world"]
        r_min_c, r_max_c = get_camera_aabb(ref_corners_world, E_w2c)
        is_visible, ref_area = passes_reference_visibility_filter(
            ref_info["min_2d"], ref_info["max_2d"], img_w, img_h
        )
        if not is_visible:
            continue

        occlusion_ratio = compute_occlusion_ratio(
            ref_id=ref.obj_id,
            ref_min_2d=ref_info["min_2d"],
            ref_max_2d=ref_info["max_2d"],
            ref_area=ref_area,
            ref_depth=ref_info["depth"],
            obj_info_map=obj_info_map,
            exclude_id=exclude_id,
        )
        if occlusion_ratio >= MAX_OCCLUSION_RATIO:
            continue

        center_dist = center_distance(t_min_c, t_max_c, r_min_c, r_max_c)
        score = 1.0 / (center_dist + 1e-5)
        relation = describe_spatial_relation(target_corners_world, ref_corners_world, E_w2c, K)
        vertical_priority = 1 if relation in ("the top of", "below") else 0
        valid_candidates.append((vertical_priority, score, center_dist, ref, relation))

    if not valid_candidates:
        return [(None, "near", float('inf'))]

    valid_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top_candidates = valid_candidates[:MAX_REFERENCE_CANDIDATES]

    all_candidates = []
    for _, _, dist, ref, relation in top_candidates:
        all_candidates.append((ref, relation, dist))

    return all_candidates or [(None, "near", float('inf'))]

def calculate_spatial_relation(
    target_corners_world: np.ndarray,
    reference_objects: List[ObjectInfo],
    camera: CameraParams,
    exclude_id: str = None,
    mapping_data: dict = None,
) -> List[Tuple[str, str]]:
    """
    用法: relations = calculate_spatial_relation(target_corners_world, reference_objects, camera, exclude_id, mapping_data)
    作用: 将目标 box 与候选参照物转换为自然语言空间关系和参照物名称。
    输入: 目标 world box 角点、参照物列表、相机参数、可选排除 obj_id 和类别映射表。
    输出: list[(relation, ref_name)]，用于填充自动标注模板。
    """
    all_candidates = find_nearest_reference(
        target_corners_world, reference_objects, camera, exclude_id
    )

    if mapping_data is None:
        mapping_data = {}
    result = []
    for ref, relation, _ in all_candidates:
        if ref is None:
            result.append(("nowhere", "nothing"))
        else:
            ref_name = mapping_data.get(ref.class_name, ref.class_name)
            result.append((relation, ref_name))

    return result

# ===================== 3. 标注生成函数 =====================
def generate_label(
    sample_record: dict,
    scene_data: SceneData,
    reference_objects: List[ObjectInfo],
    target_object_name: str,
    mapping_data: dict,
) -> str:
    """
    用法: label = generate_label(sample_record, scene_data, reference_objects, target_object_name, mapping_data)
    作用: 生成单个 placement sample 的自然语言移动指令。
    输入: sample 记录、场景数据、参照物列表、目标物体展示名和类别映射表。
    输出: str，完整自然语言标注。
    """
    canonical_aabb = np.asarray(sample_record["canonical_aabb_object"], dtype=np.float64)
    canonical_corners = _get_bbox_corners(canonical_aabb)

    original_pose = np.asarray(sample_record["original_pose_world"], dtype=np.float64)
    original_corners = transform_points(canonical_corners, original_pose)

    placement_pose = np.asarray(sample_record["transform_world"], dtype=np.float64)
    placement_corners = transform_points(canonical_corners, placement_pose)

    target_obj_id = sample_record.get('object_id')
    if not target_obj_id:
        target_obj_id = sample_record.get('sample_id').split('_')[2]

    # 1. 先正常计算原始位置的描述
    original_relations = calculate_spatial_relation(
        original_corners, reference_objects, scene_data.camera, exclude_id=target_obj_id, mapping_data=mapping_data
    )
    rel_original, ref_a_name = original_relations[0] if original_relations else ("near", "the reference object")

    # 2. 计算目标位置的描述
    placement_relations = calculate_spatial_relation(
        placement_corners, reference_objects, scene_data.camera, exclude_id=None, mapping_data=mapping_data
    )
    rel_placement, ref_b_name = placement_relations[0] if placement_relations else ("near", "the reference object")

    # 【关键修复】只有当「参照物相同 AND 方位也相同」时，才强制换一个
    if (ref_b_name == ref_a_name) and (rel_placement == rel_original):
        # 找到 ref_a 对应的 object_id
        ref_a_id = None
        for obj in reference_objects:
            if mapping_data.get(obj.class_name, obj.class_name) == ref_a_name:
                ref_a_id = obj.obj_id
                break

        # 把 ref_a 从候选列表里拿掉，重新选一个
        if ref_a_id is not None:
            filtered_refs = [obj for obj in reference_objects if obj.obj_id != ref_a_id]
            if filtered_refs:
                new_placement_relations = calculate_spatial_relation(
                    placement_corners, filtered_refs, scene_data.camera, exclude_id=target_obj_id, mapping_data=mapping_data
                )
                # 再次保护：确保新列表有值
                rel_placement, ref_b_name = new_placement_relations[0] if new_placement_relations else (rel_placement, ref_b_name)

    return LABEL_TEMPLATE.format(
        object_name=target_object_name,
        rel_original=rel_original,
        ref_a_name=ref_a_name,
        rel_placement=rel_placement,
        ref_b_name=ref_b_name,
    )


# ===================== 4. 基于图片的遍历处理逻辑 =====================
def build_parser() -> argparse.ArgumentParser:
    """
    用法: parser = build_parser()
    作用: 构建 auto_label.py 的命令行参数解析器。
    输入: 无。
    输出: argparse.ArgumentParser。
    """
    parser = argparse.ArgumentParser(description="以图片文件为索引，自动生成对应的 placement 标注")
    parser.add_argument("--image-dir", required=True, type=Path, help="可视化的图片目录 (例如 outputs/placement_rgb_bbox_vis)")
    parser.add_argument("--outputs-base", type=Path, default=PROJECT_ROOT / "outputs", help="数据集的输出基准目录")
    parser.add_argument("--output-dir", required=True, type=Path, help="标注 JSON 和 HTML 报告输出目录")
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH, help="类别名称映射文件路径 (JSON)")
    parser.add_argument("--limit", type=int, default=None, help="仅标注前 N 个样本")
    parser.add_argument("--sample-ids", nargs="+", default=None, help="仅标注指定 sample_id，可一次传入多个")
    parser.add_argument("--sample-ids-file", type=Path, default=None, help="从文本文件读取 sample_id，每行一个")
    parser.add_argument("--overwrite", action="store_true", help="兼容旧调用，当前 JSON/HTML 输出会直接刷新")
    return parser

def load_sample_ids_file(sample_ids_file: Path) -> List[str]:
    """
    用法: sample_ids = load_sample_ids_file(Path("sample_ids.txt"))
    作用: 从文本文件读取待标注 sample_id，忽略空行和 # 开头的注释行。
    输入: sample_ids_file 为 sample_id 文本文件路径。
    输出: list[str]，按文件顺序返回有效 sample_id。
    """
    sample_ids = []
    with sample_ids_file.open("r", encoding="utf-8") as f:
        for line in f:
            sample_id = line.strip()
            if not sample_id or sample_id.startswith("#"):
                continue
            sample_ids.append(sample_id)
    return sample_ids

def build_sample_id_filter(
    sample_ids: List[str] = None,
    sample_ids_file: Path = None,
) -> Optional[Set[str]]:
    """
    用法: sample_id_filter = build_sample_id_filter(args.sample_ids, args.sample_ids_file)
    作用: 合并命令行和文件中的 sample_id，构建快速过滤集合。
    输入: sample_ids 为命令行 sample_id 列表；sample_ids_file 为可选文本文件。
    输出: set[str] 或 None；None 表示不按 sample_id 过滤。
    """
    merged_sample_ids = []
    if sample_ids:
        merged_sample_ids.extend(str(item).strip() for item in sample_ids if str(item).strip())
    if sample_ids_file is not None:
        merged_sample_ids.extend(load_sample_ids_file(sample_ids_file))
    if not merged_sample_ids:
        return None
    return set(merged_sample_ids)

def infer_config_path(source_name: str) -> Path:
    """
    用法: config_path = infer_config_path(source_name)
    作用: 根据数据源名称推断 placement 配置文件路径。
    输入: source_name 为图片文件名前缀，如 hope 或 housecat6d。
    输出: Path，对应配置文件路径。
    """
    if "housecat" in source_name.lower():
        return PROJECT_ROOT / "configs/annotation/placement_housecat6d.yaml"
    if "ycbv" in source_name.lower() or "ycb_video" in source_name.lower():
        return PROJECT_ROOT / "configs/annotation/placement_ycbv_test.yaml"
    if "scannet" in source_name.lower():
        return PROJECT_ROOT / "configs/annotation/placement_scannet.yaml"
    if "dopose" in source_name.lower():
        return PROJECT_ROOT / "configs/annotation/placement_dopose.yaml"
    return PROJECT_ROOT / "configs/annotation/placement.yaml"

def load_scene_cached(scene_cache: Dict, adapter, source_dir: Path, scene_id: str, frame_id: str):
    """
    用法: scene = load_scene_cached(scene_cache, adapter, source_dir, scene_id, frame_id)
    作用: 按 source/scene/frame 缓存加载场景，避免重复读取同一帧。
    输入: 场景缓存、数据集 adapter、source 目录、scene_id 和 frame_id。
    输出: SceneData 场景对象。
    """
    key = (source_dir.name, scene_id, frame_id)
    if key not in scene_cache:
        scene_path = Path(adapter.root_dir) / scene_id
        scene_cache[key] = adapter.load_scene(str(scene_path), frame_id)
    return scene_cache[key]

def parse_image_filename(image_path: Path) -> Optional[Tuple[str, str]]:
    """
    用法: parsed = parse_image_filename(Path("hope__scene_0000_0000_obj_1_p000.png"))
    作用: 从可视化图片文件名解析 source_name 和 sample_id。
    输入: image_path 为图片路径，文件名需符合 {source}__{sample_id}.png。
    输出: (source_name, sample_id) 或 None。
    """
    parts = image_path.stem.split("__", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]

def collect_image_files(image_dir: Path) -> List[Path]:
    """
    用法: image_files = collect_image_files(image_dir)
    作用: 收集图片目录下支持的可视化图片。
    输入: image_dir 为图片目录。
    输出: 排序后的 png/jpg 图片路径列表。
    """
    return sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))

def filter_image_files_by_sample_ids(
    image_files: List[Path],
    sample_id_filter: Optional[Set[str]],
) -> List[Path]:
    """
    用法: filtered = filter_image_files_by_sample_ids(image_files, sample_id_filter)
    作用: 按 sample_id 集合过滤图片；未指定集合时返回原列表。
    输入: 图片路径列表和可选 sample_id 集合。
    输出: 过滤后的图片路径列表。
    """
    if sample_id_filter is None:
        return image_files

    filtered_image_files = []
    for image_file in image_files:
        parsed = parse_image_filename(image_file)
        if parsed is None:
            continue
        _, sample_id = parsed
        if sample_id in sample_id_filter:
            filtered_image_files.append(image_file)
    return filtered_image_files

def collect_source_names(image_files: List[Path]) -> Set[str]:
    """
    用法: source_names = collect_source_names(image_files)
    作用: 从图片文件名中收集需要处理的数据源名称。
    输入: 图片路径列表。
    输出: set[str]，数据源名称集合。
    """
    source_names = set()
    for image_file in image_files:
        parsed = parse_image_filename(image_file)
        if parsed is not None:
            source_names.add(parsed[0])
    return source_names

def load_yaml_config(config_path: Path) -> dict:
    """
    用法: cfg = load_yaml_config(config_path)
    作用: 读取 YAML 配置文件。
    输入: config_path 为 YAML 文件路径。
    输出: dict，配置内容。
    """
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_adapter_from_config(ds_cfg: dict):
    """
    用法: adapter = build_adapter_from_config(ds_cfg)
    作用: 根据 dataset 配置构建对应数据集 adapter。
    输入: ds_cfg 为配置中的 dataset 字段。
    输出: HopeAdapter 或 HouseCat6DAdapter 实例。
    """
    ds_type = ds_cfg.get("type", "hope")
    if ds_type == "hope":
        return HopeAdapter(
            root_dir=ds_cfg["root_dir"],
            mesh_dir=ds_cfg.get("mesh_dir"),
            frame_step=ds_cfg.get("frame_step", 60),
        )
    if ds_type == "housecat6d":
        return HouseCat6DAdapter(
            root_dir=ds_cfg["root_dir"],
            frame_step=ds_cfg.get("frame_step", 60),
        )
    if ds_type == "ycb_video":
        return YCBVideoAdapter(
            root_dir=ds_cfg["root_dir"],
            models_info_path=ds_cfg["models_info_path"],
            frame_step=ds_cfg.get("frame_step", 5),
            min_visib_fract=ds_cfg.get("min_visib_fract", 0.0),
        )
    if ds_type == "scannet":
        return ScanNetAdapter(
            root_dir=ds_cfg["root_dir"],
            frame_step=ds_cfg.get("frame_step", 100),
            instance_dir_name=ds_cfg.get("instance_dir_name", "2d-instance"),
            min_visible_pixels=ds_cfg.get("min_visible_pixels", 1),
            excluded_labels=ds_cfg.get("excluded_labels"),
        )
    if ds_type == "dopose":
        return DoPoseAdapter(
            root_dir=ds_cfg["root_dir"],
            models_info_path=ds_cfg["models_info_path"],
            models_names_path=ds_cfg.get("models_names_path"),
            subsets=ds_cfg.get("subsets"),
            frame_step=ds_cfg.get("frame_step", 1),
            min_visib_fract=ds_cfg.get("min_visib_fract", 0.0),
        )
    raise ValueError(f"不支持的数据集类型: {ds_type}")

def load_sample_lookup(samples_dir: Path) -> Dict[str, dict]:
    """
    用法: lookup = load_sample_lookup(samples_dir)
    作用: 读取 source/samples 下所有 placement sample JSON 并按 sample_id 建索引。
    输入: samples_dir 为 samples 目录。
    输出: dict，键为 sample_id，值为 sample record。
    """
    lookup = {}
    if not samples_dir.exists():
        return lookup
    for json_file in samples_dir.glob("*.json"):
        with json_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for record in payload.get("samples", []):
            lookup[record["sample_id"]] = record
    return lookup

def build_source_indexes(
    source_names: Set[str],
    outputs_base: Path,
) -> Tuple[Dict[str, Dict[str, dict]], Dict[str, object]]:
    """
    用法: lookup_table, adapters = build_source_indexes(source_names, outputs_base)
    作用: 为本次需要处理的数据源构建 sample 查找表和 adapter。
    输入: source_names 为数据源名称集合；outputs_base 为 placement 输出根目录。
    输出: (lookup_table, adapters)，均按 source_name 索引。
    """
    lookup_table = {}
    adapters = {}
    for source_name in sorted(source_names):
        source_dir = outputs_base / source_name
        if not source_dir.exists():
            continue

        cfg = load_yaml_config(infer_config_path(source_name))
        adapters[source_name] = build_adapter_from_config(cfg.get("dataset", {}))
        lookup_table[source_name] = load_sample_lookup(source_dir / "samples")
    return lookup_table, adapters

def build_label_record(
    img_file: Path,
    source_name: str,
    sample_id: str,
    sample_record: dict,
    source_dir: Path,
    adapter,
    scene_cache: Dict,
    mapping_data: dict,
) -> dict:
    """
    用法: record = build_label_record(img_file, source_name, sample_id, sample_record, source_dir, adapter, scene_cache, mapping_data)
    作用: 为单张图片和 sample record 生成 all_labels.json 中的一条记录。
    输入: 图片路径、source/sample 信息、数据源目录、adapter、场景缓存和类别映射表。
    输出: dict，包含图片名、sample_id、目标名称和生成 label。
    """
    scene = load_scene_cached(
        scene_cache,
        adapter,
        source_dir,
        str(sample_record["scene_id"]),
        str(sample_record["frame_id"]),
    )
    target_object_name, is_found_target = get_target_object_name(sample_record, source_dir, mapping_data)
    reference_objects, _ = get_reference_objects_with_names(scene, mapping_data)
    label = generate_label(sample_record, scene, reference_objects, target_object_name, mapping_data)
    return {
        "image_filename": img_file.name,
        "sample_id": sample_id,
        "source_name": source_name,
        "target_object_name": target_object_name,
        "is_found_target": is_found_target,
        "label": label,
    }

def save_all_labels(output_dir: Path, all_labels: List[dict]) -> Path:
    """
    用法: path = save_all_labels(output_dir, all_labels)
    作用: 保存 all_labels.json 汇总文件。
    输入: 输出目录和 label 记录列表。
    输出: Path，写入的 JSON 文件路径。
    """
    all_labels_path = output_dir / "all_labels.json"
    with all_labels_path.open("w", encoding="utf-8") as f:
        json.dump(all_labels, f, indent=2, ensure_ascii=False)
    return all_labels_path

def build_report_html(image_dir: Path, output_dir: Path, all_labels: List[dict]) -> str:
    """
    用法: html = build_report_html(image_dir, output_dir, all_labels)
    作用: 构建只读 HTML 标注查看报告。
    输入: 图片目录、输出目录和 label 记录列表。
    输出: str，完整 HTML 内容。
    """
    html_lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>标注查看</title>",
        "<style>",
        "  body { font-family: 'Segoe UI', sans-serif; background-color: #f4f4f9; padding: 20px; padding-top: 60px; }",
        "  .header-bar { position: fixed; top: 0; left: 0; right: 0; background: #2c3e50; color: white; padding: 10px 40px; z-index: 1000; box-shadow: 0 2px 10px rgba(0,0,0,0.3); }",
        "  .container { max-width: 1200px; margin: auto; }",
        "  .card { display: flex; background: white; margin-bottom: 15px; padding: 15px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); align-items: center; }",
        "  .card img { max-width: 380px; max-height: 380px; border-radius: 4px; object-fit: contain; margin-right: 30px; background: #eee; }",
        "  .info { flex: 1; }",
        "  .filename { color: #888; font-size: 13px; margin-bottom: 8px; font-family: monospace; }",
        "  .label { font-size: 20px; font-weight: bold; color: #34495e; line-height: 1.5; }",
        "  .highlight { color: #e74c3c; }",
        "</style>",
        "</head><body>",
        "<div class='header-bar'>",
        "  <h2 style='margin:0'>📸 标注查看</h2>",
        "</div>",
        "<div class='container'>",
    ]

    for item in all_labels:
        img_rel_path = os.path.relpath(image_dir / item["image_filename"], output_dir)
        fname = item["image_filename"]
        html_lines.append("  <div class='card'>")
        html_lines.append(f"    <img src='{img_rel_path}' loading='lazy' />")
        html_lines.append("    <div class='info'>")
        html_lines.append(f"      <div class='filename'>📄 {fname}</div>")
        html_lines.append(f"      <div class='label'>👉 <span class='highlight'>{item['label']}</span></div>")
        html_lines.append("    </div></div>")

    html_lines.extend(["</div></body></html>"])
    return "\n".join(html_lines)

def save_report_html(image_dir: Path, output_dir: Path, all_labels: List[dict]) -> Optional[Path]:
    """
    用法: report_path = save_report_html(image_dir, output_dir, all_labels)
    作用: 当存在 label 记录时保存只读 HTML 报告。
    输入: 图片目录、输出目录和 label 记录列表。
    输出: Path 或 None；无记录时不生成报告。
    """
    if not all_labels:
        return None
    report_path = output_dir / "report.html"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(build_report_html(image_dir, output_dir, all_labels))
    return report_path

def print_missing_sample_ids(sample_id_filter: Optional[Set[str]], processed_sample_ids: Set[str]) -> None:
    """
    用法: print_missing_sample_ids(sample_id_filter, processed_sample_ids)
    作用: 打印指定但未成功标注的 sample_id 摘要。
    输入: 用户指定的 sample_id 集合和已成功处理的 sample_id 集合。
    输出: None，仅打印提示。
    """
    if sample_id_filter is None:
        return
    missing_sample_ids = sorted(sample_id_filter - processed_sample_ids)
    if not missing_sample_ids:
        return
    preview = ", ".join(missing_sample_ids[:20])
    suffix = " ..." if len(missing_sample_ids) > 20 else ""
    print(f"未成功标注的指定 sample_id ({len(missing_sample_ids)}): {preview}{suffix}")

def auto_label_from_images(
    image_dir: Path,
    outputs_base: Path,
    output_dir: Path,
    limit: int = None,
    overwrite: bool = False,
    sample_id_filter: Optional[Set[str]] = None,
    mapping_path: str = None,
) -> int:
    """
    用法: count = auto_label_from_images(image_dir, outputs_base, output_dir, sample_id_filter={"sample_a"}, mapping_path="mapping.json")
    作用: 以可视化图片为索引生成自动标注，可选只处理指定 sample_id。
    输入: 图片目录、placement 输出根目录、标注输出目录、limit、overwrite、sample_id 过滤集合和类别映射文件路径。
    输出: int，实际成功标注的样本数量。
    """
    del overwrite
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_data = get_mapping(mapping_path)

    image_files = collect_image_files(image_dir)
    if not image_files:
        print(f"未在 {image_dir} 中找到任何图片！")
        return 0

    if sample_id_filter is not None:
        image_files = filter_image_files_by_sample_ids(image_files, sample_id_filter)
        print(f"指定 sample_id 数量: {len(sample_id_filter)}，匹配到图片: {len(image_files)} 张")
        if not image_files:
            missing_preview = ", ".join(sorted(sample_id_filter)[:20])
            print(f"未找到指定 sample_id 对应的图片: {missing_preview}")
            return 0

    print(f"找到 {len(image_files)} 张图片，开始构建数据索引...")
    lookup_table, adapters = build_source_indexes(collect_source_names(image_files), outputs_base)
    print("数据索引构建完成，开始执行标注...")

    scene_cache = {}
    all_labels = []
    labeled = 0
    processed_sample_ids = set()

    for img_file in image_files:
        parsed = parse_image_filename(img_file)
        if parsed is None:
            continue
        source_name, sample_id = parsed

        if source_name not in lookup_table or sample_id not in lookup_table[source_name]:
            continue
        if source_name not in adapters:
            continue

        all_labels.append(
            build_label_record(
                img_file=img_file,
                source_name=source_name,
                sample_id=sample_id,
                sample_record=lookup_table[source_name][sample_id],
                source_dir=outputs_base / source_name,
                adapter=adapters[source_name],
                scene_cache=scene_cache,
                mapping_data=mapping_data,
            )
        )

        labeled += 1
        processed_sample_ids.add(sample_id)
        if labeled % 50 == 0:
            print(f"已标注 {labeled} 张图片，最新: {img_file.name}")
        if limit is not None and labeled >= limit:
            break

    print_missing_sample_ids(sample_id_filter, processed_sample_ids)
    save_all_labels(output_dir, all_labels)
    if all_labels:
        print("\n正在生成只读 HTML 报告...")
        save_report_html(image_dir, output_dir, all_labels)
        print(f"✅ 只读 HTML 报告已生成！请使用 Web 服务查看。")
    
    print(f"\n✅ 标注完成")
    print(f"实际成功标注: {labeled} 个样本")
    
    return labeled


def main() -> None:
    """
    用法: python tools/auto_label.py --image-dir outputs/placement_rgb_bbox_vis --output-dir outputs/auto_labels
    作用: 解析命令行参数并执行自动标注主流程。
    输入: 无，参数来自命令行。
    输出: None，在终端打印处理结果。
    """
    args = build_parser().parse_args()
    image_dir = args.image_dir.resolve()
    outputs_base = args.outputs_base.resolve()
    output_dir = args.output_dir.resolve()
    mapping_path = str(args.mapping.resolve()) if args.mapping else None
    sample_id_filter = build_sample_id_filter(args.sample_ids, args.sample_ids_file)

    auto_label_from_images(
        image_dir=image_dir,
        outputs_base=outputs_base,
        output_dir=output_dir,
        limit=args.limit,
        overwrite=args.overwrite,
        sample_id_filter=sample_id_filter,
        mapping_path=mapping_path,
    )
    
    print(f"图片来源目录: {image_dir}")
    print(f"数据索引目录: {outputs_base}")

if __name__ == "__main__":
    
    """
        启动一个简易的 python Web 服务器，监听 8080 端口
        python3 -m http.server 8080
        如果使用vscode，运行该命令后会自动弹窗，打开浏览器后即可查看标注
        信息量较大，可能卡顿，请耐心等待加载完成！

        或者
        打开本地电脑的终端，使用 SSH 端口转发功能把服务器的 8080 端口映射到本地电脑的 8080 端口：
        ssh -L 8080:localhost:8080 your_username@server_address
        然后在你本地电脑的浏览器里访问 http://localhost:8080/outputs/auto_labels/report.html 就可以看到报告了！
    """

    main()
