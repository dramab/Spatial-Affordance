#!/usr/bin/env python3
"""
tools/auto_label.py
-------------------
以渲染好的可视化图片为索引，自动为 placement 空位框样本生成自然语言标注。
结合 3D 物理测距与 2D 视觉方位（8向最大间隙法），生成高精度的空间关系描述。

======================== 用法示例 ========================
python tools/auto_label.py \
    --image-dir /data/jiajun.xie/Spatial-Affordance/outputs/placement_rgb_bbox_vis \
    --outputs-base /data/jiajun.xie/Spatial-Affordance/outputs \
    --output-dir /data/jiajun.xie/Spatial-Affordance/outputs/auto_labels \
    --limit 50

======================== 参数说明 ========================
--image-dir:    渲染好的 RGB 图片目录（作为数据驱动的基准，图片名需符合 {source}__{id}.png 规范）
--outputs-base: JSON 等原始数据的根目录（脚本会去这里找 source_name 对应的 samples/ 和 categories/）
--output-dir:   标注文本 txt、汇总 json 及 report.html 的统一输出目录
--limit:        (可选) 限制处理的图片数量，方便快速测试
--overwrite:    (可选) 覆盖已存在的标注文本

======================== 输出内容 ========================
1. 独立的 .txt 文件，包含一句话标注，与图片同名。
2. all_labels.json，汇总所有成功标注的信息。
3. report.html，单文件离线网页，内嵌图片和文本，可直接下载到本地浏览器双击查看！
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple, Union
import numpy as np
import yaml

# 请确保项目根目录正确
PROJECT_ROOT = Path("/data/jiajun.xie/Spatial-Affordance")
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.free_bbox.grid_ops import _get_bbox_corners
from src.utils.coord_utils import transform_points
from src.datasets.hope_adapter import HopeAdapter
from src.datasets.housecat6d_adapter import HouseCat6DAdapter
from src.annotation.free_bbox.datatypes import SceneData, ObjectInfo, CameraParams

# ===================== 全局配置与缓存 =====================
LABEL_TEMPLATE = "Move {object_name} located at {rel_original} {ref_a_name} to {rel_placement} {ref_b_name}."
MAPPING_PATH = '/data/wenhao.hai/Spatial-Affordance/tools/api_label/mapping.json'

# 对角线方位融合阈值 (像素差值): 当最大间隙与第二大间隙(属于不同轴)的差值小于该值时，融合为斜向方位
DIAGONAL_THRESHOLD = 40.0 

GLOBAL_MAPPING_CACHE = None
CATEGORY_JSON_CACHE = {}

# ===================== 1. 集成的Mapping与名称获取函数 =====================
def get_mapping():
    global GLOBAL_MAPPING_CACHE
    if GLOBAL_MAPPING_CACHE is None:
        try:
            with open(MAPPING_PATH, 'r', encoding="utf-8") as f:
                GLOBAL_MAPPING_CACHE = json.load(f)
                if 'mapping' in GLOBAL_MAPPING_CACHE:
                    GLOBAL_MAPPING_CACHE = GLOBAL_MAPPING_CACHE['mapping']
        except Exception as e:
            print(f"⚠️ 无法读取 Mapping 文件: {e}")
            GLOBAL_MAPPING_CACHE = {}
    return GLOBAL_MAPPING_CACHE

def get_target_object_name(sample_record: dict, source_dir: Path) -> Tuple[str, bool]:
    mapping_data = get_mapping()
    target_class_name = sample_record.get('class_name')
    is_found_target = False
    if target_class_name:
        target_object_name = mapping_data.get(target_class_name, target_class_name)
        is_found_target = True
    else:
        target_object_name = "the object"
    return target_object_name, is_found_target

def get_reference_objects_with_names(
    adapter: Union[HouseCat6DAdapter, HopeAdapter],
    scene_data: SceneData,
    frame_id: str,
    source_dir: Path,
) -> Tuple[List[ObjectInfo], List[str]]:
    scene_id = scene_data.scene_id
    mapping_data = get_mapping()
    
    try:
        frame_idx = int(frame_id)
    except ValueError:
        frame_idx = 0
    
    categories_dir = source_dir / "categories"
    reference_names = []
    
    if categories_dir.exists():
        cat_files = list(categories_dir.glob(f"*_{scene_id}_categories.json"))
        if not cat_files:
            cat_files = list(categories_dir.glob(f"{scene_id}_categories.json"))
        
        if cat_files:
            cat_path = str(cat_files[0])
            if cat_path not in CATEGORY_JSON_CACHE:
                with open(cat_path, 'r', encoding="utf-8") as f:
                    CATEGORY_JSON_CACHE[cat_path] = json.load(f)
            
            cat_data = CATEGORY_JSON_CACHE[cat_path]
            safe_row_idx = min(frame_idx, len(cat_data) - 1)
            raw_classes = cat_data[safe_row_idx] if cat_data else []
            
            mapped_objects = set()
            for rc in raw_classes:
                mapped_name = mapping_data.get(rc, rc)
                mapped_objects.add(mapped_name)
            reference_names = list(mapped_objects)
    
    reference_objects = []
    if reference_names:
        for obj in scene_data.objects:
            obj_mapped_name = mapping_data.get(obj.class_name, obj.class_name)
            if obj_mapped_name in reference_names:
                reference_objects.append(obj)
    
    return reference_objects, reference_names


# ===================== 2. 空间几何计算 (8向最大间隙法) =====================
def get_camera_aabb(corners_world: np.ndarray, E_w2c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    corners_homo = np.concatenate([corners_world, np.ones((corners_world.shape[0], 1))], axis=1)
    corners_cam = (E_w2c @ corners_homo.T).T[:, :3]
    return corners_cam.min(axis=0), corners_cam.max(axis=0)

def get_2d_bbox(corners_world: np.ndarray, E_w2c: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    corners_homo = np.concatenate([corners_world, np.ones((corners_world.shape[0], 1))], axis=1)
    corners_cam = (E_w2c @ corners_homo.T).T[:, :3]
    corners_img = (K @ corners_cam.T).T
    
    z = corners_img[:, 2:3]
    z[z == 0] = 1e-6 
    
    corners_img = corners_img[:, :2] / z
    return corners_img.min(axis=0), corners_img.max(axis=0)

def check_1d_overlap(min1, max1, min2, max2, margin=0.0):
    return (min1 - margin) < max2 and (max1 + margin) > min2

# 【新增】计算两个包围盒中心点的欧式距离
def center_distance(min1: np.ndarray, max1: np.ndarray, min2: np.ndarray, max2: np.ndarray) -> float:
    center1 = (min1 + max1) / 2.0
    center2 = (min2 + max2) / 2.0
    return float(np.linalg.norm(center1 - center2))

def bbox_distance(min1: np.ndarray, max1: np.ndarray, min2: np.ndarray, max2: np.ndarray) -> float:
    delta = np.maximum(0.0, np.maximum(min2 - max1, min1 - max2))
    return float(np.linalg.norm(delta))

def find_nearest_reference(
    target_corners_world: np.ndarray,
    reference_objects: List[ObjectInfo],
    camera: CameraParams,
    exclude_id: str = None,  
) -> Tuple[ObjectInfo, str, float]:
    # 物理3D测距 + 【极其严格的2D可见性过滤】 + 遮挡检测 + 8向视觉方位
    if not reference_objects:
        return None, "near", float('inf')
    
    E_w2c = np.linalg.inv(np.asarray(camera.E_c2w, dtype=np.float64))
    K = camera.K

    # 智能判断图像尺寸
    est_w = int(K[0, 2] * 2)
    est_h = int(K[1, 2] * 2)
    if abs(est_w - 640) < 100 or abs(est_h - 480) < 100:
        img_w, img_h = 640, 480
    elif abs(est_w - 1096) < 150 or abs(est_h - 852) < 150:
        img_w, img_h = 1096, 852
    else:
        img_w, img_h = max(640, est_w), max(480, est_h)

    t_min_c, t_max_c = get_camera_aabb(target_corners_world, E_w2c)
    
    # ===================== 【修复：安全地预计算所有物体信息】 =====================
    # 使用字典存储，通过 obj_id 索引，避免列表索引错位
    obj_info_map = {} 
    for obj in reference_objects:
        obj_corners_world = transform_points(_get_bbox_corners(obj.bbox3d_canonical), obj.pose_world)
        obj_min_2d, obj_max_2d = get_2d_bbox(obj_corners_world, E_w2c, K)
        obj_depth = get_camera_aabb(obj_corners_world, E_w2c)[0][2]
        obj_info_map[obj.obj_id] = {
            "min_2d": obj_min_2d,
            "max_2d": obj_max_2d,
            "depth": obj_depth,
            "obj": obj
        }
    # ==================================================================================

    valid_candidates = []
    for ref in reference_objects:
        if exclude_id is not None and ref.obj_id == exclude_id:
            continue

        # 1. 计算3D距离
        ref_corners_world = transform_points(_get_bbox_corners(ref.bbox3d_canonical), ref.pose_world)
        r_min_c, r_max_c = get_camera_aabb(ref_corners_world, E_w2c)
        dist = bbox_distance(t_min_c, t_max_c, r_min_c, r_max_c)

        # 2. 计算2D包围盒
        r_min_2d, r_max_2d = get_2d_bbox(ref_corners_world, E_w2c, K)
        box_w = max(0, r_max_2d[0] - r_min_2d[0])
        box_h = max(0, r_max_2d[1] - r_min_2d[1])
        area = box_w * box_h

        # 3. 【铁律1：物体必须和画面有交集】
        inter_xmin = max(r_min_2d[0], 0)
        inter_ymin = max(r_min_2d[1], 0)
        inter_xmax = min(r_max_2d[0], img_w)
        inter_ymax = min(r_max_2d[1], img_h)
        inter_w = max(0, inter_xmax - inter_xmin)
        inter_h = max(0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        if inter_area <= 0:
            continue  # 完全在画面外，直接排除

        # 4. 【铁律2：物体在画面内的面积必须足够大】
        area_thresh = 2500 if img_w < 800 else 5000
        if inter_area < area_thresh:
            continue  # 在画面里只露出一点点，直接排除

        # 5. 【铁律3：物体的可见比例必须足够高】
        visibility_ratio = inter_area / (area + 1e-6)
        if visibility_ratio < 0.4:
            continue  # 超过60%的部分在画面外或被裁剪，直接排除

        # 6. 【修复：安全的遮挡检测】
        occluded_area = 0.0
        ref_depth = obj_info_map[ref.obj_id]["depth"]
        
        for other_id, other_info in obj_info_map.items():
            if other_id == ref.obj_id:
                continue
            if other_id == exclude_id:
                continue
                
            other_depth = other_info["depth"]
            # 只有比它离相机更近的物体才会挡住它
            if other_depth >= ref_depth:
                continue
            
            other_min = other_info["min_2d"]
            other_max = other_info["max_2d"]
            
            # 计算两个物体2D包围盒的重叠面积
            overlap_xmin = max(r_min_2d[0], other_min[0])
            overlap_ymin = max(r_min_2d[1], other_min[1])
            overlap_xmax = min(r_max_2d[0], other_max[0])
            overlap_ymax = min(r_max_2d[1], other_max[1])
            overlap_w = max(0, overlap_xmax - overlap_xmin)
            overlap_h = max(0, overlap_ymax - overlap_ymin)
            occluded_area += overlap_w * overlap_h

        occlusion_ratio = occluded_area / (area + 1e-6)

        if occlusion_ratio < 0.5:
            # 评分 = 1/距离，这样距离越近分数越高
            # 【修改】使用中心点距离来选择参照物，更符合人类直觉
            center_dist = center_distance(t_min_c, t_max_c, r_min_c, r_max_c)
            score = 1.0 / (center_dist + 1e-5)
            valid_candidates.append((score, center_dist, ref))

    if not valid_candidates:
        return None, "near", float('inf')
    # 选择前3个评分最高（距离最近）的物体作为候选
    valid_candidates.sort(key=lambda x: x[0], reverse=True)
    top3_candidates = valid_candidates[:3]

# 为每个候选计算对应的方位关系
    all_candidates = []
    for score, dist, ref in top3_candidates:
        ref_corners_world = transform_points(_get_bbox_corners(ref.bbox3d_canonical), ref.pose_world)

    # 2. 判断特殊的上下叠加交互 (真实世界 Z 轴)
        t_world_min = target_corners_world.min(axis=0)
        t_world_max = target_corners_world.max(axis=0)
        r_world_min = ref_corners_world.min(axis=0)
        r_world_max = ref_corners_world.max(axis=0)

        overlap_x = check_1d_overlap(t_world_min[0], t_world_max[0], r_world_min[0], r_world_max[0])
        overlap_y = check_1d_overlap(t_world_min[1], t_world_max[1], r_world_min[1], r_world_max[1])

        if overlap_x and overlap_y:
            if t_world_min[2] >= r_world_max[2] + 0.05:
                relation = "the top of"
            elif t_world_max[2] <= r_world_min[2] - 0.05:
                relation = "below"
            else:
                # 3. 基于 2D 相机视角的视觉方位判断
                t_min_2d, t_max_2d = get_2d_bbox(target_corners_world, E_w2c, K)
                r_min_2d, r_max_2d = get_2d_bbox(ref_corners_world, E_w2c, K)

                gaps_2d = {
                    "the left of": r_min_2d[0] - t_max_2d[0],
                    "the right of": t_min_2d[0] - r_max_2d[0],
                    "behind": r_min_2d[1] - t_max_2d[1],
                    "in front of": t_min_2d[1] - r_max_2d[1]
                }
            
                sorted_gaps = sorted(gaps_2d.items(), key=lambda item: item[1], reverse=True)
                best_dir, best_val = sorted_gaps[0]
                second_dir, second_val = sorted_gaps[1]
                relation = best_dir

                x_axes = {"the left of", "the right of"}
                y_axes = {"behind", "in front of"}

                if (best_val - second_val) < DIAGONAL_THRESHOLD:
                    if (best_dir in x_axes and second_dir in y_axes) or (best_dir in y_axes and second_dir in x_axes):
                        dir_set = {best_dir, second_dir}
                        if dir_set == {"the left of", "in front of"}:
                            relation = "the front left of"
                        elif dir_set == {"the right of", "in front of"}:
                            relation = "the front right of"
                        elif dir_set == {"the left of", "behind"}:
                            relation = "the back left of"
                        elif dir_set == {"the right of", "behind"}:
                            relation = "the back right of"
        else:
            # 3. 基于 2D 相机视角的视觉方位判断
            t_min_2d, t_max_2d = get_2d_bbox(target_corners_world, E_w2c, K)
            r_min_2d, r_max_2d = get_2d_bbox(ref_corners_world, E_w2c, K)

            gaps_2d = {
                "the left of": r_min_2d[0] - t_max_2d[0],
                "the right of": t_min_2d[0] - r_max_2d[0],
                "behind": r_min_2d[1] - t_max_2d[1],
                "in front of": t_min_2d[1] - r_max_2d[1]
            }
        
            sorted_gaps = sorted(gaps_2d.items(), key=lambda item: item[1], reverse=True)
            best_dir, best_val = sorted_gaps[0]
            second_dir, second_val = sorted_gaps[1]
            relation = best_dir

            x_axes = {"the left of", "the right of"}
            y_axes = {"behind", "in front of"}

            if (best_val - second_val) < DIAGONAL_THRESHOLD:
                if (best_dir in x_axes and second_dir in y_axes) or (best_dir in y_axes and second_dir in x_axes):
                    dir_set = {best_dir, second_dir}
                    if dir_set == {"the left of", "in front of"}:
                        relation = "the front left of"
                    elif dir_set == {"the right of", "in front of"}:
                        relation = "the front right of"
                    elif dir_set == {"the left of", "behind"}:
                        relation = "the back left of"
                    elif dir_set == {"the right of", "behind"}:
                        relation = "the back right of"

        all_candidates.append((ref, relation, dist))

    if not all_candidates:
        return [(None, "near", float('inf'))]

    return all_candidates

def calculate_spatial_relation(
    target_corners_world: np.ndarray,
    reference_objects: List[ObjectInfo],
    camera: CameraParams,
    exclude_id: str = None,  
) -> List[Tuple[str, str]]:
    # 现在返回多个候选 (relation, ref_name)
    all_candidates = find_nearest_reference(
        target_corners_world, reference_objects, camera, exclude_id
    )
    
    mapping_data = get_mapping()
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
) -> str:
    canonical_aabb = np.asarray(sample_record["canonical_aabb_object"], dtype=np.float64)
    canonical_corners = _get_bbox_corners(canonical_aabb)
    
    original_pose = np.asarray(sample_record["original_pose_world"], dtype=np.float64)
    original_corners = transform_points(canonical_corners, original_pose)
    
    placement_pose = np.asarray(sample_record["transform_world"], dtype=np.float64)
    placement_corners = transform_points(canonical_corners, placement_pose)
    
    target_obj_id = sample_record.get('object_id')
    if not target_obj_id:
        target_obj_id = sample_record.get('sample_id').split('_')[2]
        
    # 1. 先正常计算原始位置的描述（修复：取候选列表第一个元素，增加空值保护）
    original_relations = calculate_spatial_relation(
        original_corners, reference_objects, scene_data.camera, exclude_id=target_obj_id
    )
    rel_original, ref_a_name = original_relations[0] if original_relations else ("near", "the reference object")

    # 2. 计算目标位置的描述（修复：取候选列表第一个元素，增加空值保护）
    placement_relations = calculate_spatial_relation(
        placement_corners, reference_objects, scene_data.camera, exclude_id=None
    )
    rel_placement, ref_b_name = placement_relations[0] if placement_relations else ("near", "the reference object")

    # 【关键修复】只有当「参照物相同 AND 方位也相同」时，才强制换一个
    if (ref_b_name == ref_a_name) and (rel_placement == rel_original):
        # 找到 ref_a 对应的 object_id
        ref_a_id = None
        mapping_data = get_mapping()
        for obj in reference_objects:
            if mapping_data.get(obj.class_name, obj.class_name) == ref_a_name:
                ref_a_id = obj.obj_id
                break
        
        # 把 ref_a 从候选列表里拿掉，重新选一个
        if ref_a_id is not None:
            filtered_refs = [obj for obj in reference_objects if obj.obj_id != ref_a_id]
            if filtered_refs:
                new_placement_relations = calculate_spatial_relation(
                    placement_corners, filtered_refs, scene_data.camera, exclude_id=target_obj_id
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
    parser = argparse.ArgumentParser(description="以图片文件为索引，自动生成对应的 placement 标注")
    parser.add_argument("--image-dir", required=True, type=Path, help="可视化的图片目录 (例如 outputs/placement_rgb_bbox_vis)")
    parser.add_argument("--outputs-base", type=Path, default=PROJECT_ROOT / "outputs", help="数据集的输出基准目录")
    parser.add_argument("--output-dir", required=True, type=Path, help="标注文件输出目录")
    parser.add_argument("--limit", type=int, default=None, help="仅标注前 N 个样本")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的标注文件")
    return parser

def infer_config_path(source_name: str) -> Path:
    if "housecat" in source_name.lower():
        return PROJECT_ROOT / "configs/annotation/placement_housecat6d.yaml"
    return PROJECT_ROOT / "configs/annotation/placement.yaml"

def load_scene_cached(scene_cache: Dict, adapter, source_dir: Path, scene_id: str, frame_id: str):
    key = (source_dir.name, scene_id, frame_id)
    if key not in scene_cache:
        scene_path = Path(adapter.root_dir) / scene_id
        scene_cache[key] = adapter.load_scene(str(scene_path), frame_id)
    return scene_cache[key]

def auto_label_from_images(
    image_dir: Path, outputs_base: Path, output_dir: Path, limit: int = None, overwrite: bool = False
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    if not image_files:
        print(f"未在 {image_dir} 中找到任何图片！")
        return 0

    print(f"找到 {len(image_files)} 张图片，开始构建数据索引...")

    lookup_table = {}  
    adapters = {}
    
    source_names = set()
    for img_file in image_files:
        parts = img_file.stem.split('__', 1)
        if len(parts) == 2:
            source_names.add(parts[0])
            
    for source_name in source_names:
        source_dir = outputs_base / source_name
        if not source_dir.exists():
            continue
            
        config_path = infer_config_path(source_name)
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        ds_cfg = cfg.get("dataset", {})
        ds_type = ds_cfg.get("type", "hope")
        
        if ds_type == "hope":
            adapters[source_name] = HopeAdapter(
                root_dir=ds_cfg["root_dir"], mesh_dir=ds_cfg.get("mesh_dir"), frame_step=ds_cfg.get("frame_step", 60)
            )
        elif ds_type == "housecat6d":
            adapters[source_name] = HouseCat6DAdapter(
                root_dir=ds_cfg["root_dir"], frame_step=ds_cfg.get("frame_step", 60)
            )
            
        lookup_table[source_name] = {}
        samples_dir = source_dir / "samples"
        if samples_dir.exists():
            for json_file in samples_dir.glob("*.json"):
                with json_file.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                    for record in payload.get("samples", []):
                        lookup_table[source_name][record["sample_id"]] = record

    print("数据索引构建完成，开始执行标注...")

    scene_cache = {}
    all_labels = []
    labeled = 0

    for img_file in image_files:
        parts = img_file.stem.split('__', 1)
        if len(parts) != 2:
            continue
            
        source_name = parts[0]  
        sample_id = parts[1]
            
        if source_name not in lookup_table or sample_id not in lookup_table[source_name]:
            continue
            
        sample_record = lookup_table[source_name][sample_id]
        source_dir = outputs_base / source_name
        adapter = adapters[source_name]

        scene = load_scene_cached(
            scene_cache, adapter, source_dir, str(sample_record["scene_id"]), str(sample_record["frame_id"])
        )
        
        target_object_name, is_found_target = get_target_object_name(sample_record, source_dir)
        reference_objects, _ = get_reference_objects_with_names(
            adapter, scene, str(sample_record["frame_id"]), source_dir
        )
        label = generate_label(sample_record, scene, reference_objects, target_object_name)
            
        all_labels.append({
            "image_filename": img_file.name,
            "sample_id": sample_id,
            "source_name": source_name,
            "target_object_name": target_object_name,
            "is_found_target": is_found_target,
            "label": label,
        })
        
        labeled += 1
        if labeled % 50 == 0:
            print(f"已标注 {labeled} 张图片，最新: {img_file.name}")
        if limit is not None and labeled >= limit:
            break

    all_labels_path = output_dir / "all_labels.json"
    with all_labels_path.open("w", encoding="utf-8") as f:
        json.dump(all_labels, f, indent=2, ensure_ascii=False)

    # ==========================================
    # 【生成HTML 报告】
    # ==========================================                                                                        
        # ==========================================
    # 【简化版】只读 HTML 报告（无交互）
    # ==========================================
    if all_labels:
        print("\n正在生成只读 HTML 报告...")
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
            "<div class='container'>"
        ]
        
        for item in all_labels:
            img_rel_path = os.path.relpath(image_dir / item["image_filename"], output_dir)
            fname = item["image_filename"]
            html_lines.append(f"  <div class='card'>")
            html_lines.append(f"    <img src='{img_rel_path}' loading='lazy' />")
            html_lines.append(f"    <div class='info'>")
            html_lines.append(f"      <div class='filename'>📄 {fname}</div>")
            html_lines.append(f"      <div class='label'>👉 <span class='highlight'>{item['label']}</span></div>")
            html_lines.append("    </div></div>")
            
        html_lines.extend(["</div></body></html>"])
        report_path = output_dir / "report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_lines))
            
        print(f"✅ 只读 HTML 报告已生成！请使用 Web 服务查看。")

    # ==========================================
    # 保存 JSON 汇总
    # ==========================================
    all_labels_path = output_dir / "all_labels.json"
    with all_labels_path.open("w", encoding="utf-8") as f:
        json.dump(all_labels, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 标注完成")
    print(f"实际成功标注: {labeled} 个样本")
    
    return labeled


def main() -> None:
    args = build_parser().parse_args()
    image_dir = args.image_dir.resolve()
    outputs_base = args.outputs_base.resolve()
    output_dir = args.output_dir.resolve()
    
    labeled = auto_label_from_images(
        image_dir=image_dir, outputs_base=outputs_base, output_dir=output_dir, limit=args.limit, overwrite=args.overwrite,
    )
    
    print("\n✅ 标注完成")
    print(f"图片来源目录: {image_dir}")
    print(f"数据索引目录: {outputs_base}")
    print(f"实际成功标注: {labeled} 个样本")

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