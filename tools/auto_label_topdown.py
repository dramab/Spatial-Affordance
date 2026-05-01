#!/usr/bin/env python3
"""
tools/auto_label_topdown.py
---------------------------
面向从上至下或近似从上至下视角的 placement 空位框样本，生成自然语言标注。

该脚本不修改通用标注脚本 tools/auto_label.py，而是复用其数据加载、参照物可见性过滤、
遮挡估计和报告输出能力，仅替换空间方向关系：上下关系仍由 3D world box 判断，水平
方向关系按图像平面解释。

======================== 用法示例 ========================
conda run -n spatial python tools/auto_label_topdown.py \
    --image-dir outputs/placement_rgb_bbox_vis_dopose \
    --outputs-base outputs \
    --mapping configs/annotation/mappingv2.json \
    --output-dir outputs/auto_labels_topdown \
    --limit 50 \
    --axis-half-width-deg 30

conda run -n spatial python tools/auto_label_topdown.py \
    --image-dir outputs/placement_rgb_bbox_vis_dopose \
    --outputs-base outputs \
    --mapping configs/annotation/mappingv2.json \
    --output-dir outputs/auto_labels_topdown_selected \
    --sample-ids test_bin_000001_000000_obj_000011_3_p000

======================== 俯视方向约定 ========================
1. 图像右侧为 the right of，图像左侧为 the left of。
2. 图像下方为 in front of，图像上方为 behind。
3. 同时存在水平和垂直图像偏移时，生成 front right/front left/back right/back left。
4. 主方向半宽默认 30°，可通过 --axis-half-width-deg 调整；取值范围为 (0, 45)。
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.free_bbox.grid_ops import _get_bbox_corners
from src.annotation.free_bbox.datatypes import CameraParams, ObjectInfo, SceneData
from src.utils.coord_utils import transform_points
from tools.auto_label import (
    LABEL_TEMPLATE,
    LATERAL_DIRECTION_MIN_PX,
    MAX_OCCLUSION_RATIO,
    MAX_REFERENCE_CANDIDATES,
    build_reference_projection_info,
    build_sample_id_filter,
    build_source_indexes,
    collect_image_files,
    collect_source_names,
    compute_occlusion_ratio,
    describe_angle_relation,
    describe_vertical_relation,
    filter_image_files_by_sample_ids,
    get_mapping,
    get_reference_objects_with_names,
    get_target_object_name,
    infer_image_size_from_camera,
    load_scene_cached,
    parse_image_filename,
    passes_reference_visibility_filter,
    print_missing_sample_ids,
    project_box_center_to_pixel,
    save_all_labels,
    save_report_html,
)

TOPDOWN_DIRECTION_MIN_PX = LATERAL_DIRECTION_MIN_PX
TOPDOWN_AXIS_HALF_WIDTH_DEG = 30.0


def describe_topdown_horizontal_relation(
    target_corners_world: np.ndarray,
    ref_corners_world: np.ndarray,
    E_w2c: np.ndarray,
    K: np.ndarray,
    min_offset_px: float = TOPDOWN_DIRECTION_MIN_PX,
    axis_half_width_deg: float = TOPDOWN_AXIS_HALF_WIDTH_DEG,
) -> str:
    """
    用法: relation = describe_topdown_horizontal_relation(target_corners_world, ref_corners_world, E_w2c, K)
    作用: 基于俯视图像平面的中心点角度，判断目标相对参照物的水平方向。
    输入: target/ref 世界 box 角点、world->camera 外参、相机内参、中心距离阈值和主方向半宽。
    输出: str，俯视语义下的 8 向关系；中心过近时返回 "near"。
    """
    target_uv = project_box_center_to_pixel(target_corners_world, E_w2c, K)
    ref_uv = project_box_center_to_pixel(ref_corners_world, E_w2c, K)
    delta_uv = target_uv - ref_uv
    if float(np.linalg.norm(delta_uv)) < float(min_offset_px):
        return "near"

    angle_deg = float(np.degrees(np.arctan2(delta_uv[1], delta_uv[0])))
    return describe_angle_relation(angle_deg, axis_half_width_deg=axis_half_width_deg)


def describe_topdown_spatial_relation(
    target_corners_world: np.ndarray,
    ref_corners_world: np.ndarray,
    E_w2c: np.ndarray,
    K: np.ndarray,
    axis_half_width_deg: float = TOPDOWN_AXIS_HALF_WIDTH_DEG,
) -> str:
    """
    用法: relation = describe_topdown_spatial_relation(target_corners_world, ref_corners_world, E_w2c, K)
    作用: 优先用 3D box 判断上下关系，否则按俯视图像平面判断水平空间关系。
    输入: target/ref 世界 box 角点、world->camera 外参、相机内参和主方向半宽。
    输出: str，自动标注模板可直接使用的空间关系短语。
    """
    vertical_relation = describe_vertical_relation(target_corners_world, ref_corners_world)
    if vertical_relation is not None:
        return vertical_relation
    return describe_topdown_horizontal_relation(
        target_corners_world,
        ref_corners_world,
        E_w2c,
        K,
        axis_half_width_deg=axis_half_width_deg,
    )


def find_nearest_reference_topdown(
    target_corners_world: np.ndarray,
    reference_objects: List[ObjectInfo],
    camera: CameraParams,
    exclude_id: str = None,
    axis_half_width_deg: float = TOPDOWN_AXIS_HALF_WIDTH_DEG,
) -> List[Tuple[Optional[ObjectInfo], str, float]]:
    """
    用法: candidates = find_nearest_reference_topdown(target_corners_world, reference_objects, camera, exclude_id)
    作用: 为俯视标注筛选可见参照物，并按上下关系优先、像素中心距离最近排序。
    输入: 目标 world box 角点、参照物列表、相机参数、可选排除 obj_id 和主方向半宽。
    输出: list[(ObjectInfo|None, relation, distance_px)]，最多返回若干候选参照物。
    """
    if not reference_objects:
        return [(None, "near", float("inf"))]

    E_w2c = np.linalg.inv(np.asarray(camera.E_c2w, dtype=np.float64))
    K = camera.K
    img_w, img_h = infer_image_size_from_camera(camera)
    target_uv = project_box_center_to_pixel(target_corners_world, E_w2c, K)
    obj_info_map = build_reference_projection_info(reference_objects, E_w2c, K)

    valid_candidates = []
    for ref in reference_objects:
        if exclude_id is not None and ref.obj_id == exclude_id:
            continue

        ref_info = obj_info_map[ref.obj_id]
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

        ref_uv = project_box_center_to_pixel(ref_info["corners_world"], E_w2c, K)
        pixel_dist = float(np.linalg.norm(target_uv - ref_uv))
        relation = describe_topdown_spatial_relation(
            target_corners_world,
            ref_info["corners_world"],
            E_w2c,
            K,
            axis_half_width_deg=axis_half_width_deg,
        )
        vertical_priority = 1 if relation in ("the top of", "below") else 0
        valid_candidates.append((vertical_priority, pixel_dist, ref, relation))

    if not valid_candidates:
        return [(None, "near", float("inf"))]

    valid_candidates.sort(key=lambda item: (-item[0], item[1]))
    top_candidates = valid_candidates[:MAX_REFERENCE_CANDIDATES]
    return [(ref, relation, dist) for _, dist, ref, relation in top_candidates]


def calculate_topdown_spatial_relation(
    target_corners_world: np.ndarray,
    reference_objects: List[ObjectInfo],
    camera: CameraParams,
    exclude_id: str = None,
    mapping_data: dict = None,
    axis_half_width_deg: float = TOPDOWN_AXIS_HALF_WIDTH_DEG,
) -> List[Tuple[str, str, Optional[str]]]:
    """
    用法: relations = calculate_topdown_spatial_relation(target_corners_world, refs, camera, exclude_id, mapping_data)
    作用: 将俯视参照物候选转换为自然语言关系、参照物名称和参照物 id。
    输入: 目标 world box 角点、参照物列表、相机参数、可选排除 obj_id、类别映射表和主方向半宽。
    输出: list[(relation, ref_name, ref_id)]，用于生成移动指令。
    """
    candidates = find_nearest_reference_topdown(
        target_corners_world,
        reference_objects,
        camera,
        exclude_id=exclude_id,
        axis_half_width_deg=axis_half_width_deg,
    )
    mapping_data = mapping_data or {}

    result = []
    for ref, relation, _ in candidates:
        if ref is None:
            result.append(("nowhere", "nothing", None))
        else:
            ref_name = mapping_data.get(ref.class_name, ref.class_name)
            result.append((relation, ref_name, ref.obj_id))
    return result


def choose_distinct_placement_relation(
    placement_relations: List[Tuple[str, str, Optional[str]]],
    rel_original: str,
    ref_a_id: Optional[str],
) -> Tuple[str, str, Optional[str]]:
    """
    用法: rel, name, ref_id = choose_distinct_placement_relation(placement_relations, rel_original, ref_a_id)
    作用: 当目标位置与原始位置描述完全相同时，优先选择下一个可用参照物。
    输入: 目标位置候选列表、原始关系短语和原始参照物 id。
    输出: tuple，最终使用的目标位置关系、参照物名称和参照物 id。
    """
    if not placement_relations:
        return "near", "the reference object", None

    first_relation = placement_relations[0]
    if first_relation[0] != rel_original or first_relation[2] != ref_a_id:
        return first_relation

    for candidate in placement_relations[1:]:
        if candidate[2] != ref_a_id or candidate[0] != rel_original:
            return candidate
    return first_relation


def generate_topdown_label(
    sample_record: dict,
    scene_data: SceneData,
    reference_objects: List[ObjectInfo],
    target_object_name: str,
    mapping_data: dict,
    axis_half_width_deg: float = TOPDOWN_AXIS_HALF_WIDTH_DEG,
) -> str:
    """
    用法: label = generate_topdown_label(sample_record, scene_data, reference_objects, target_object_name, mapping_data)
    作用: 为单个 placement sample 生成俯视语义下的自然语言移动指令。
    输入: sample 记录、场景数据、参照物列表、目标物体展示名、类别映射表和主方向半宽。
    输出: str，完整自然语言标注。
    """
    canonical_aabb = np.asarray(sample_record["canonical_aabb_object"], dtype=np.float64)
    canonical_corners = _get_bbox_corners(canonical_aabb)

    original_pose = np.asarray(sample_record["original_pose_world"], dtype=np.float64)
    original_corners = transform_points(canonical_corners, original_pose)

    placement_pose = np.asarray(sample_record["transform_world"], dtype=np.float64)
    placement_corners = transform_points(canonical_corners, placement_pose)

    target_obj_id = sample_record.get("object_id")
    if not target_obj_id:
        target_obj_id = sample_record.get("sample_id").split("_")[2]

    original_relations = calculate_topdown_spatial_relation(
        original_corners,
        reference_objects,
        scene_data.camera,
        exclude_id=target_obj_id,
        mapping_data=mapping_data,
        axis_half_width_deg=axis_half_width_deg,
    )
    rel_original, ref_a_name, ref_a_id = (
        original_relations[0] if original_relations else ("near", "the reference object", None)
    )

    placement_relations = calculate_topdown_spatial_relation(
        placement_corners,
        reference_objects,
        scene_data.camera,
        exclude_id=target_obj_id,
        mapping_data=mapping_data,
        axis_half_width_deg=axis_half_width_deg,
    )
    rel_placement, ref_b_name, _ = choose_distinct_placement_relation(
        placement_relations,
        rel_original,
        ref_a_id,
    )

    return LABEL_TEMPLATE.format(
        object_name=target_object_name,
        rel_original=rel_original,
        ref_a_name=ref_a_name,
        rel_placement=rel_placement,
        ref_b_name=ref_b_name,
    )


def build_parser() -> argparse.ArgumentParser:
    """
    用法: parser = build_parser()
    作用: 构建俯视自动标注脚本的命令行参数解析器。
    输入: 无。
    输出: argparse.ArgumentParser。
    """
    parser = argparse.ArgumentParser(description="为俯视或近俯视 placement 样本自动生成标注")
    parser.add_argument("--image-dir", required=True, type=Path, help="可视化图片目录")
    parser.add_argument("--outputs-base", type=Path, default=PROJECT_ROOT / "outputs", help="placement 输出根目录")
    parser.add_argument("--output-dir", required=True, type=Path, help="标注 JSON 和 HTML 报告输出目录")
    parser.add_argument("--mapping", type=Path, default=PROJECT_ROOT / "configs/annotation/mapping.json", help="类别名称映射 JSON")
    parser.add_argument("--limit", type=int, default=None, help="仅标注前 N 个样本")
    parser.add_argument("--sample-ids", nargs="+", default=None, help="仅标注指定 sample_id")
    parser.add_argument("--sample-ids-file", type=Path, default=None, help="从文本文件读取 sample_id，每行一个")
    parser.add_argument(
        "--axis-half-width-deg",
        type=float,
        default=TOPDOWN_AXIS_HALF_WIDTH_DEG,
        help="俯视图像平面主方向扇区半宽，默认 30 度",
    )
    parser.add_argument("--overwrite", action="store_true", help="兼容旧调用，当前 JSON/HTML 输出会直接刷新")
    return parser


def build_topdown_label_record(
    img_file: Path,
    source_name: str,
    sample_id: str,
    sample_record: dict,
    source_dir: Path,
    adapter,
    scene_cache: Dict,
    mapping_data: dict,
    axis_half_width_deg: float,
) -> dict:
    """
    用法: record = build_topdown_label_record(img_file, source_name, sample_id, sample_record, source_dir, adapter, scene_cache, mapping_data)
    作用: 为单张俯视样本图片生成 all_labels.json 中的一条记录。
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
    label = generate_topdown_label(
        sample_record,
        scene,
        reference_objects,
        target_object_name,
        mapping_data,
        axis_half_width_deg=axis_half_width_deg,
    )
    return {
        "image_filename": img_file.name,
        "sample_id": sample_id,
        "source_name": source_name,
        "target_object_name": target_object_name,
        "is_found_target": is_found_target,
        "label": label,
    }


def auto_label_topdown_from_images(
    image_dir: Path,
    outputs_base: Path,
    output_dir: Path,
    limit: int = None,
    overwrite: bool = False,
    sample_id_filter: Optional[Set[str]] = None,
    mapping_path: str = None,
    axis_half_width_deg: float = TOPDOWN_AXIS_HALF_WIDTH_DEG,
) -> int:
    """
    用法: count = auto_label_topdown_from_images(image_dir, outputs_base, output_dir, sample_id_filter={"sample_a"})
    作用: 以可视化图片为索引，为输入样本按俯视语义生成自动标注。
    输入: 图片目录、placement 输出根目录、标注输出目录、limit、overwrite、sample_id 过滤集合和映射文件路径。
    输出: int，实际成功标注的样本数量。
    """
    del overwrite
    if not 0.0 < float(axis_half_width_deg) < 45.0:
        raise ValueError("--axis-half-width-deg 必须在 (0, 45) 范围内")
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

    print(f"找到 {len(image_files)} 张图片，开始构建俯视标注数据索引...")
    lookup_table, adapters = build_source_indexes(collect_source_names(image_files), outputs_base)
    print("数据索引构建完成，开始执行俯视标注...")

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
            build_topdown_label_record(
                img_file=img_file,
                source_name=source_name,
                sample_id=sample_id,
                sample_record=lookup_table[source_name][sample_id],
                source_dir=outputs_base / source_name,
                adapter=adapters[source_name],
                scene_cache=scene_cache,
                mapping_data=mapping_data,
                axis_half_width_deg=axis_half_width_deg,
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
        print("只读 HTML 报告已生成，请使用 Web 服务查看。")

    print("\n俯视标注完成")
    print(f"实际成功标注: {labeled} 个样本")
    return labeled


def main() -> None:
    """
    用法: python tools/auto_label_topdown.py --image-dir outputs/placement_rgb_bbox_vis --output-dir outputs/auto_labels_topdown
    作用: 解析命令行参数并执行俯视自动标注主流程。
    输入: 无，参数来自命令行。
    输出: None，在终端打印处理结果。
    """
    args = build_parser().parse_args()
    image_dir = args.image_dir.resolve()
    outputs_base = args.outputs_base.resolve()
    output_dir = args.output_dir.resolve()
    mapping_path = str(args.mapping.resolve()) if args.mapping else None
    sample_id_filter = build_sample_id_filter(args.sample_ids, args.sample_ids_file)

    auto_label_topdown_from_images(
        image_dir=image_dir,
        outputs_base=outputs_base,
        output_dir=output_dir,
        limit=args.limit,
        overwrite=args.overwrite,
        sample_id_filter=sample_id_filter,
        mapping_path=mapping_path,
        axis_half_width_deg=args.axis_half_width_deg,
    )

    print(f"图片来源目录: {image_dir}")
    print(f"数据索引目录: {outputs_base}")


if __name__ == "__main__":
    main()
