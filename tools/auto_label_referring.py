#!/usr/bin/env python3
"""
tools/auto_label_referring.py
-----------------------------
为同类多实例场景生成可指代的 placement 自然语言标注。

本脚本复用 tools/auto_label.py 的图片索引、样本读取、空间关系计算与报告输出逻辑，
并在目标物体和参照物名称前加入视觉序数描述，解决多个同名物体导致的指代歧义。

======================== 用法示例 ========================
python tools/auto_label_referring.py \
    --image-dir /data/jiajun.xie/Spatial-Affordance/outputs/Refering_difficult \
    --outputs-base /data/jiajun.xie/Spatial-Affordance/outputs \
    --mapping configs/annotation/mappingv3.json \
    --output-dir /data/jiajun.xie/Spatial-Affordance/outputs/auto_labels_referring_difficult \
    --limit 20

python tools/auto_label_referring.py \
    --image-dir outputs/Refering_difficult \
    --outputs-base outputs \
    --mapping configs/annotation/mappingv3.json \
    --output-dir outputs/auto_labels_referring_difficult

======================== 输出内容 ========================
1. all_labels.json，字段与 tools/auto_label.py 对齐。
2. report.html，单文件离线网页，引用图片和文本，可通过 Web 服务查看。
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

PROJECT_ROOT = Path("/data/jiajun.xie/Spatial-Affordance")
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.free_bbox.datatypes import ObjectInfo, SceneData
from src.annotation.free_bbox.grid_ops import _get_bbox_corners
from src.utils.coord_utils import transform_points
from tools.auto_label import (
    LABEL_TEMPLATE,
    MAPPING_PATH,
    build_sample_id_filter,
    build_source_indexes,
    collect_image_files,
    collect_source_names,
    filter_image_files_by_sample_ids,
    find_nearest_reference,
    get_mapping,
    get_object_corners_world,
    get_target_object_name,
    load_scene_cached,
    parse_image_filename,
    print_missing_sample_ids,
    project_box_center_to_camera,
    project_box_center_to_pixel,
    save_all_labels,
    save_report_html,
)

AXIS_TIE_EPS_PX = 12.0
MAPPING_V3_PATH = PROJECT_ROOT / "configs/annotation/mappingv3.json"
MAPPING_V2_PATH = PROJECT_ROOT / "configs/annotation/mappingv2.json"
DEFAULT_MAPPING_PATH = (
    MAPPING_V3_PATH
    if MAPPING_V3_PATH.exists()
    else MAPPING_V2_PATH if MAPPING_V2_PATH.exists() else MAPPING_PATH
)


def build_parser() -> argparse.ArgumentParser:
    """
    用法: parser = build_parser()
    作用: 构建同类多实例指代标注脚本的命令行参数解析器。
    输入: 无。
    输出: argparse.ArgumentParser。
    """
    parser = argparse.ArgumentParser(description="为同类多实例 placement 图片生成可指代标注")
    parser.add_argument("--image-dir", required=True, type=Path, help="可视化图片目录，例如 outputs/Refering_difficult")
    parser.add_argument("--outputs-base", type=Path, default=PROJECT_ROOT / "outputs", help="placement 输出根目录")
    parser.add_argument("--output-dir", required=True, type=Path, help="all_labels.json 和 report.html 输出目录")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH, help="类别名称映射文件路径")
    parser.add_argument("--limit", type=int, default=None, help="仅标注前 N 个样本")
    parser.add_argument("--sample-ids", nargs="+", default=None, help="仅标注指定 sample_id，可一次传入多个")
    parser.add_argument("--sample-ids-file", type=Path, default=None, help="从文本文件读取 sample_id，每行一个")
    parser.add_argument("--overwrite", action="store_true", help="兼容旧调用，当前输出会直接刷新")
    return parser


def ordinal_word(index: int) -> str:
    """
    用法: word = ordinal_word(2)
    作用: 将从 1 开始的序号转为英文序数词。
    输入: index 为正整数。
    输出: str，英文序数词。
    """
    fixed_words = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eighth",
        9: "ninth",
        10: "tenth",
    }
    if index in fixed_words:
        return fixed_words[index]
    suffix = "th"
    if index % 100 not in (11, 12, 13):
        if index % 10 == 1:
            suffix = "st"
        elif index % 10 == 2:
            suffix = "nd"
        elif index % 10 == 3:
            suffix = "rd"
    return f"{index}{suffix}"


def has_stable_axis_order(values: List[float], tie_eps: float) -> bool:
    """
    用法: stable = has_stable_axis_order([10.0, 30.0], 12.0)
    作用: 判断一组投影坐标在某个轴上是否有足够间隔用于稳定排序。
    输入: values 为轴坐标列表；tie_eps 为相邻坐标最小间隔阈值。
    输出: bool，True 表示可用于生成视觉序数描述。
    """
    if len(values) <= 1:
        return False
    sorted_values = sorted(float(value) for value in values)
    min_gap = min(
        sorted_values[idx + 1] - sorted_values[idx]
        for idx in range(len(sorted_values) - 1)
    )
    return bool(min_gap > float(tie_eps))


def build_rank_phrase(rank: int, total: int, axis: str) -> str:
    """
    用法: phrase = build_rank_phrase(1, 3, "x")
    作用: 根据排序轴、名次和总数生成实例消歧短语。
    输入: rank 为从 0 开始的名次；total 为同名实例数量；axis 为 x、y 或 depth。
    输出: str，leftmost/middle/rightmost 等短语。
    """
    if axis == "x":
        first_phrase, last_phrase, origin_phrase = "leftmost", "rightmost", "from the left"
    elif axis == "y":
        first_phrase, last_phrase, origin_phrase = "topmost", "bottommost", "from the top"
    else:
        first_phrase, last_phrase, origin_phrase = "frontmost", "backmost", "from the front"

    if rank == 0:
        return first_phrase
    if rank == total - 1:
        return last_phrase
    if total % 2 == 1 and rank == total // 2:
        return "middle"
    return f"{ordinal_word(rank + 1)} {origin_phrase}"


def get_display_name(obj: ObjectInfo, mapping_data: dict) -> str:
    """
    用法: name = get_display_name(obj, mapping_data)
    作用: 获取物体类别映射后的展示名。
    输入: obj 为 ObjectInfo；mapping_data 为类别映射表。
    输出: str，展示名。
    """
    return mapping_data.get(obj.class_name, obj.class_name)


def build_object_descriptor_map(
    reference_objects: List[ObjectInfo],
    scene_data: SceneData,
    mapping_data: dict,
) -> Dict[str, str]:
    """
    用法: descriptors = build_object_descriptor_map(reference_objects, scene_data, mapping_data)
    作用: 为场景中每个物体生成唯一可指代名称。
    输入: 参照物列表、场景数据和类别映射表。
    输出: dict，键为 obj_id，值为带冠词和序数的实例描述。
    """
    E_w2c = np.linalg.inv(np.asarray(scene_data.camera.E_c2w, dtype=np.float64))
    K = scene_data.camera.K
    groups: Dict[str, List[Tuple[ObjectInfo, float, float, float]]] = {}

    for obj in reference_objects:
        corners_world = get_object_corners_world(obj)
        center_uv = project_box_center_to_pixel(corners_world, E_w2c, K)
        center_cam = project_box_center_to_camera(corners_world, E_w2c)
        display_name = get_display_name(obj, mapping_data)
        groups.setdefault(display_name, []).append(
            (obj, float(center_uv[0]), float(center_uv[1]), float(center_cam[2]))
        )

    descriptor_map = {}
    for display_name, items in groups.items():
        if len(items) == 1:
            descriptor_map[items[0][0].obj_id] = f"the {display_name}"
            continue

        if has_stable_axis_order([item[1] for item in items], AXIS_TIE_EPS_PX):
            axis = "x"
            sorted_items = sorted(items, key=lambda item: (item[1], item[2], item[3], item[0].obj_id))
        elif has_stable_axis_order([item[2] for item in items], AXIS_TIE_EPS_PX):
            axis = "y"
            sorted_items = sorted(items, key=lambda item: (item[2], item[1], item[3], item[0].obj_id))
        else:
            axis = "depth"
            sorted_items = sorted(items, key=lambda item: (item[3], item[1], item[2], item[0].obj_id))

        total = len(sorted_items)
        for rank, item in enumerate(sorted_items):
            obj = item[0]
            descriptor_map[obj.obj_id] = f"the {build_rank_phrase(rank, total, axis)} {display_name}"

    return descriptor_map


def get_target_obj_id(sample_record: dict) -> Optional[str]:
    """
    用法: obj_id = get_target_obj_id(sample_record)
    作用: 从 placement sample 中读取目标物体实例 id。
    输入: sample_record 为 placement 样本记录。
    输出: str 或 None，目标 obj_id。
    """
    if sample_record.get("object_id"):
        return sample_record["object_id"]
    sample_id = sample_record.get("sample_id")
    if not sample_id:
        return None
    marker = "_obj_"
    if marker in sample_id:
        after_marker = sample_id.split(marker, 1)[1]
        obj_parts = after_marker.rsplit("_p", 1)[0].split("_")
        if len(obj_parts) >= 2:
            return f"obj_{obj_parts[0]}_{obj_parts[1]}"
        if len(obj_parts) == 1:
            return f"obj_{obj_parts[0]}"
    return None


def get_descriptor_for_object(
    obj: Optional[ObjectInfo],
    descriptor_map: Dict[str, str],
    mapping_data: dict,
) -> str:
    """
    用法: desc = get_descriptor_for_object(obj, descriptor_map, mapping_data)
    作用: 获取参照物的实例级描述。
    输入: obj 为可选 ObjectInfo；descriptor_map 为 obj_id 到描述的映射；mapping_data 为类别映射表。
    输出: str，实例级描述；obj 为空时返回 nothing。
    """
    if obj is None:
        return "nothing"
    if obj.obj_id in descriptor_map:
        return descriptor_map[obj.obj_id]
    return f"the {get_display_name(obj, mapping_data)}"


def find_first_reference(
    target_corners_world: np.ndarray,
    reference_objects: List[ObjectInfo],
    scene_data: SceneData,
    exclude_id: Optional[str],
) -> Tuple[Optional[ObjectInfo], str]:
    """
    用法: ref, relation = find_first_reference(corners, refs, scene, exclude_id)
    作用: 复用 auto_label 的候选排序逻辑，返回最优参照物和空间关系。
    输入: 目标角点、候选参照物、场景数据和可选排除 obj_id。
    输出: (ObjectInfo|None, relation)。
    """
    candidates = find_nearest_reference(
        target_corners_world,
        reference_objects,
        scene_data.camera,
        exclude_id=exclude_id,
    )
    if not candidates:
        return None, "near"
    ref, relation, _ = candidates[0]
    return ref, relation


def resolve_placement_reference(
    placement_corners: np.ndarray,
    reference_objects: List[ObjectInfo],
    scene_data: SceneData,
    target_obj_id: Optional[str],
) -> Tuple[Optional[ObjectInfo], str]:
    """
    用法: ref, relation = resolve_placement_reference(corners, refs, scene, target_obj_id)
    作用: 查找目标位置参照物，仅在发生自指时排除目标实例并重算。
    输入: 目标位置角点、参照物列表、场景数据和目标 obj_id。
    输出: (ObjectInfo|None, relation)。
    """
    ref, relation = find_first_reference(
        placement_corners,
        reference_objects,
        scene_data,
        exclude_id=None,
    )
    if ref is None or target_obj_id is None or ref.obj_id != target_obj_id:
        return ref, relation

    filtered_refs = [obj for obj in reference_objects if obj.obj_id != target_obj_id]
    if not filtered_refs:
        return ref, relation
    return find_first_reference(
        placement_corners,
        filtered_refs,
        scene_data,
        exclude_id=target_obj_id,
    )


def generate_referring_label(
    sample_record: dict,
    scene_data: SceneData,
    reference_objects: List[ObjectInfo],
    target_object_name: str,
    mapping_data: dict,
) -> str:
    """
    用法: label = generate_referring_label(sample_record, scene, refs, target_name, mapping)
    作用: 生成带同类多实例消歧词的 placement 移动指令。
    输入: sample 记录、场景数据、参照物列表、目标物体展示名和类别映射表。
    输出: str，完整自然语言标注。
    """
    canonical_aabb = np.asarray(sample_record["canonical_aabb_object"], dtype=np.float64)
    canonical_corners = _get_bbox_corners(canonical_aabb)
    original_pose = np.asarray(sample_record["original_pose_world"], dtype=np.float64)
    placement_pose = np.asarray(sample_record["transform_world"], dtype=np.float64)
    original_corners = transform_points(canonical_corners, original_pose)
    placement_corners = transform_points(canonical_corners, placement_pose)

    target_obj_id = get_target_obj_id(sample_record)
    descriptor_map = build_object_descriptor_map(reference_objects, scene_data, mapping_data)
    target_description = (
        descriptor_map.get(target_obj_id, f"the {target_object_name}")
        if target_obj_id is not None
        else f"the {target_object_name}"
    )

    original_ref, rel_original = find_first_reference(
        original_corners,
        reference_objects,
        scene_data,
        exclude_id=target_obj_id,
    )
    placement_ref, rel_placement = resolve_placement_reference(
        placement_corners,
        reference_objects,
        scene_data,
        target_obj_id=target_obj_id,
    )

    ref_a_description = get_descriptor_for_object(original_ref, descriptor_map, mapping_data)
    ref_b_description = get_descriptor_for_object(placement_ref, descriptor_map, mapping_data)

    if ref_a_description == ref_b_description and rel_original == rel_placement and placement_ref is not None:
        filtered_refs = [obj for obj in reference_objects if obj.obj_id != placement_ref.obj_id]
        if filtered_refs:
            placement_ref, rel_placement = find_first_reference(
                placement_corners,
                filtered_refs,
                scene_data,
                exclude_id=target_obj_id,
            )
            ref_b_description = get_descriptor_for_object(placement_ref, descriptor_map, mapping_data)

    return LABEL_TEMPLATE.format(
        object_name=target_description,
        rel_original=rel_original,
        ref_a_name=ref_a_description,
        rel_placement=rel_placement,
        ref_b_name=ref_b_description,
    )


def build_referring_label_record(
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
    用法: record = build_referring_label_record(img, source, sample_id, sample, source_dir, adapter, cache, mapping)
    作用: 为单张图片生成与 auto_label.py 对齐的 JSON 记录。
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
    label = generate_referring_label(
        sample_record=sample_record,
        scene_data=scene,
        reference_objects=list(scene.objects),
        target_object_name=target_object_name,
        mapping_data=mapping_data,
    )
    return {
        "image_filename": img_file.name,
        "sample_id": sample_id,
        "source_name": source_name,
        "target_object_name": target_object_name,
        "is_found_target": is_found_target,
        "label": label,
    }


def auto_label_referring_from_images(
    image_dir: Path,
    outputs_base: Path,
    output_dir: Path,
    limit: int = None,
    overwrite: bool = False,
    sample_id_filter: Optional[Set[str]] = None,
    mapping_path: str = None,
) -> int:
    """
    用法: count = auto_label_referring_from_images(image_dir, outputs_base, output_dir)
    作用: 以图片文件为索引，为同类多实例困难样本生成可指代标注。
    输入: 图片目录、placement 输出根目录、输出目录、limit、overwrite、sample_id 过滤集合和映射路径。
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
    print("数据索引构建完成，开始执行同类多实例指代标注...")

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
            build_referring_label_record(
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
        print("只读 HTML 报告已生成，请使用 Web 服务查看。")

    print("\n标注完成")
    print(f"实际成功标注: {labeled} 个样本")
    return labeled


def main() -> None:
    """
    用法: python tools/auto_label_referring.py --image-dir outputs/Refering_difficult --output-dir outputs/auto_labels_referring_difficult
    作用: 解析命令行参数并执行同类多实例指代标注流程。
    输入: 无，参数来自命令行。
    输出: None，在终端打印处理结果。
    """
    args = build_parser().parse_args()
    sample_id_filter = build_sample_id_filter(args.sample_ids, args.sample_ids_file)
    auto_label_referring_from_images(
        image_dir=args.image_dir.resolve(),
        outputs_base=args.outputs_base.resolve(),
        output_dir=args.output_dir.resolve(),
        limit=args.limit,
        overwrite=args.overwrite,
        sample_id_filter=sample_id_filter,
        mapping_path=str(args.mapping.resolve()) if args.mapping else None,
    )
    print(f"图片来源目录: {args.image_dir.resolve()}")
    print(f"数据索引目录: {args.outputs_base.resolve()}")


if __name__ == "__main__":
    main()
