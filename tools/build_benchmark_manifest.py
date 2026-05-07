#!/usr/bin/env python3
"""
tools/build_benchmark_manifest.py
---------------------------------
从多模态 annotation、auto-label、placement 输出汇总生成自包含评测 benchmark 包。

用法:
    conda run -n spatial python tools/build_benchmark_manifest.py \
        --annotation-dir data/annotations/placement_multimodal \
        --label-json outputs/prompt_merged/all_labels.json \
        --outputs-base outputs \
        --output-dir benchmark/placement_v1 \
        --split test \
        --overwrite

作用:
    - 固化 size/collision/direction/object_center metric 所需全部标注字段
    - 将 occupancy grid 复制进 benchmark 包
    - 后续评测只依赖 benchmark manifest 和 predictions.json

输入:
    --annotation-dir: 多模态 annotation 目录，包含 split.json
    --label-json: 包含 spatial_relation 的 auto-label JSON
    --outputs-base: placement 输出根目录，包含 outputs/{source}/scene_objects、grid_meta、occupancy_grids
    --output-dir: benchmark 输出目录
    --split: annotation split 名称，默认 test

输出:
    output-dir/
        - manifest.json
        - summary.json
        - occupancy_grids/{source}/{scene_id}_{frame_id}.npy

使用示例:
    conda run -n spatial python tools/build_benchmark_manifest.py \
        --annotation-dir data/annotations/placement_multimodal \
        --label-json outputs/prompt_merged/all_labels.json \
        --output-dir benchmark/placement_v1 \
        --split test
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.bbox3d.bbox_utils import get_bbox_corners
from src.utils.coord_utils import transform_points


SCHEMA_VERSION = "placement_benchmark_manifest/v1"


def build_parser() -> argparse.ArgumentParser:
    """
    用法: parser = build_parser()
    作用: 构建命令行参数解析器
    输入: 无
    输出: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="构建 placement benchmark manifest")
    parser.add_argument("--annotation-dir", type=Path, required=True, help="多模态 annotation 目录")
    parser.add_argument("--label-json", type=Path, required=True, help="包含 spatial_relation 的 label JSON")
    parser.add_argument("--outputs-base", type=Path, default=Path("outputs"), help="placement 输出根目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="benchmark 输出目录")
    parser.add_argument("--split", type=str, default="test", help="annotation split 名称，默认 test")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 occupancy 复制文件")
    return parser


def resolve_project_path(path_value: str | Path) -> Path:
    """
    用法: path = resolve_project_path("outputs/demo.json")
    作用: 将相对路径解析到仓库根目录
    输入: path_value: str | Path
    输出: Path，绝对路径
    """
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def path_to_record(path_value: Path) -> str:
    """
    用法: text = path_to_record(Path("outputs/demo.json"))
    作用: 将路径转换为相对仓库根目录的稳定记录
    输入: path_value: Path
    输出: str，相对路径或绝对路径
    """
    resolved_path = path_value.resolve()
    try:
        return resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def load_json(json_path: Path) -> Any:
    """
    用法: payload = load_json(Path("demo.json"))
    作用: 读取 JSON 文件
    输入: json_path: Path
    输出: JSON 对象
    """
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(output_path: Path, payload: Any) -> None:
    """
    用法: save_json(Path("manifest.json"), payload)
    作用: 保存缩进 JSON
    输入: output_path: Path；payload: 可序列化对象
    输出: None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_label_lookup(label_records: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """
    用法: lookup = build_label_lookup(records)
    作用: 按 (source_name, sample_id) 建立 label 查找表
    输入: label_records: label 记录序列
    输出: dict[(source_name, sample_id), record]
    """
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for record in label_records:
        key = (str(record["source_name"]), str(record["sample_id"]))
        if key in lookup:
            raise ValueError(f"Duplicated label record: {key}")
        lookup[key] = dict(record)
    return lookup


def load_annotation_samples(annotation_dir: Path, split: str) -> list[dict[str, Any]]:
    """
    用法: samples = load_annotation_samples(Path("data/annotations/demo"), "test")
    作用: 读取指定 split 的 annotation 样本列表
    输入: annotation_dir: Path；split: str
    输出: list[dict]
    """
    split_path = annotation_dir / f"{split}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"annotation split not found: {split_path}")
    payload = load_json(split_path)
    return list(payload.get("samples", []))


def find_scene_object(scene_objects_payload: Mapping[str, Any], object_id: str, path: Path) -> dict[str, Any]:
    """
    用法: obj = find_scene_object(payload, "obj_1", path)
    作用: 在 scene_objects payload 中查找指定 object_id
    输入: scene_objects_payload: dict；object_id: str；path: Path，用于报错
    输出: dict，物体记录
    """
    for record in scene_objects_payload.get("objects", []):
        if str(record.get("object_id")) == str(object_id):
            return dict(record)
    raise KeyError(f"reference object {object_id} not found in {path}")


def get_reference_corners_world(object_record: Mapping[str, Any]) -> list[list[float]]:
    """
    用法: corners = get_reference_corners_world(object_record)
    作用: 获取或由 canonical AABB 和 pose_world 计算 reference 世界角点
    输入: object_record: scene_objects 中的物体记录
    输出: list[list[float]]，8 个世界坐标角点
    """
    corners = object_record.get("corners_world")
    if isinstance(corners, list) and len(corners) > 0:
        return np.asarray(corners, dtype=np.float64).tolist()
    canonical = np.asarray(object_record["canonical_aabb_object"], dtype=np.float64)
    pose = np.asarray(object_record["pose_world"], dtype=np.float64)
    return transform_points(get_bbox_corners(canonical), pose).tolist()


def copy_occupancy_grid(source_path: Path, output_dir: Path, source_name: str, prefix: str, overwrite: bool) -> str:
    """
    用法: rel = copy_occupancy_grid(src, output_dir, "hope", "scene_0000_0000", False)
    作用: 将 occupancy grid 复制进 benchmark 包
    输入: source_path: Path；output_dir: Path；source_name/prefix: str；overwrite: bool
    输出: str，相对 benchmark 根目录的 occupancy 路径
    """
    if not source_path.exists():
        raise FileNotFoundError(f"occupancy grid not found: {source_path}")
    target_path = output_dir / "occupancy_grids" / source_name / f"{prefix}.npy"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not target_path.exists():
        shutil.copy2(source_path, target_path)
    return target_path.relative_to(output_dir).as_posix()


def build_manifest_sample(
    sample: Mapping[str, Any],
    label_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
    outputs_base: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """
    用法: record = build_manifest_sample(sample, labels, outputs_base, output_dir, False)
    作用: 将单条 annotation 样本转换成 benchmark manifest 记录
    输入: sample/label_lookup/outputs_base/output_dir/overwrite
    输出: dict，单条 benchmark 样本
    """
    source_name = str(sample["source_name"])
    sample_id = str(sample["sample_id"])
    scene_id, frame_id, parsed_object_id = parse_sample_identity(sample)
    object_id = None if sample.get("object_id") is None else str(sample.get("object_id"))
    if object_id is None:
        object_id = parsed_object_id
    prefix = f"{scene_id}_{frame_id}"
    label_key = (source_name, sample_id)
    if label_key not in label_lookup:
        raise KeyError(f"missing label record for {label_key}")
    label_record = label_lookup[label_key]

    spatial_relation = label_record.get("spatial_relation") or sample.get("spatial_relation") or {}
    placement_relation = spatial_relation.get("placement") if isinstance(spatial_relation, dict) else None
    if not isinstance(placement_relation, dict) or not placement_relation.get("relation"):
        raise ValueError(f"missing placement spatial_relation for {label_key}")
    reference_id = placement_relation.get("reference_object_id")
    if reference_id is None:
        raise ValueError(f"missing placement reference_object_id for {label_key}")

    source_dir = outputs_base / source_name
    grid_meta_path = source_dir / "grid_meta" / f"{prefix}.json"
    scene_objects_path = source_dir / "scene_objects" / f"{prefix}.json"
    occupancy_source_path = source_dir / "occupancy_grids" / f"{prefix}.npy"
    if not grid_meta_path.exists():
        raise FileNotFoundError(f"grid_meta not found: {grid_meta_path}")
    if not scene_objects_path.exists():
        raise FileNotFoundError(f"scene_objects not found: {scene_objects_path}")

    grid_meta = load_json(grid_meta_path)
    scene_objects = load_json(scene_objects_path)
    reference_object = find_scene_object(scene_objects, str(reference_id), scene_objects_path)
    occupancy_rel_path = copy_occupancy_grid(
        occupancy_source_path,
        output_dir=output_dir,
        source_name=source_name,
        prefix=prefix,
        overwrite=overwrite,
    )
    occupancy_shape = tuple(int(value) for value in np.load(output_dir / occupancy_rel_path, mmap_mode="r").shape)
    grid_shape = tuple(int(value) for value in grid_meta.get("grid_shape", occupancy_shape))
    if grid_shape != occupancy_shape:
        raise ValueError(f"grid shape mismatch for {occupancy_source_path}: meta={grid_shape}, npy={occupancy_shape}")

    placement = sample.get("placement", {})
    if "target_box" not in placement:
        raise ValueError(f"missing placement.target_box for {label_key}")
    if "object_center" not in placement:
        raise ValueError(f"missing placement.object_center for {label_key}")
    camera = sample.get("camera") or grid_meta.get("camera")
    if not isinstance(camera, dict):
        raise ValueError(f"missing camera for {label_key}")

    return {
        "sample_id": sample_id,
        "source_name": source_name,
        "scene_id": scene_id,
        "frame_id": frame_id,
        "object_id": object_id,
        "class_name": None if sample.get("class_name") is None else str(sample.get("class_name")),
        "prompt": str(sample.get("prompt", label_record.get("label", ""))),
        "polished_prompt": str(sample.get("polished_prompt", label_record.get("polished_label", "")) or ""),
        "rgb_path": sample.get("rgb_path"),
        "point_cloud_path": sample.get("point_cloud_path"),
        "target_box_world": list(placement["target_box"]),
        "object_center_world": list(placement["object_center"]),
        "camera": camera,
        "occupancy": {
            "path": occupancy_rel_path,
            "voxel_params": dict(grid_meta["voxel_params"]),
            "grid_shape": list(grid_shape),
            "source_occupancy_grid_path": path_to_record(occupancy_source_path),
            "source_grid_meta_path": path_to_record(grid_meta_path),
        },
        "direction": {
            "expected_relation": str(placement_relation["relation"]),
            "reference_object_id": str(reference_id),
            "reference_class_name": placement_relation.get("reference_class_name"),
            "reference_name": placement_relation.get("reference_name"),
            "reference_corners_world": get_reference_corners_world(reference_object),
        },
    }


def parse_sample_identity(sample: Mapping[str, Any]) -> tuple[str, str, str | None]:
    """
    用法: scene_id, frame_id, object_id = parse_sample_identity(sample)
    作用: 从 annotation 字段或 sample_id 中解析 scene/frame/object 标识
    输入: sample: annotation 样本，至少包含 sample_id
    输出: tuple[str, str, str|None]，场景 ID、帧 ID、物体 ID
    """
    scene_id = sample.get("scene_id")
    frame_id = sample.get("frame_id")
    object_id = sample.get("object_id")
    if scene_id is not None and frame_id is not None:
        return str(scene_id), str(frame_id), None if object_id is None else str(object_id)

    sample_id = str(sample["sample_id"])
    if "_obj_" not in sample_id:
        raise ValueError(f"Cannot parse scene/frame from sample_id without '_obj_': {sample_id}")
    scene_frame_part, object_part = sample_id.split("_obj_", 1)
    if "_p" not in object_part:
        raise ValueError(f"Cannot parse object/placement rank from sample_id: {sample_id}")
    object_stem = object_part.rsplit("_p", 1)[0]
    if "_" not in scene_frame_part:
        raise ValueError(f"Cannot parse frame_id from sample_id: {sample_id}")
    parsed_scene_id, parsed_frame_id = scene_frame_part.rsplit("_", 1)
    parsed_object_id = f"obj_{object_stem}"
    return parsed_scene_id, parsed_frame_id, parsed_object_id if object_id is None else str(object_id)


def build_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """
    用法: summary = build_summary(samples)
    作用: 汇总 benchmark 样本数量和 source 分布
    输入: samples: list[dict]
    输出: dict，summary 信息
    """
    by_source: dict[str, int] = {}
    for sample in samples:
        source_name = str(sample["source_name"])
        by_source[source_name] = by_source.get(source_name, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_count": len(samples),
        "by_source": dict(sorted(by_source.items())),
    }


def build_benchmark_manifest(
    annotation_dir: Path,
    label_json: Path,
    outputs_base: Path,
    output_dir: Path,
    split: str,
    overwrite: bool,
) -> dict[str, Any]:
    """
    用法: payload = build_benchmark_manifest(annotation_dir, label_json, outputs_base, output_dir, "test", False)
    作用: 构建 benchmark manifest 并写出文件
    输入: annotation_dir/label_json/outputs_base/output_dir/split/overwrite
    输出: dict，manifest payload
    """
    annotation_dir = resolve_project_path(annotation_dir)
    label_json = resolve_project_path(label_json)
    outputs_base = resolve_project_path(outputs_base)
    output_dir = resolve_project_path(output_dir)
    annotation_samples = load_annotation_samples(annotation_dir, split)
    label_lookup = build_label_lookup(load_json(label_json))
    manifest_samples = [
        build_manifest_sample(
            sample=sample,
            label_lookup=label_lookup,
            outputs_base=outputs_base,
            output_dir=output_dir,
            overwrite=overwrite,
        )
        for sample in annotation_samples
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "split": str(split),
        "sample_count": len(manifest_samples),
        "inputs": {
            "annotation_dir": path_to_record(annotation_dir),
            "label_json": path_to_record(label_json),
            "outputs_base": path_to_record(outputs_base),
        },
        "samples": manifest_samples,
    }
    summary = build_summary(manifest_samples)
    save_json(output_dir / "manifest.json", payload)
    save_json(output_dir / "summary.json", summary)
    return payload


def main() -> None:
    """
    用法: main()
    作用: CLI 入口，构建 benchmark manifest
    输入: 无
    输出: None
    """
    parser = build_parser()
    args = parser.parse_args()
    payload = build_benchmark_manifest(
        annotation_dir=args.annotation_dir,
        label_json=args.label_json,
        outputs_base=args.outputs_base,
        output_dir=args.output_dir,
        split=args.split,
        overwrite=args.overwrite,
    )
    print(json.dumps({"schema_version": SCHEMA_VERSION, "sample_count": payload["sample_count"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
