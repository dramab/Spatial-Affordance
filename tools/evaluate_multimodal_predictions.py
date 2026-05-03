#!/usr/bin/env python3
"""
tools/evaluate_multimodal_predictions.py
----------------------------------------
评测多模态推理导出的 test 预测结果。

用法:
    conda run -n spatial python tools/evaluate_multimodal_predictions.py \
        --predictions outputs/infer_ptv3/predictions.json \
        --annotation-dir data/annotations/placement_multimodal_simple \
        --outputs-base outputs \
        --output-dir outputs/infer_ptv3_eval \
        --write-csv

作用:
    - 读取 infer_multimodal.py 导出的 predictions.json
    - 结合 annotation 与 placement samples 补齐 scene/frame/object 上游信息
    - 使用 occupancy grid 计算无碰撞，并计算方向一致、尺寸一致三类 metric
    - 导出 summary 与 per-sample 评测结果

输入:
    --predictions: 推理输出 predictions.json
    --annotation-dir: 多模态标注目录，包含 test.json 等 split 文件
    --outputs-base: placement 输出根目录，包含 outputs/{source}/samples、occupancy_grids、grid_meta
    --output-dir: 评测结果输出目录
    --mapping: auto_label 使用的类别名称映射文件，仅旧数据 fallback 重算方向关系时使用

输出:
    output-dir/
        - metrics_summary.json
        - per_sample_metrics.json
        - per_sample_metrics.csv，可选

使用示例:
    conda run -n spatial python tools/evaluate_multimodal_predictions.py \
        --predictions outputs/infer_ptv3/predictions.json \
        --annotation-dir data/annotations/placement_multimodal_simple \
        --output-dir outputs/infer_ptv3_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.free_bbox.datatypes import ObjectInfo, SceneData
from src.metrics.placement_eval import (
    DEFAULT_COLLISION_RATIO_THRESHOLD,
    DEFAULT_SIZE_MAX_REL_THRESHOLD,
    DEFAULT_SIZE_MEAN_REL_THRESHOLD,
    evaluate_collision,
    evaluate_direction,
    evaluate_size_consistency,
    merge_sample_metric_status,
    object_info_to_corners_world,
    summarize_by_source,
    summarize_metric_records,
)
from tools.build_multimodal_dataset import build_adapter, build_source_configs


SCHEMA_VERSION = "placement_prediction_metrics/v1"


def build_parser() -> argparse.ArgumentParser:
    """
    用法: parser = build_parser()
    作用: 构建命令行参数解析器
    输入: 无
    输出: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="评测多模态 placement 预测结果")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="infer_multimodal.py 导出的 predictions.json",
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        required=True,
        help="多模态 annotation 目录，包含 split.json",
    )
    parser.add_argument(
        "--outputs-base",
        type=Path,
        default=Path("outputs"),
        help="placement 输出根目录，默认 outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/multimodal_prediction_eval"),
        help="评测结果输出目录",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("configs/annotation/mappingv2.json"),
        help="类别名称映射 JSON，仅旧数据 fallback 重算方向关系时使用",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="annotation split 名称；默认使用 predictions.json 中的 split",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        help="可选，仅评测指定 sample_id",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="可选，仅评测筛选后的前 N 个样本，用于快速 smoke test",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="每处理多少个样本打印一次进度，默认 50",
    )
    parser.add_argument(
        "--collision-ratio-threshold",
        type=float,
        default=DEFAULT_COLLISION_RATIO_THRESHOLD,
        help="最大 OCCUPIED 体素占预测体素比例阈值，默认 0.01",
    )
    parser.add_argument(
        "--size-mean-rel-threshold",
        type=float,
        default=DEFAULT_SIZE_MEAN_REL_THRESHOLD,
        help="尺寸三轴平均相对误差阈值，默认 0.10",
    )
    parser.add_argument(
        "--size-max-rel-threshold",
        type=float,
        default=DEFAULT_SIZE_MAX_REL_THRESHOLD,
        help="尺寸单轴最大相对误差阈值，默认 0.15",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="额外导出 per_sample_metrics.csv",
    )
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
    作用: 将路径转换为 JSON 中稳定记录的相对路径
    输入: path_value: Path
    输出: str，相对仓库路径或绝对路径
    """
    resolved_path = path_value.resolve()
    try:
        return resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def load_json(json_path: Path) -> Any:
    """
    用法: payload = load_json(Path("outputs/predictions.json"))
    作用: 读取 JSON 文件
    输入: json_path: Path
    输出: JSON 解析后的对象
    """
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(output_path: Path, payload: Any) -> None:
    """
    用法: save_json(Path("outputs/metrics.json"), payload)
    作用: 将对象保存为缩进 JSON
    输入: output_path: Path；payload: 可序列化对象
    输出: None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_annotation_lookup(annotation_dir: Path, split: str) -> dict[tuple[str, str], dict[str, Any]]:
    """
    用法: lookup = load_annotation_lookup(Path("data/annotations/demo"), "test")
    作用: 读取 split annotation 并按 (source_name, sample_id) 建索引
    输入: annotation_dir: 标注目录；split: split 名称
    输出: dict[(source_name, sample_id), sample_record]
    """
    split_path = annotation_dir / f"{split}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"annotation split not found: {split_path}")
    payload = load_json(split_path)
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for sample in payload.get("samples", []):
        key = (str(sample["source_name"]), str(sample["sample_id"]))
        lookup[key] = sample
    return lookup


def load_source_sample_lookup(
    outputs_base: Path,
    source_names: Iterable[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    用法: lookup = load_source_sample_lookup(Path("outputs"), ["hope"])
    作用: 读取 outputs/{source}/samples 下的原始 placement sample 记录
    输入: outputs_base: placement 输出根目录；source_names: 数据源名称列表
    输出: dict[(source_name, sample_id), placement_sample_record]
    """
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for source_name in sorted(set(source_names)):
        samples_dir = outputs_base / source_name / "samples"
        if not samples_dir.exists():
            continue
        for sample_json in sorted(samples_dir.glob("*.json")):
            payload = load_json(sample_json)
            for record in payload.get("samples", []):
                key = (source_name, str(record["sample_id"]))
                lookup[key] = record
    return lookup


def load_occupancy_frame(
    outputs_base: Path,
    source_name: str,
    scene_id: str,
    frame_id: str,
) -> dict[str, Any]:
    """
    用法: frame = load_occupancy_frame(Path("outputs"), "ycbv_test", "000059", "001140")
    作用: 读取指定帧的 occupancy grid 和 grid_meta
    输入: outputs_base: placement 输出根目录；source_name/scene_id/frame_id: 样本定位字段
    输出: dict，包含 grid、voxel_params、occupancy_grid_path、grid_meta_path
    """
    prefix = f"{scene_id}_{frame_id}"
    occupancy_grid_path = outputs_base / source_name / "occupancy_grids" / f"{prefix}.npy"
    grid_meta_path = outputs_base / source_name / "grid_meta" / f"{prefix}.json"
    if not occupancy_grid_path.exists():
        raise FileNotFoundError(f"occupancy grid not found: {occupancy_grid_path}")
    if not grid_meta_path.exists():
        raise FileNotFoundError(f"grid meta not found: {grid_meta_path}")
    meta = load_json(grid_meta_path)
    voxel_params = dict(meta.get("voxel_params") or {})
    if "origin" not in voxel_params or "voxel_size" not in voxel_params:
        raise ValueError(f"grid meta missing voxel_params origin/voxel_size: {grid_meta_path}")
    grid = np.load(occupancy_grid_path)
    meta_grid_shape = meta.get("grid_shape")
    if meta_grid_shape is not None:
        expected_shape = tuple(int(value) for value in meta_grid_shape)
        if expected_shape != tuple(grid.shape):
            raise ValueError(
                f"grid shape mismatch for {occupancy_grid_path}: "
                f"meta={expected_shape}, npy={tuple(grid.shape)}"
            )
    return {
        "grid": grid,
        "voxel_params": voxel_params,
        "occupancy_grid_path": occupancy_grid_path,
        "grid_meta_path": grid_meta_path,
    }


def load_occupancy_frame_cached(
    occupancy_cache: dict[tuple[str, str, str], dict[str, Any]],
    outputs_base: Path,
    source_name: str,
    scene_id: str,
    frame_id: str,
) -> dict[str, Any]:
    """
    用法: frame = load_occupancy_frame_cached(cache, outputs_base, "hope", "scene_0000", "0000")
    作用: 加载并缓存同一 source/scene/frame 的 occupancy grid
    输入: occupancy_cache: 缓存字典；outputs_base/source_name/scene_id/frame_id: 样本定位字段
    输出: dict，包含 grid 与 voxel_params
    """
    key = (str(source_name), str(scene_id), str(frame_id))
    if key not in occupancy_cache:
        occupancy_cache[key] = load_occupancy_frame(
            outputs_base=outputs_base,
            source_name=source_name,
            scene_id=scene_id,
            frame_id=frame_id,
        )
    return occupancy_cache[key]


def select_predictions(
    predictions: list[dict[str, Any]],
    sample_ids: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    """
    用法: selected = select_predictions(predictions, ["sample_a"], 10)
    作用: 按 sample_id 和 limit 筛选待评测预测记录
    输入: predictions: 预测记录列表；sample_ids: 可选 sample_id 列表；limit: 可选数量上限
    输出: list[dict]，筛选后的预测记录
    """
    selected = predictions
    if sample_ids:
        requested = [str(sample_id) for sample_id in sample_ids]
        index = {str(item["sample_id"]): item for item in predictions}
        missing = [sample_id for sample_id in requested if sample_id not in index]
        if missing:
            raise KeyError(f"sample_ids not found in predictions: {missing}")
        selected = [index[sample_id] for sample_id in requested]
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("--limit must be positive when provided")
        selected = selected[:int(limit)]
    return selected


def load_mapping(mapping_path: Path) -> dict[str, str]:
    """
    用法: mapping = load_mapping(Path("configs/annotation/mappingv2.json"))
    作用: 读取类别展示名映射，兼容 {"mapping": {...}} 与直接 dict 两种格式
    输入: mapping_path: Path
    输出: dict，类别名到展示名的映射
    """
    resolved_path = resolve_project_path(mapping_path)
    if not resolved_path.exists():
        fallback_path = PROJECT_ROOT / "configs/annotation/mapping.json"
        if not fallback_path.exists():
            return {}
        resolved_path = fallback_path
    payload = load_json(resolved_path)
    if isinstance(payload, dict) and isinstance(payload.get("mapping"), dict):
        payload = payload["mapping"]
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items()}


def build_scene_adapters(source_names: Iterable[str]) -> tuple[dict[str, dict], dict[str, Any]]:
    """
    用法: configs, adapters = build_scene_adapters(["hope", "dopose"])
    作用: 根据 source_name 构建数据集配置和 adapter
    输入: source_names: 数据源名称序列
    输出: tuple，分别为 source 配置和 adapter 映射
    """
    source_configs = build_source_configs(source_names)
    adapters = {
        source_name: build_adapter(cfg)
        for source_name, cfg in source_configs.items()
    }
    return source_configs, adapters


def load_scene_cached(
    scene_cache: dict[tuple[str, str, str], SceneData],
    source_configs: Mapping[str, dict],
    adapters: Mapping[str, Any],
    source_name: str,
    scene_id: str,
    frame_id: str,
) -> SceneData:
    """
    用法: scene = load_scene_cached(cache, configs, adapters, "hope", "scene_0000", "0000")
    作用: 加载并缓存同一 source/scene/frame 的 SceneData
    输入: scene_cache/source_configs/adapters；source_name/scene_id/frame_id
    输出: SceneData
    """
    key = (str(source_name), str(scene_id), str(frame_id))
    if key not in scene_cache:
        dataset_cfg = source_configs[source_name].get("dataset", {})
        scene_path = Path(str(dataset_cfg["root_dir"])) / str(scene_id)
        scene_cache[key] = adapters[source_name].load_scene(str(scene_path), str(frame_id))
    return scene_cache[key]


def find_object_by_id(scene_objects: Iterable[ObjectInfo], object_id: str | None) -> ObjectInfo | None:
    """
    用法: obj = find_object_by_id(scene.objects, "obj_1")
    作用: 在当前帧物体列表中按 obj_id 查找物体
    输入: scene_objects: ObjectInfo 序列；object_id: 目标 ID
    输出: ObjectInfo 或 None
    """
    if object_id is None:
        return None
    target_id = str(object_id)
    for obj in scene_objects:
        if str(obj.obj_id) == target_id:
            return obj
    return None


def get_target_box(prediction: Mapping[str, Any], annotation: Mapping[str, Any] | None) -> list[float] | None:
    """
    用法: target_box = get_target_box(prediction, annotation)
    作用: 从 prediction 或 annotation 中获取 GT 目标 box
    输入: prediction: 单条预测记录；annotation: 可选 annotation 记录
    输出: list[float] 或 None
    """
    if "gt_box_world" in prediction:
        return list(prediction["gt_box_world"])
    if annotation is not None:
        placement = annotation.get("placement", {})
        if "target_box" in placement:
            return list(placement["target_box"])
    return None


def get_object_id(annotation: Mapping[str, Any] | None, sample_record: Mapping[str, Any] | None) -> str | None:
    """
    用法: object_id = get_object_id(annotation, sample_record)
    作用: 从新 annotation 或旧 placement sample 中获取目标物体 ID
    输入: annotation/sample_record: 可选样本记录
    输出: str 或 None
    """
    if annotation is not None and annotation.get("object_id"):
        return str(annotation["object_id"])
    if sample_record is not None and sample_record.get("object_id"):
        return str(sample_record["object_id"])
    return None


def get_scene_frame(
    annotation: Mapping[str, Any] | None,
    sample_record: Mapping[str, Any] | None,
) -> tuple[str | None, str | None]:
    """
    用法: scene_id, frame_id = get_scene_frame(annotation, sample_record)
    作用: 从新 annotation 或旧 placement sample 中获取 scene/frame
    输入: annotation/sample_record: 可选样本记录
    输出: tuple[str|None, str|None]
    """
    scene_id = None
    frame_id = None
    if annotation is not None:
        scene_id = annotation.get("scene_id")
        frame_id = annotation.get("frame_id")
    if (not scene_id or not frame_id) and sample_record is not None:
        scene_id = sample_record.get("scene_id")
        frame_id = sample_record.get("frame_id")
    return (None if scene_id is None else str(scene_id), None if frame_id is None else str(frame_id))


def get_structured_placement_relation(
    annotation: Mapping[str, Any] | None,
    sample_record: Mapping[str, Any] | None,
    scene: SceneData | None,
    mapping_data: Mapping[str, str],
) -> dict[str, Any] | None:
    """
    用法: relation = get_structured_placement_relation(annotation, sample_record, scene, mapping)
    作用: 获取指令目标位置的结构化 relation，优先读取新字段，旧数据则临时重算
    输入: annotation/sample_record/scene/mapping_data
    输出: dict 或 None，包含 relation 和 reference_object_id
    """
    if annotation is not None:
        spatial_relation = annotation.get("spatial_relation") or {}
        placement_relation = spatial_relation.get("placement") if isinstance(spatial_relation, dict) else None
        if isinstance(placement_relation, dict) and placement_relation.get("relation"):
            return dict(placement_relation)

    if sample_record is None or scene is None:
        return None

    from tools import auto_label

    target_object_name, _ = auto_label.get_target_object_name(
        dict(sample_record),
        Path("."),
        dict(mapping_data),
    )
    _, spatial_relation = auto_label.generate_label_with_spatial_relation(
        sample_record=dict(sample_record),
        scene_data=scene,
        reference_objects=list(scene.objects),
        target_object_name=target_object_name,
        mapping_data=dict(mapping_data),
    )
    placement_relation = spatial_relation.get("placement")
    if isinstance(placement_relation, dict) and placement_relation.get("relation"):
        return dict(placement_relation)
    return None


def make_skip_result(reason: str) -> dict[str, Any]:
    """
    用法: result = make_skip_result("missing scene")
    作用: 构造未评估 metric 的统一结果
    输入: reason: str，跳过原因
    输出: dict
    """
    return {
        "evaluated": False,
        "reason": str(reason),
    }


def evaluate_one_prediction(
    prediction: Mapping[str, Any],
    annotation: Mapping[str, Any] | None,
    sample_record: Mapping[str, Any] | None,
    scene: SceneData | None,
    occupancy_frame: Mapping[str, Any] | None,
    mapping_data: Mapping[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    用法: record = evaluate_one_prediction(pred, ann, sample_record, scene, occupancy, mapping, args)
    作用: 对单条预测计算 collision、direction、size 三类 metric
    输入: prediction/annotation/sample_record/scene/occupancy_frame/mapping_data/args
    输出: dict，单样本评测结果
    """
    sample_id = str(prediction["sample_id"])
    source_name = str(prediction["source_name"])
    pred_box = prediction["pred_box_world"]
    target_box = get_target_box(prediction, annotation)
    object_id = get_object_id(annotation, sample_record)
    scene_id, frame_id = get_scene_frame(annotation, sample_record)

    errors: list[str] = []
    if target_box is None:
        size_result = make_skip_result("missing gt target_box")
    else:
        try:
            size_result = evaluate_size_consistency(
                pred_box_world=pred_box,
                target_box_world=target_box,
                mean_relative_threshold=args.size_mean_rel_threshold,
                max_axis_relative_threshold=args.size_max_rel_threshold,
            )
        except Exception as exc:
            size_result = make_skip_result(f"size metric failed: {exc}")
            errors.append(str(exc))

    if occupancy_frame is None:
        collision_result = make_skip_result("missing occupancy grid")
    else:
        try:
            collision_result = evaluate_collision(
                pred_box_world=pred_box,
                occupancy_grid=occupancy_frame["grid"],
                voxel_params=occupancy_frame["voxel_params"],
                collision_ratio_threshold=args.collision_ratio_threshold,
            )
            collision_result["occupancy_grid_path"] = path_to_record(
                occupancy_frame["occupancy_grid_path"]
            )
            collision_result["grid_meta_path"] = path_to_record(
                occupancy_frame["grid_meta_path"]
            )
        except Exception as exc:
            collision_result = make_skip_result(f"collision metric failed: {exc}")
            errors.append(str(exc))

    placement_relation = get_structured_placement_relation(
        annotation=annotation,
        sample_record=sample_record,
        scene=scene,
        mapping_data=mapping_data,
    )
    if scene is None:
        direction_result = make_skip_result("missing scene")
    elif placement_relation is None:
        direction_result = make_skip_result("missing structured placement relation")
    else:
        reference_id = placement_relation.get("reference_object_id")
        reference_obj = find_object_by_id(scene.objects, None if reference_id is None else str(reference_id))
        if reference_obj is None:
            direction_result = make_skip_result(f"missing reference object: {reference_id}")
        else:
            try:
                direction_result = evaluate_direction(
                    pred_box_world=pred_box,
                    reference_corners_world=object_info_to_corners_world(reference_obj),
                    camera=annotation.get("camera", {}) if annotation is not None else {
                        "fx": scene.camera.fx,
                        "fy": scene.camera.fy,
                        "cx": scene.camera.cx,
                        "cy": scene.camera.cy,
                        "E_c2w": scene.camera.E_c2w.tolist(),
                    },
                    expected_relation=str(placement_relation["relation"]),
                )
                direction_result["reference_object_id"] = str(reference_id)
                direction_result["reference_class_name"] = placement_relation.get("reference_class_name")
                direction_result["reference_name"] = placement_relation.get("reference_name")
            except Exception as exc:
                direction_result = make_skip_result(f"direction metric failed: {exc}")
                errors.append(str(exc))

    status = merge_sample_metric_status(collision_result, direction_result, size_result)
    return {
        "sample_id": sample_id,
        "source_name": source_name,
        "scene_id": scene_id,
        "frame_id": frame_id,
        "object_id": object_id,
        "collision": collision_result,
        "direction": direction_result,
        "size": size_result,
        "status": status,
        "errors": errors,
    }


def build_direction_confusion(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """
    用法: confusion = build_direction_confusion(records)
    作用: 汇总方向 expected_relation -> pred_relation 混淆矩阵
    输入: records: 单样本评测记录
    输出: dict[expected][pred] = count
    """
    confusion: dict[str, dict[str, int]] = {}
    for record in records:
        direction = record.get("direction", {})
        if not direction.get("evaluated"):
            continue
        expected = str(direction.get("expected_relation"))
        predicted = str(direction.get("pred_relation"))
        confusion.setdefault(expected, {})
        confusion[expected][predicted] = confusion[expected].get(predicted, 0) + 1
    return {
        expected: dict(sorted(pred_counts.items()))
        for expected, pred_counts in sorted(confusion.items())
    }


def write_per_sample_csv(output_path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """
    用法: write_per_sample_csv(Path("metrics.csv"), records)
    作用: 将 per-sample metric 摘要写成 CSV
    输入: output_path: Path；records: 单样本评测记录
    输出: None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "source_name",
        "scene_id",
        "frame_id",
        "object_id",
        "placement_success",
        "collision_evaluated",
        "collision_free",
        "pred_voxel_count",
        "occupied_voxel_count",
        "occupied_collision_ratio",
        "unknown_voxel_count",
        "unknown_overlap_ratio",
        "occupancy_grid_path",
        "grid_meta_path",
        "direction_evaluated",
        "direction_correct",
        "expected_relation",
        "pred_relation",
        "reference_object_id",
        "size_evaluated",
        "size_consistent",
        "mean_relative_size_error",
        "max_axis_relative_size_error",
        "size_l2_cm",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            collision = record.get("collision", {})
            direction = record.get("direction", {})
            size = record.get("size", {})
            status = record.get("status", {})
            writer.writerow({
                "sample_id": record.get("sample_id"),
                "source_name": record.get("source_name"),
                "scene_id": record.get("scene_id"),
                "frame_id": record.get("frame_id"),
                "object_id": record.get("object_id"),
                "placement_success": status.get("placement_success"),
                "collision_evaluated": collision.get("evaluated"),
                "collision_free": collision.get("collision_free"),
                "pred_voxel_count": collision.get("pred_voxel_count"),
                "occupied_voxel_count": collision.get("occupied_voxel_count"),
                "occupied_collision_ratio": collision.get("occupied_collision_ratio"),
                "unknown_voxel_count": collision.get("unknown_voxel_count"),
                "unknown_overlap_ratio": collision.get("unknown_overlap_ratio"),
                "occupancy_grid_path": collision.get("occupancy_grid_path"),
                "grid_meta_path": collision.get("grid_meta_path"),
                "direction_evaluated": direction.get("evaluated"),
                "direction_correct": direction.get("direction_correct"),
                "expected_relation": direction.get("expected_relation"),
                "pred_relation": direction.get("pred_relation"),
                "reference_object_id": direction.get("reference_object_id"),
                "size_evaluated": size.get("evaluated"),
                "size_consistent": size.get("size_consistent"),
                "mean_relative_size_error": size.get("mean_relative_size_error"),
                "max_axis_relative_size_error": size.get("max_axis_relative_size_error"),
                "size_l2_cm": size.get("size_l2_cm"),
            })


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """
    用法: payload = run_evaluation(args)
    作用: 执行预测结果评测主流程
    输入: args: argparse.Namespace
    输出: dict，summary payload
    """
    predictions_path = resolve_project_path(args.predictions)
    annotation_dir = resolve_project_path(args.annotation_dir)
    outputs_base = resolve_project_path(args.outputs_base)
    output_dir = resolve_project_path(args.output_dir)

    predictions_payload = load_json(predictions_path)
    predictions = select_predictions(
        predictions=list(predictions_payload.get("predictions", [])),
        sample_ids=args.sample_ids,
        limit=args.limit,
    )
    split = str(args.split or predictions_payload.get("split", "test"))
    annotation_lookup = load_annotation_lookup(annotation_dir, split)
    source_names = sorted({str(item["source_name"]) for item in predictions})
    sample_lookup = load_source_sample_lookup(outputs_base, source_names)
    mapping_data = load_mapping(args.mapping)
    source_configs, adapters = build_scene_adapters(source_names)
    scene_cache: dict[tuple[str, str, str], SceneData] = {}
    occupancy_cache: dict[tuple[str, str, str], dict[str, Any]] = {}

    per_sample_records: list[dict[str, Any]] = []
    total_predictions = len(predictions)
    progress_interval = max(0, int(args.progress_interval))
    for idx, prediction in enumerate(predictions, start=1):
        key = (str(prediction["source_name"]), str(prediction["sample_id"]))
        annotation = annotation_lookup.get(key)
        sample_record = sample_lookup.get(key)
        scene_id, frame_id = get_scene_frame(annotation, sample_record)
        scene = None
        occupancy_frame = None
        if scene_id is not None and frame_id is not None:
            try:
                scene = load_scene_cached(
                    scene_cache=scene_cache,
                    source_configs=source_configs,
                    adapters=adapters,
                    source_name=key[0],
                    scene_id=scene_id,
                    frame_id=frame_id,
                )
            except Exception as exc:
                scene = None
                print(f"跳过 scene 加载失败样本 {key}: {exc}")
            try:
                occupancy_frame = load_occupancy_frame_cached(
                    occupancy_cache=occupancy_cache,
                    outputs_base=outputs_base,
                    source_name=key[0],
                    scene_id=scene_id,
                    frame_id=frame_id,
                )
            except Exception as exc:
                occupancy_frame = None
                print(f"跳过 occupancy 加载失败样本 {key}: {exc}")

        per_sample_records.append(
            evaluate_one_prediction(
                prediction=prediction,
                annotation=annotation,
                sample_record=sample_record,
                scene=scene,
                occupancy_frame=occupancy_frame,
                mapping_data=mapping_data,
                args=args,
            )
        )
        if progress_interval > 0 and (idx == total_predictions or idx % progress_interval == 0):
            print(f"已评测 {idx}/{total_predictions} 个样本")

    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "predictions": path_to_record(predictions_path),
            "annotation_dir": path_to_record(annotation_dir),
            "outputs_base": path_to_record(outputs_base),
            "split": split,
        },
        "thresholds": {
            "collision_ratio_threshold": float(args.collision_ratio_threshold),
            "size_mean_relative_threshold": float(args.size_mean_rel_threshold),
            "size_max_axis_relative_threshold": float(args.size_max_rel_threshold),
        },
        "sample_count": len(per_sample_records),
        "summary": summarize_metric_records(per_sample_records),
        "by_source": summarize_by_source(per_sample_records),
        "direction_confusion": build_direction_confusion(per_sample_records),
        "outputs": {
            "per_sample_metrics": path_to_record(output_dir / "per_sample_metrics.json"),
            "metrics_summary": path_to_record(output_dir / "metrics_summary.json"),
        },
    }
    if args.write_csv:
        summary_payload["outputs"]["per_sample_metrics_csv"] = path_to_record(output_dir / "per_sample_metrics.csv")

    save_json(output_dir / "per_sample_metrics.json", {
        "schema_version": SCHEMA_VERSION,
        "sample_count": len(per_sample_records),
        "samples": per_sample_records,
    })
    save_json(output_dir / "metrics_summary.json", summary_payload)
    if args.write_csv:
        write_per_sample_csv(output_dir / "per_sample_metrics.csv", per_sample_records)

    return summary_payload


def main() -> None:
    """
    用法: main()
    作用: CLI 入口，执行评测并打印摘要
    输入: 无，参数来自命令行
    输出: None
    """
    args = build_parser().parse_args()
    payload = run_evaluation(args)
    summary = payload["summary"]
    output_dir = resolve_project_path(args.output_dir)
    print("评测完成")
    print(f"输出目录: {output_dir}")
    print(f"样本数量: {payload['sample_count']}")
    print(f"完整 metric 覆盖率: {summary['full_metric_coverage']}")
    print(f"placement_success_rate: {summary['placement_success_rate']}")
    print(f"collision_free_rate: {summary['collision_free_rate']}")
    print(f"direction_correct_rate: {summary['direction_correct_rate']}")
    print(f"size_consistent_rate: {summary['size_consistent_rate']}")


if __name__ == "__main__":
    main()
