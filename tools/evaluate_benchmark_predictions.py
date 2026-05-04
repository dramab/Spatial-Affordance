#!/usr/bin/env python3
"""
tools/evaluate_benchmark_predictions.py
---------------------------------------
使用统一 benchmark 包评测 placement predictions。

用法:
    conda run -n spatial python tools/evaluate_benchmark_predictions.py \
        --benchmark-dir benchmark/placement_v1 \
        --predictions outputs/infer_ptv3/predictions.json \
        --output-dir outputs/infer_ptv3_benchmark_eval \
        --write-csv

作用:
    - 只读取 benchmark manifest、benchmark 内 occupancy grid 和 predictions.json
    - 不访问 annotation-dir、outputs-base、mapping 或原始数据集 adapter
    - 计算 collision、direction、size 三类 metric 并导出 summary

输入:
    --benchmark-dir: build_benchmark_manifest.py 生成的 benchmark 目录
    --predictions: infer_multimodal.py 导出的 predictions.json
    --output-dir: 评测结果输出目录

输出:
    output-dir/
        - metrics_summary.json
        - per_sample_metrics.json
        - per_sample_metrics.csv，可选

使用示例:
    conda run -n spatial python tools/evaluate_benchmark_predictions.py \
        --benchmark-dir benchmark/placement_v1 \
        --predictions outputs/infer_ptv3/predictions.json \
        --output-dir outputs/infer_ptv3_benchmark_eval
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

from src.metrics.placement_eval import (
    DEFAULT_COLLISION_RATIO_THRESHOLD,
    DEFAULT_SIZE_MAX_REL_THRESHOLD,
    DEFAULT_SIZE_MEAN_REL_THRESHOLD,
    evaluate_collision,
    evaluate_direction,
    evaluate_size_consistency,
    merge_sample_metric_status,
    summarize_by_source,
    summarize_metric_records,
)


SCHEMA_VERSION = "placement_benchmark_metrics/v1"


def build_parser() -> argparse.ArgumentParser:
    """
    用法: parser = build_parser()
    作用: 构建命令行参数解析器
    输入: 无
    输出: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="使用 benchmark 包评测 placement predictions")
    parser.add_argument("--benchmark-dir", type=Path, required=True, help="benchmark 目录")
    parser.add_argument("--predictions", type=Path, required=True, help="predictions.json 路径")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark_prediction_eval"), help="输出目录")
    parser.add_argument("--sample-ids", nargs="+", default=None, help="可选，仅评测指定 sample_id")
    parser.add_argument("--limit", type=int, default=None, help="可选，仅评测前 N 个样本")
    parser.add_argument("--progress-interval", type=int, default=50, help="进度打印间隔，默认 50")
    parser.add_argument("--collision-ratio-threshold", type=float, default=DEFAULT_COLLISION_RATIO_THRESHOLD)
    parser.add_argument("--size-mean-rel-threshold", type=float, default=DEFAULT_SIZE_MEAN_REL_THRESHOLD)
    parser.add_argument("--size-max-rel-threshold", type=float, default=DEFAULT_SIZE_MAX_REL_THRESHOLD)
    parser.add_argument("--write-csv", action="store_true", help="额外导出 per_sample_metrics.csv")
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
    用法: payload = load_json(Path("predictions.json"))
    作用: 读取 JSON 文件
    输入: json_path: Path
    输出: JSON 对象
    """
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(output_path: Path, payload: Any) -> None:
    """
    用法: save_json(Path("metrics.json"), payload)
    作用: 保存缩进 JSON
    输入: output_path: Path；payload: 可序列化对象
    输出: None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_manifest_lookup(benchmark_dir: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """
    用法: lookup, payload = load_manifest_lookup(Path("benchmark/placement_v1"))
    作用: 读取 manifest 并按 (source_name, sample_id) 建索引
    输入: benchmark_dir: Path
    输出: tuple(lookup, manifest_payload)
    """
    manifest_path = benchmark_dir / "manifest.json"
    payload = load_json(manifest_path)
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for sample in payload.get("samples", []):
        key = (str(sample["source_name"]), str(sample["sample_id"]))
        if key in lookup:
            raise ValueError(f"Duplicated benchmark sample: {key}")
        lookup[key] = sample
    return lookup, payload


def select_predictions(
    predictions: list[dict[str, Any]],
    sample_ids: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    """
    用法: selected = select_predictions(predictions, ["sample_a"], 10)
    作用: 按 sample_id 和 limit 筛选预测记录
    输入: predictions: list；sample_ids: 可选；limit: 可选
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


def make_skip_result(reason: str) -> dict[str, Any]:
    """
    用法: result = make_skip_result("missing field")
    作用: 构造未评估 metric 的统一结果
    输入: reason: str
    输出: dict
    """
    return {"evaluated": False, "reason": str(reason)}


def load_occupancy_cached(
    cache: dict[str, np.ndarray],
    benchmark_dir: Path,
    occupancy_path: str,
) -> np.ndarray:
    """
    用法: grid = load_occupancy_cached(cache, benchmark_dir, "occupancy_grids/hope/a.npy")
    作用: 加载并缓存 benchmark 内 occupancy grid
    输入: cache: dict；benchmark_dir: Path；occupancy_path: str
    输出: ndarray，occupancy grid
    """
    if occupancy_path not in cache:
        grid_path = benchmark_dir / occupancy_path
        if not grid_path.exists():
            raise FileNotFoundError(f"benchmark occupancy grid not found: {grid_path}")
        cache[occupancy_path] = np.load(grid_path)
    return cache[occupancy_path]


def evaluate_one_prediction(
    prediction: Mapping[str, Any],
    benchmark_sample: Mapping[str, Any],
    occupancy_grid: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    用法: record = evaluate_one_prediction(prediction, sample, grid, args)
    作用: 对单条预测计算 benchmark metric
    输入: prediction: 预测记录；benchmark_sample: manifest 样本；occupancy_grid: ndarray；args: CLI 参数
    输出: dict，单样本评测结果
    """
    sample_id = str(prediction["sample_id"])
    source_name = str(prediction["source_name"])
    pred_box = prediction["pred_box_world"]
    errors: list[str] = []

    try:
        size_result = evaluate_size_consistency(
            pred_box_world=pred_box,
            target_box_world=benchmark_sample["target_box_world"],
            mean_relative_threshold=args.size_mean_rel_threshold,
            max_axis_relative_threshold=args.size_max_rel_threshold,
        )
    except Exception as exc:
        size_result = make_skip_result(f"size metric failed: {exc}")
        errors.append(str(exc))

    occupancy = benchmark_sample["occupancy"]
    try:
        collision_result = evaluate_collision(
            pred_box_world=pred_box,
            occupancy_grid=occupancy_grid,
            voxel_params=occupancy["voxel_params"],
            collision_ratio_threshold=args.collision_ratio_threshold,
        )
        collision_result["occupancy_grid_path"] = occupancy["path"]
    except Exception as exc:
        collision_result = make_skip_result(f"collision metric failed: {exc}")
        errors.append(str(exc))

    direction = benchmark_sample.get("direction", {})
    try:
        direction_result = evaluate_direction(
            pred_box_world=pred_box,
            reference_corners_world=np.asarray(direction["reference_corners_world"], dtype=np.float64),
            camera=benchmark_sample["camera"],
            expected_relation=str(direction["expected_relation"]),
        )
        direction_result["reference_object_id"] = str(direction["reference_object_id"])
        direction_result["reference_class_name"] = direction.get("reference_class_name")
        direction_result["reference_name"] = direction.get("reference_name")
    except Exception as exc:
        direction_result = make_skip_result(f"direction metric failed: {exc}")
        errors.append(str(exc))

    status = merge_sample_metric_status(collision_result, direction_result, size_result)
    return {
        "sample_id": sample_id,
        "source_name": source_name,
        "scene_id": benchmark_sample.get("scene_id"),
        "frame_id": benchmark_sample.get("frame_id"),
        "object_id": benchmark_sample.get("object_id"),
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
        "sample_id", "source_name", "scene_id", "frame_id", "object_id",
        "placement_success", "collision_evaluated", "collision_free",
        "occupied_collision_ratio", "direction_evaluated", "direction_correct",
        "expected_relation", "pred_relation", "reference_object_id",
        "size_evaluated", "size_consistent", "mean_relative_size_error",
        "max_axis_relative_size_error", "size_l2_cm",
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
                "occupied_collision_ratio": collision.get("occupied_collision_ratio"),
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
    用法: summary = run_evaluation(args)
    作用: 执行 benchmark 预测评测主流程
    输入: args: argparse.Namespace
    输出: dict，summary payload
    """
    benchmark_dir = resolve_project_path(args.benchmark_dir)
    predictions_path = resolve_project_path(args.predictions)
    output_dir = resolve_project_path(args.output_dir)
    benchmark_lookup, manifest_payload = load_manifest_lookup(benchmark_dir)
    predictions_payload = load_json(predictions_path)
    predictions = select_predictions(
        predictions=list(predictions_payload.get("predictions", [])),
        sample_ids=args.sample_ids,
        limit=args.limit,
    )

    occupancy_cache: dict[str, np.ndarray] = {}
    per_sample_records: list[dict[str, Any]] = []
    progress_interval = max(0, int(args.progress_interval))
    total_predictions = len(predictions)
    for idx, prediction in enumerate(predictions, start=1):
        key = (str(prediction["source_name"]), str(prediction["sample_id"]))
        if key not in benchmark_lookup:
            raise KeyError(f"prediction sample not found in benchmark manifest: {key}")
        benchmark_sample = benchmark_lookup[key]
        occupancy_path = str(benchmark_sample["occupancy"]["path"])
        occupancy_grid = load_occupancy_cached(occupancy_cache, benchmark_dir, occupancy_path)
        per_sample_records.append(evaluate_one_prediction(prediction, benchmark_sample, occupancy_grid, args))
        if progress_interval > 0 and (idx == total_predictions or idx % progress_interval == 0):
            print(f"已评测 {idx}/{total_predictions} 个样本")

    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "benchmark_dir": path_to_record(benchmark_dir),
            "predictions": path_to_record(predictions_path),
            "manifest_schema_version": manifest_payload.get("schema_version"),
            "split": manifest_payload.get("split"),
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
    作用: CLI 入口，执行 benchmark predictions 评测
    输入: 无
    输出: None
    """
    parser = build_parser()
    args = parser.parse_args()
    summary = run_evaluation(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
