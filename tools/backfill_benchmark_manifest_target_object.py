#!/usr/bin/env python3
"""
tools/backfill_benchmark_manifest_target_object.py
--------------------------------------------------
将旧版 benchmark manifest 升级为新版目标物体投影评测所需格式。

用法:
    conda run -n spatial python tools/backfill_benchmark_manifest_target_object.py \
        --benchmark-dir benchmark/placement_v1 \
        --outputs-base outputs \
        --output-dir benchmark/placement_v2 \
        --overwrite

作用:
    - 读取旧 benchmark/manifest.json
    - 从 outputs/{source}/scene_objects/{scene_id}_{frame_id}.json 补齐 target_object.corners_world
    - 删除旧 manifest 中的 object_center_world 字段
    - 复制旧 benchmark 内 occupancy_grids 到新版 benchmark 目录

输入:
    --benchmark-dir: 旧 benchmark 目录
    --outputs-base: placement 输出根目录，包含 scene_objects
    --output-dir: 新版 benchmark 输出目录

输出:
    output-dir/
        - manifest.json
        - summary.json
        - occupancy_grids/

使用示例:
    conda run -n spatial python tools/backfill_benchmark_manifest_target_object.py \
        --benchmark-dir benchmark/placement_v1 \
        --outputs-base outputs \
        --output-dir benchmark/placement_v2 \
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_benchmark_manifest import SCHEMA_VERSION, find_scene_object, get_object_corners_world


def build_parser() -> argparse.ArgumentParser:
    """
    用法: parser = build_parser()
    作用: 构建命令行参数解析器
    输入: 无
    输出: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="升级旧 benchmark manifest，补齐 target_object 几何字段")
    parser.add_argument("--benchmark-dir", type=Path, required=True, help="旧 benchmark 目录")
    parser.add_argument("--outputs-base", type=Path, default=Path("outputs"), help="placement 输出根目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="新版 benchmark 输出目录")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖 output-dir 中已有 manifest/summary")
    return parser


def resolve_project_path(path_value: str | Path) -> Path:
    """
    用法: path = resolve_project_path("benchmark/placement_v1")
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
    用法: payload = load_json(Path("manifest.json"))
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


def build_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """
    用法: summary = build_summary(samples)
    作用: 汇总新版 benchmark 样本数量和 source 分布
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


def copy_occupancy_grids(benchmark_dir: Path, output_dir: Path) -> None:
    """
    用法: copy_occupancy_grids(old_dir, new_dir)
    作用: 将旧 benchmark 内 occupancy_grids 复制到新版 benchmark 输出目录
    输入: benchmark_dir: 旧 benchmark 目录；output_dir: 新 benchmark 目录
    输出: None
    """
    source_dir = benchmark_dir / "occupancy_grids"
    target_dir = output_dir / "occupancy_grids"
    if not source_dir.exists():
        return
    if source_dir.resolve() == target_dir.resolve():
        return
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)


def build_target_object_record(
    sample: Mapping[str, Any],
    outputs_base: Path,
) -> dict[str, Any]:
    """
    用法: record = build_target_object_record(sample, Path("outputs"))
    作用: 从 scene_objects 文件补齐单个样本的目标物体几何记录
    输入: sample: manifest 样本；outputs_base: placement 输出根目录
    输出: dict，包含 object_id、class_name、corners_world
    """
    source_name = str(sample["source_name"])
    scene_id = str(sample["scene_id"])
    frame_id = str(sample["frame_id"])
    object_id = str(sample["object_id"])
    scene_objects_path = outputs_base / source_name / "scene_objects" / f"{scene_id}_{frame_id}.json"
    if not scene_objects_path.exists():
        raise FileNotFoundError(f"scene_objects not found: {scene_objects_path}")
    scene_objects = load_json(scene_objects_path)
    target_object = find_scene_object(scene_objects, object_id, scene_objects_path, "target object")
    class_name = target_object.get("class_name", sample.get("class_name"))
    return {
        "object_id": object_id,
        "class_name": None if class_name is None else str(class_name),
        "corners_world": get_object_corners_world(target_object),
    }


def backfill_sample(
    sample: Mapping[str, Any],
    outputs_base: Path,
) -> dict[str, Any]:
    """
    用法: new_sample = backfill_sample(old_sample, Path("outputs"))
    作用: 将旧版 manifest 样本升级为新版样本格式
    输入: sample: 旧样本；outputs_base: placement 输出根目录
    输出: dict，删除 object_center_world 并补齐 target_object 的样本
    """
    new_sample = dict(sample)
    new_sample.pop("object_center_world", None)
    new_sample["target_object"] = build_target_object_record(sample, outputs_base)
    return new_sample


def backfill_benchmark_manifest(
    benchmark_dir: Path,
    outputs_base: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """
    用法: payload = backfill_benchmark_manifest(old_dir, outputs_base, new_dir, True)
    作用: 执行旧 benchmark manifest 升级并写出新版 benchmark
    输入: benchmark_dir/outputs_base/output_dir: Path；overwrite: 是否允许覆盖
    输出: dict，新版 manifest payload
    """
    benchmark_dir = resolve_project_path(benchmark_dir)
    outputs_base = resolve_project_path(outputs_base)
    output_dir = resolve_project_path(output_dir)
    manifest_path = benchmark_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"benchmark manifest not found: {manifest_path}")

    same_output_dir = benchmark_dir.resolve() == output_dir.resolve()
    if same_output_dir and not overwrite:
        raise ValueError("in-place backfill requires --overwrite")
    if not same_output_dir and (output_dir / "manifest.json").exists() and not overwrite:
        raise FileExistsError(f"output manifest already exists: {output_dir / 'manifest.json'}")

    old_payload = load_json(manifest_path)
    samples = [
        backfill_sample(sample, outputs_base)
        for sample in old_payload.get("samples", [])
    ]
    inputs = dict(old_payload.get("inputs", {}))
    inputs.update({
        "source_benchmark_dir": path_to_record(benchmark_dir),
        "outputs_base": path_to_record(outputs_base),
    })
    payload = dict(old_payload)
    payload.update({
        "schema_version": SCHEMA_VERSION,
        "sample_count": len(samples),
        "inputs": inputs,
        "samples": samples,
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    copy_occupancy_grids(benchmark_dir, output_dir)
    save_json(output_dir / "manifest.json", payload)
    save_json(output_dir / "summary.json", build_summary(samples))
    return payload


def main() -> None:
    """
    用法: main()
    作用: CLI 入口，执行 benchmark manifest backfill
    输入: 无
    输出: None
    """
    parser = build_parser()
    args = parser.parse_args()
    payload = backfill_benchmark_manifest(
        benchmark_dir=args.benchmark_dir,
        outputs_base=args.outputs_base,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps({"schema_version": SCHEMA_VERSION, "sample_count": payload["sample_count"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
