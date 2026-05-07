#!/usr/bin/env python3
"""
tools/backfill_multimodal_object_center.py
------------------------------------------
将旧版 placement 多模态 annotation 补全为包含移动前物体中心监督的 v2 annotation。

用法:
    conda run -n spatial python tools/backfill_multimodal_object_center.py \
        --annotation-dir data/annotations/placement_multimodal \
        --output-dir data/annotations/placement_multimodal_v2 \
        --source-dirs outputs/hope outputs/housecat6d

作用:
    - 读取旧 annotation 的 train/valid/test JSON
    - 从 outputs/{source}/samples/{scene_id}_{frame_id}.json 查找原始 placement sample
    - 使用 canonical_aabb_object 和 original_pose_world 计算 placement.object_center
    - 写出新的 v2 annotation 目录，不修改原始 annotation 和 placement 输出

输入:
    --annotation-dir: 旧 annotation 目录
    --output-dir: v2 annotation 输出目录，默认是 <annotation-dir>_v2
    --source-dirs: 一个或多个 placement 输出根目录
    --overwrite: 允许覆盖已存在的输出目录
    --dry-run: 只统计补全结果，不写入磁盘

输出:
    output-dir/
        - train.json
        - valid.json
        - test.json
        - summary.json

使用示例:
    conda run -n spatial python tools/backfill_multimodal_object_center.py \
        --annotation-dir data/annotations/placement_multimodal_simple \
        --output-dir data/annotations/placement_multimodal_simple_v2 \
        --source-dirs outputs/dopose outputs/hope outputs/housecat6d outputs/ycbv_test
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_benchmark_manifest import parse_sample_identity
from tools.build_multimodal_dataset import SCHEMA_VERSION, compute_original_object_center


DEFAULT_SPLITS = ("train", "valid", "test")


def build_parser() -> argparse.ArgumentParser:
    """
    用法: parser = build_parser()
    作用: 构建命令行参数解析器
    输入: 无
    输出: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="补全多模态 annotation 的 placement.object_center 字段")
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        required=True,
        help="旧 annotation 目录，包含 train/valid/test JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="v2 annotation 输出目录，默认是 <annotation-dir>_v2",
    )
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="placement 输出根目录列表，如 outputs/hope outputs/housecat6d",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="需要补全的数据划分，默认 train valid test",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的输出目录",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计将要写出的样本，不写入磁盘",
    )
    return parser


def load_json(json_path: Path) -> dict[str, Any]:
    """
    用法: payload = load_json(Path("data/annotations/train.json"))
    作用: 读取 JSON 文件
    输入: json_path: Path，JSON 文件路径
    输出: dict，解析后的 JSON 对象
    """
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(json_path: Path, payload: Mapping[str, Any]) -> None:
    """
    用法: save_json(Path("data/annotations/train.json"), payload)
    作用: 将 JSON 对象写入磁盘
    输入: json_path: Path；payload: Mapping，待写入对象
    输出: None
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def default_output_dir(annotation_dir: Path) -> Path:
    """
    用法: output_dir = default_output_dir(Path("data/annotations/placement_multimodal"))
    作用: 构造默认 v2 annotation 输出目录
    输入: annotation_dir: Path，旧 annotation 目录
    输出: Path，默认输出目录
    """
    return annotation_dir.parent / f"{annotation_dir.name}_v2"


def build_source_dir_lookup(source_dirs: Iterable[Path]) -> dict[str, Path]:
    """
    用法: lookup = build_source_dir_lookup([Path("outputs/hope")])
    作用: 按 source_name 建立 placement 输出目录查找表
    输入: source_dirs: Iterable[Path]，placement 输出根目录列表
    输出: dict[str, Path]，键为 source_dir.name
    """
    lookup: dict[str, Path] = {}
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        source_name = source_path.name
        if source_name in lookup:
            raise ValueError(f"Duplicated source directory name: {source_name}")
        samples_dir = source_path / "samples"
        if not samples_dir.exists():
            raise FileNotFoundError(f"Missing samples directory: {samples_dir}")
        lookup[source_name] = source_path
    return lookup


def find_raw_sample_record(
    sample: Mapping[str, Any],
    source_dir_lookup: Mapping[str, Path],
    raw_payload_cache: dict[Path, dict[str, Any]],
) -> Mapping[str, Any]:
    """
    用法: raw = find_raw_sample_record(sample, source_lookup, cache)
    作用: 从原始 placement samples JSON 中查找 annotation 样本对应的原始记录
    输入: sample: annotation 样本；source_dir_lookup: source_name 到输出目录映射；raw_payload_cache: JSON 缓存
    输出: Mapping[str, Any]，原始 placement sample 记录
    """
    source_name = str(sample["source_name"])
    sample_id = str(sample["sample_id"])
    if source_name not in source_dir_lookup:
        raise KeyError(f"source_name={source_name} not found in --source-dirs")

    scene_id, frame_id, _object_id = parse_sample_identity(sample)
    raw_path = source_dir_lookup[source_name] / "samples" / f"{scene_id}_{frame_id}.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"raw placement sample JSON not found: {raw_path}")
    if raw_path not in raw_payload_cache:
        raw_payload_cache[raw_path] = load_json(raw_path)

    for record in raw_payload_cache[raw_path].get("samples", []):
        if str(record.get("sample_id")) == sample_id:
            return record
    raise KeyError(f"sample_id={sample_id} not found in raw placement file: {raw_path}")


def compute_object_center_from_raw_sample(raw_sample: Mapping[str, Any]) -> list[float]:
    """
    用法: center = compute_object_center_from_raw_sample(raw_sample)
    作用: 从原始 placement sample 计算移动前物体中心
    输入: raw_sample: Mapping，包含 canonical_aabb_object 和 original_pose_world
    输出: list[float]，长度为 3 的世界坐标中心
    """
    canonical_aabb = np.asarray(raw_sample["canonical_aabb_object"], dtype=np.float64)
    original_pose = np.asarray(raw_sample["original_pose_world"], dtype=np.float64)
    center = compute_original_object_center(
        canonical_aabb_object=canonical_aabb,
        original_pose_world=original_pose,
    )
    return [float(center[0]), float(center[1]), float(center[2])]


def backfill_sample(
    sample: Mapping[str, Any],
    source_dir_lookup: Mapping[str, Path],
    raw_payload_cache: dict[Path, dict[str, Any]],
) -> dict[str, Any]:
    """
    用法: sample_v2 = backfill_sample(sample, source_lookup, cache)
    作用: 为单条 annotation 样本补齐 placement.object_center
    输入: sample: 旧 annotation 样本；source_dir_lookup: source 查找表；raw_payload_cache: 原始 samples 缓存
    输出: dict，补齐后的 v2 annotation 样本
    """
    sample_v2 = copy.deepcopy(dict(sample))
    placement = dict(sample_v2.get("placement") or {})
    if "target_box" not in placement:
        raise ValueError(f"missing placement.target_box for sample_id={sample_v2.get('sample_id')}")

    raw_sample = find_raw_sample_record(
        sample=sample_v2,
        source_dir_lookup=source_dir_lookup,
        raw_payload_cache=raw_payload_cache,
    )
    placement["object_center"] = compute_object_center_from_raw_sample(raw_sample)
    sample_v2["placement"] = placement
    return sample_v2


def backfill_split_payload(
    payload: Mapping[str, Any],
    source_dir_lookup: Mapping[str, Path],
    raw_payload_cache: dict[Path, dict[str, Any]],
) -> dict[str, Any]:
    """
    用法: payload_v2 = backfill_split_payload(payload, source_lookup, cache)
    作用: 为一个 split payload 中的所有样本补齐 object_center
    输入: payload: 旧 split JSON；source_dir_lookup: source 查找表；raw_payload_cache: 原始 samples 缓存
    输出: dict，v2 split JSON
    """
    samples = [
        backfill_sample(sample, source_dir_lookup, raw_payload_cache)
        for sample in payload.get("samples", [])
    ]
    payload_v2 = copy.deepcopy(dict(payload))
    payload_v2["schema_version"] = SCHEMA_VERSION
    payload_v2["sample_count"] = len(samples)
    payload_v2["samples"] = samples
    return payload_v2


def build_summary(split_payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """
    用法: summary = build_summary({"train": payload})
    作用: 汇总补全后的 annotation 样本数量与 source 分布
    输入: split_payloads: split 名到 payload 的映射
    输出: dict，summary 信息
    """
    by_source: dict[str, dict[str, int]] = {}
    total_samples = 0
    for split_name, payload in split_payloads.items():
        split_counts: dict[str, int] = {}
        for sample in payload.get("samples", []):
            source_name = str(sample["source_name"])
            split_counts[source_name] = split_counts.get(source_name, 0) + 1
            total_samples += 1
        by_source[split_name] = dict(sorted(split_counts.items()))

    return {
        "schema_version": SCHEMA_VERSION,
        "total_samples": total_samples,
        "split_samples": {
            split_name: int(payload.get("sample_count", 0))
            for split_name, payload in split_payloads.items()
        },
        "by_source": by_source,
    }


def validate_output_dir(annotation_dir: Path, output_dir: Path, overwrite: bool, dry_run: bool) -> None:
    """
    用法: validate_output_dir(annotation_dir, output_dir, overwrite=False, dry_run=False)
    作用: 校验输出目录不会误覆盖输入目录或已有结果
    输入: annotation_dir: Path；output_dir: Path；overwrite: bool；dry_run: bool
    输出: None，非法时抛出异常
    """
    if output_dir.resolve() == annotation_dir.resolve():
        raise ValueError("--output-dir must differ from --annotation-dir")
    if output_dir.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"output directory already exists, use --overwrite: {output_dir}")


def backfill_annotation_dir(
    annotation_dir: Path,
    output_dir: Path | None,
    source_dirs: Iterable[Path],
    splits: Iterable[str] = DEFAULT_SPLITS,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    用法: summary = backfill_annotation_dir(annotation_dir, None, source_dirs)
    作用: 将旧 annotation 目录补全为 v2 annotation 目录
    输入: annotation_dir/output_dir/source_dirs/splits/overwrite/dry_run
    输出: dict，补全统计 summary
    """
    annotation_dir = Path(annotation_dir)
    output_dir = default_output_dir(annotation_dir) if output_dir is None else Path(output_dir)
    if not annotation_dir.exists():
        raise FileNotFoundError(f"annotation directory not found: {annotation_dir}")
    validate_output_dir(annotation_dir, output_dir, overwrite=overwrite, dry_run=dry_run)

    source_dir_lookup = build_source_dir_lookup(source_dirs)
    raw_payload_cache: dict[Path, dict[str, Any]] = {}
    split_payloads: dict[str, dict[str, Any]] = {}

    for split_name in splits:
        split = str(split_name)
        split_path = annotation_dir / f"{split}.json"
        if not split_path.exists():
            continue
        split_payloads[split] = backfill_split_payload(
            payload=load_json(split_path),
            source_dir_lookup=source_dir_lookup,
            raw_payload_cache=raw_payload_cache,
        )

    if not split_payloads:
        raise FileNotFoundError(f"no split JSON files found in annotation directory: {annotation_dir}")

    summary = build_summary(split_payloads)
    summary.update({
        "annotation_dir": annotation_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "dry_run": bool(dry_run),
        "overwrite": bool(overwrite),
        "raw_files_read": len(raw_payload_cache),
    })
    if dry_run:
        return summary

    for split_name, payload in split_payloads.items():
        save_json(output_dir / f"{split_name}.json", payload)
    save_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    """
    用法: main()
    作用: CLI 入口，补全旧多模态 annotation
    输入: 无，参数来自命令行
    输出: None，终端打印 JSON summary
    """
    args = build_parser().parse_args()
    summary = backfill_annotation_dir(
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir,
        source_dirs=args.source_dirs,
        splits=args.splits,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
