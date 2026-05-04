#!/usr/bin/env python3
"""
tools/backfill_placement_scene_objects.py
-----------------------------------------
为旧版 placement 输出补齐每帧全场景物体几何快照。

用法:
    conda run -n spatial python tools/backfill_placement_scene_objects.py \
        --source-dirs outputs/hope outputs/housecat6d \
        --dry-run

    conda run -n spatial python tools/backfill_placement_scene_objects.py \
        --source-dirs outputs/hope outputs/housecat6d

作用:
    - 遍历 placement 输出目录中的 samples/*.json，收集已处理 scene/frame
    - 仅通过 dataset adapter 加载对应帧 SceneData，不重新运行 placement planning
    - 写出 outputs/{source}/scene_objects/{scene_id}_{frame_id}.json

输入:
    --source-dirs: 一个或多个 placement 输出根目录
    --overwrite: 覆盖已经存在的 scene_objects JSON
    --dry-run: 只统计将要写入的文件，不写磁盘

输出:
    标准输出打印每个 source 的补全统计；非 dry-run 时写入 scene_objects/*.json

使用示例:
    conda run -n spatial python tools/backfill_placement_scene_objects.py \
        --source-dirs outputs/hope outputs/housecat6d
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.free_bbox.io_utils import scene_objects_to_json
from tools.backfill_placement_camera_meta import (
    DEFAULT_SOURCE_DIRS,
    infer_config_path,
    iter_source_frames,
    save_json,
)
from tools.run_placement import build_adapter, load_config


def build_parser() -> argparse.ArgumentParser:
    """
    用法: parser = build_parser()
    作用: 构建命令行参数解析器
    输入: 无
    输出: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="回填 placement scene_objects 全场景物体快照")
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        type=Path,
        default=DEFAULT_SOURCE_DIRS,
        help="placement 输出根目录列表",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的 scene_objects JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计将要写入的文件，不写入磁盘",
    )
    return parser


def make_empty_summary(source_dir: Path) -> dict:
    """
    用法: summary = make_empty_summary(Path("outputs/hope"))
    作用: 创建单个 source 的统计容器
    输入: source_dir: Path，placement 输出根目录
    输出: dict，统计字段
    """
    return {
        "source_dir": str(source_dir),
        "total_frames": 0,
        "updated": 0,
        "skipped_existing": 0,
        "failed": 0,
        "failures": [],
    }


def backfill_source_dir(source_dir: Path, overwrite: bool, dry_run: bool) -> dict:
    """
    用法: summary = backfill_source_dir(Path("outputs/hope"), False, False)
    作用: 为单个 placement source 补齐 scene_objects 文件
    输入: source_dir: Path；overwrite: bool；dry_run: bool
    输出: dict，补全统计
    """
    summary = make_empty_summary(source_dir)
    config_path = infer_config_path(source_dir)
    config = load_config(config_path)
    adapter = build_adapter(config)
    dataset_root = Path(str(config.get("dataset", {})["root_dir"]))
    scene_objects_dir = source_dir / "scene_objects"
    if not dry_run:
        scene_objects_dir.mkdir(parents=True, exist_ok=True)

    for scene_id, frame_id, _grid_meta_path in iter_source_frames(source_dir):
        summary["total_frames"] += 1
        output_path = scene_objects_dir / f"{scene_id}_{frame_id}.json"
        if output_path.exists() and not overwrite:
            summary["skipped_existing"] += 1
            continue
        if dry_run:
            summary["updated"] += 1
            continue

        try:
            scene_path = dataset_root / scene_id
            scene = adapter.load_scene(str(scene_path), frame_id)
            payload = scene_objects_to_json(scene)
            save_json(output_path, payload)
            summary["updated"] += 1
        except Exception as exc:
            summary["failed"] += 1
            summary["failures"].append(f"{output_path}: {exc}")

    return summary


def backfill_all(source_dirs: Iterable[Path], overwrite: bool, dry_run: bool) -> dict:
    """
    用法: summary = backfill_all([Path("outputs/hope")], False, False)
    作用: 批量补齐多个 placement source 的 scene_objects
    输入: source_dirs: Iterable[Path]；overwrite: bool；dry_run: bool
    输出: dict，总统计与逐 source 统计
    """
    source_summaries = [
        backfill_source_dir(source_dir=source_dir, overwrite=overwrite, dry_run=dry_run)
        for source_dir in source_dirs
    ]
    return {
        "dry_run": bool(dry_run),
        "overwrite": bool(overwrite),
        "sources": source_summaries,
        "total_frames": sum(item["total_frames"] for item in source_summaries),
        "updated": sum(item["updated"] for item in source_summaries),
        "skipped_existing": sum(item["skipped_existing"] for item in source_summaries),
        "failed": sum(item["failed"] for item in source_summaries),
    }


def main() -> None:
    """
    用法: main()
    作用: CLI 入口，执行 scene_objects 补全
    输入: 无
    输出: None
    """
    parser = build_parser()
    args = parser.parse_args()
    summary = backfill_all(
        source_dirs=args.source_dirs,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
