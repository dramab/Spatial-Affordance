#!/usr/bin/env python3
"""
tools/backfill_placement_camera_meta.py
---------------------------------------
为旧版 placement 输出的 grid_meta JSON 回填 camera 字段。

用法:
    python tools/backfill_placement_camera_meta.py \
        --source-dirs outputs/hope outputs/housecat6d \
        --dry-run

    python tools/backfill_placement_camera_meta.py \
        --source-dirs outputs/hope outputs/housecat6d

作用:
    - 遍历 placement 输出目录中的 samples/*.json
    - 根据 scene_id/frame_id 从原始数据集 adapter 读取标准化后的 CameraParams
    - 将 camera 写回同帧 grid_meta/*.json，供 build_multimodal_dataset 直接读取

输入:
    --source-dirs: 一个或多个 placement 输出根目录
    --overwrite: 覆盖已经存在的 grid_meta.camera
    --dry-run: 只统计将要回填的文件，不写磁盘

输出:
    标准输出打印每个 source 的回填统计；非 dry-run 时更新 grid_meta/*.json

使用示例:
    python tools/backfill_placement_camera_meta.py \
        --source-dirs outputs/hope outputs/housecat6d
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.free_bbox.io_utils import camera_to_json
from tools.run_placement import build_adapter, load_config


DEFAULT_SOURCE_DIRS = [Path("outputs/hope"), Path("outputs/housecat6d")]
CONFIG_BY_SOURCE_KEY = {
    "hope": PROJECT_ROOT / "configs/annotation/placement.yaml",
    "housecat": PROJECT_ROOT / "configs/annotation/placement_housecat6d.yaml",
    "ycbv": PROJECT_ROOT / "configs/annotation/placement_ycbv_test.yaml",
    "ycb_video": PROJECT_ROOT / "configs/annotation/placement_ycbv_test.yaml",
    "scannet": PROJECT_ROOT / "configs/annotation/placement_scannet.yaml",
    "dopose": PROJECT_ROOT / "configs/annotation/placement_dopose.yaml",
}


def build_parser() -> argparse.ArgumentParser:
    """
    用法: parser = build_parser()
    作用: 构建命令行参数解析器
    输入: 无
    输出: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="回填 placement grid_meta 中的 camera 字段")
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
        help="覆盖已经存在的 grid_meta.camera",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计将要回填的文件，不写入磁盘",
    )
    return parser


def infer_config_path(source_dir: Path) -> Path:
    """
    用法: config_path = infer_config_path(Path("outputs/hope"))
    作用: 根据 placement 输出目录名推断原始数据集配置
    输入: source_dir: Path，placement 输出根目录
    输出: Path，对应 YAML 配置路径
    """
    source_name = source_dir.name.lower()
    for key, config_path in CONFIG_BY_SOURCE_KEY.items():
        if key in source_name:
            return config_path
    raise KeyError(
        f"Cannot infer dataset config for source directory: {source_dir}. "
        "Directory name must contain one of: hope, housecat, ycbv, ycb_video, scannet, dopose."
    )


def load_json(json_path: Path) -> dict:
    """
    用法: payload = load_json(Path("outputs/hope/grid_meta/scene_0000_0000.json"))
    作用: 读取 JSON 文件
    输入: json_path: Path，JSON 路径
    输出: dict，解析后的 JSON 对象
    """
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(json_path: Path, payload: dict) -> None:
    """
    用法: save_json(path, payload)
    作用: 将 JSON 对象写回磁盘
    输入: json_path: Path；payload: dict
    输出: None
    """
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def iter_source_frames(source_dir: Path) -> Iterator[Tuple[str, str, Path]]:
    """
    用法: for scene_id, frame_id, grid_meta_path in iter_source_frames(Path("outputs/hope")): ...
    作用: 从 samples/*.json 收集每帧对应的 grid_meta 路径
    输入: source_dir: Path，placement 输出根目录
    输出: Iterator[(scene_id, frame_id, grid_meta_path)]
    """
    samples_dir = source_dir / "samples"
    grid_meta_dir = source_dir / "grid_meta"
    if not samples_dir.exists():
        raise FileNotFoundError(f"Missing samples directory: {samples_dir}")
    if not grid_meta_dir.exists():
        raise FileNotFoundError(f"Missing grid_meta directory: {grid_meta_dir}")

    seen: set[Tuple[str, str]] = set()
    for sample_json in sorted(samples_dir.glob("*.json")):
        payload = load_json(sample_json)
        samples = list(payload.get("samples", []))
        scene_id = str(payload.get("scene_id") or "").strip()
        frame_id = str(payload.get("frame_id") or "").strip()
        if (not scene_id or not frame_id) and samples:
            scene_id = str(samples[0].get("scene_id") or "").strip()
            frame_id = str(samples[0].get("frame_id") or "").strip()
        if not scene_id or not frame_id:
            raise ValueError(f"Cannot infer scene/frame from sample JSON: {sample_json}")

        frame_key = (scene_id, frame_id)
        if frame_key in seen:
            continue
        seen.add(frame_key)
        grid_meta_path = grid_meta_dir / f"{scene_id}_{frame_id}.json"
        if not grid_meta_path.exists():
            fallback_path = grid_meta_dir / f"{sample_json.stem}.json"
            if fallback_path.exists():
                grid_meta_path = fallback_path
        yield scene_id, frame_id, grid_meta_path


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
        "missing_grid_meta": 0,
        "failed": 0,
        "failures": [],
    }


def backfill_source_dir(source_dir: Path, overwrite: bool, dry_run: bool) -> dict:
    """
    用法: summary = backfill_source_dir(Path("outputs/hope"), overwrite=False, dry_run=False)
    作用: 为单个 placement source 回填 grid_meta.camera
    输入: source_dir: Path；overwrite: bool；dry_run: bool
    输出: dict，回填统计
    """
    summary = make_empty_summary(source_dir)
    config_path = infer_config_path(source_dir)
    config = load_config(config_path)
    adapter = build_adapter(config)
    dataset_root = Path(str(config.get("dataset", {})["root_dir"]))

    for scene_id, frame_id, grid_meta_path in iter_source_frames(source_dir):
        summary["total_frames"] += 1
        if not grid_meta_path.exists():
            summary["missing_grid_meta"] += 1
            summary["failures"].append(f"Missing grid_meta: {grid_meta_path}")
            continue

        try:
            grid_meta = load_json(grid_meta_path)
            if "camera" in grid_meta and not overwrite:
                summary["skipped_existing"] += 1
                continue
            if dry_run:
                summary["updated"] += 1
                continue

            scene_path = dataset_root / scene_id
            scene = adapter.load_scene(str(scene_path), frame_id)
            grid_meta["camera"] = camera_to_json(scene.camera)
            save_json(grid_meta_path, grid_meta)
            summary["updated"] += 1
        except Exception as exc:
            summary["failed"] += 1
            summary["failures"].append(f"{grid_meta_path}: {exc}")

    return summary


def backfill_all(source_dirs: Iterable[Path], overwrite: bool, dry_run: bool) -> dict:
    """
    用法: summary = backfill_all([Path("outputs/hope")], overwrite=False, dry_run=False)
    作用: 批量回填多个 placement source
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
        "missing_grid_meta": sum(item["missing_grid_meta"] for item in source_summaries),
        "failed": sum(item["failed"] for item in source_summaries),
    }


def main() -> None:
    """
    用法: main()
    作用: CLI 入口，执行旧 placement camera 回填
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
    if summary["missing_grid_meta"] > 0 or summary["failed"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
