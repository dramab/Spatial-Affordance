#!/usr/bin/env python3
"""
tools/build_multimodal_dataset.py
---------------------------------
将 RGB 可视化图、点云、文本提示与 placement 几何监督整理为统一多模态数据集，
并生成 train/valid/test 标注清单。

用法:
    python tools/build_multimodal_dataset.py \
        --rgb-dir outputs/placement_rgb_bbox_vis \
        --label-json outputs/auto_labels/all_labels_polished.json \
        --source-dirs outputs/hope outputs/housecat6d \
        --output-dir data/annotations/placement_multimodal \
        --train-ratio 0.8 \
        --valid-ratio 0.1 \
        --seed 42

作用:
    - 对齐 RGB、文本、点云、placement sample
    - 从 placement grid_meta 读取每帧相机参数
    - 按 source_name 分层随机切分 train/valid/test

输入:
    --rgb-dir: RGB 可视化图片目录
    --label-json: auto label 结果 JSON
    --source-dirs: 一个或多个 placement 输出根目录
    --output-dir: 标注输出目录
    --train-ratio: 训练集比例，默认 0.8
    --valid-ratio: 验证集比例，默认 0.1
    --seed: 随机种子，默认 42

输出:
    output-dir/
        - train.json
        - valid.json
        - test.json
        - summary.json

使用示例:
    python tools/build_multimodal_dataset.py \
        --output-dir data/annotations/placement_multimodal \
        --train-ratio 0.8 \
        --valid-ratio 0.1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.bbox3d.bbox_utils import get_bbox_corners
from src.utils.coord_utils import rotation_z_3x3, transform_points

SCHEMA_VERSION = "placement_multimodal_dataset/v1"
CAMERA_REQUIRED_KEYS = ("fx", "fy", "cx", "cy", "img_w", "img_h", "E_c2w")
CAMERA_BACKFILL_HINT = (
    "请先运行 tools/backfill_placement_camera_meta.py 回填旧 placement 输出，"
    "或重新运行 tools/run_placement.py 生成带 camera 的 grid_meta。"
)


def build_parser() -> argparse.ArgumentParser:
    """
    用法: parser = build_parser()
    作用: 构建命令行参数解析器
    输入: 无
    输出: argparse.ArgumentParser，配置完成的解析器
    """
    parser = argparse.ArgumentParser(description="构建 RGB-点云-文本多模态数据集索引")
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=Path("outputs/placement_rgb_bbox_vis"),
        help="RGB 可视化图片目录",
    )
    parser.add_argument(
        "--label-json",
        type=Path,
        default=Path("outputs/auto_labels/all_labels_polished.json"),
        help="文本标签 JSON 路径",
    )
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        type=Path,
        default=[Path("outputs/hope"), Path("outputs/housecat6d")],
        help="placement 输出根目录列表",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/annotations/v1"),
        help="输出标注目录",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="训练集比例，默认 0.8",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="验证集比例，默认 0.1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42",
    )
    return parser


def load_json(json_path: Path):
    """
    用法: payload = load_json(Path("outputs/meta.json"))
    作用: 读取 JSON 文件
    输入: json_path: Path，JSON 路径
    输出: 任意 JSON 对象
    """
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_repo_relative(path: Path) -> str:
    """
    用法: rel = to_repo_relative(Path("outputs/demo/file.json"))
    作用: 将路径转换为相对仓库根目录的 POSIX 路径
    输入: path: Path，仓库内路径
    输出: str，相对路径
    """
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def parse_source_name_from_rgb_name(image_filename: str) -> Tuple[str, str]:
    """
    用法: source_name, sample_id = parse_source_name_from_rgb_name("hope__scene_0000_0000_obj_1_p000.png")
    作用: 从 RGB 文件名中解析 source_name 和 sample_id
    输入: image_filename: str，图片文件名
    输出: tuple[str, str]，来源名与 sample_id
    """
    if "__" not in image_filename:
        raise ValueError(f"Invalid RGB filename: {image_filename}")
    source_name, sample_part = image_filename.split("__", 1)
    return source_name, Path(sample_part).stem


def build_label_lookup(label_records: Iterable[dict]) -> Dict[Tuple[str, str], dict]:
    """
    用法: lookup = build_label_lookup(label_records)
    作用: 构建 (source_name, sample_id) 到文本标签的映射表
    输入: label_records: Iterable[dict]，标签记录序列
    输出: dict，键为 (source_name, sample_id)
    """
    lookup: Dict[Tuple[str, str], dict] = {}
    for record in label_records:
        image_filename = str(record.get("image_filename", ""))
        source_name = str(record.get("source_name", "")).strip()
        sample_id = str(record.get("sample_id", "")).strip()
        if not source_name or not sample_id:
            if image_filename:
                source_name, sample_id = parse_source_name_from_rgb_name(image_filename)
            else:
                raise ValueError(f"Invalid label record without source/sample id: {record}")
        key = (source_name, sample_id)
        if key in lookup:
            raise ValueError(f"Duplicated label key: {key}")
        lookup[key] = record
    return lookup


def collect_available_rgb_filenames(rgb_dir: Path) -> set[str]:
    """
    用法: rgb_filenames = collect_available_rgb_filenames(Path("outputs/placement_rgb_bbox_vis"))
    作用: 收集 RGB 目录下实际存在的图片文件名，用于过滤无 RGB 的样本
    输入: rgb_dir: Path，RGB 图片目录
    输出: set[str]，目录中存在的图片文件名集合
    """
    return {path.name for path in rgb_dir.glob("*") if path.is_file()}


def iter_sample_records(sample_dir: Path) -> Iterator[Tuple[Path, dict]]:
    """
    用法: for sample_json, record in iter_sample_records(Path("outputs/hope/samples")): ...
    作用: 展平单个 source 目录下的 placement sample 记录
    输入: sample_dir: Path，samples 子目录
    输出: Iterator[(Path, dict)]，样本 JSON 路径与单条样本记录
    """
    for sample_json in sorted(sample_dir.glob("*.json")):
        payload = load_json(sample_json)
        for record in payload.get("samples", []):
            yield sample_json, record


def validate_camera_record(camera: object, grid_meta_path: Path) -> dict:
    """
    用法: camera = validate_camera_record(payload["camera"], grid_meta_path)
    作用: 校验并标准化 grid_meta 中的相机字段
    输入: camera: JSON 对象；grid_meta_path: Path，用于错误定位
    输出: dict，可直接写入多模态标注的相机字段
    """
    if not isinstance(camera, MappingABC):
        raise ValueError(
            f"Invalid camera in grid_meta: {grid_meta_path}. "
            f"camera must be an object. {CAMERA_BACKFILL_HINT}"
        )

    missing_keys = [key for key in CAMERA_REQUIRED_KEYS if key not in camera]
    if missing_keys:
        raise ValueError(
            f"Missing camera keys {missing_keys} in grid_meta: {grid_meta_path}. "
            f"{CAMERA_BACKFILL_HINT}"
        )

    try:
        e_c2w = np.asarray(camera["E_c2w"], dtype=np.float64)
        normalized = {
            "fx": float(camera["fx"]),
            "fy": float(camera["fy"]),
            "cx": float(camera["cx"]),
            "cy": float(camera["cy"]),
            "img_w": int(camera["img_w"]),
            "img_h": int(camera["img_h"]),
            "E_c2w": e_c2w.tolist(),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid camera values in grid_meta: {grid_meta_path}") from exc

    if e_c2w.shape != (4, 4):
        raise ValueError(
            f"camera.E_c2w must have shape (4, 4) in grid_meta: {grid_meta_path}, "
            f"got {e_c2w.shape}"
        )
    if normalized["img_w"] <= 0 or normalized["img_h"] <= 0:
        raise ValueError(f"camera image size must be positive in grid_meta: {grid_meta_path}")
    return normalized


def load_camera_from_grid_meta(grid_meta_path: Path) -> dict:
    """
    用法: camera = load_camera_from_grid_meta(Path("outputs/hope/grid_meta/scene_0000_0000.json"))
    作用: 从 placement grid_meta 中读取相机参数
    输入: grid_meta_path: Path，placement 每帧 grid_meta 文件
    输出: dict，标准化后的相机字段
    """
    payload = load_json(grid_meta_path)
    if "camera" not in payload:
        raise ValueError(f"Missing camera in grid_meta: {grid_meta_path}. {CAMERA_BACKFILL_HINT}")
    return validate_camera_record(payload["camera"], grid_meta_path)


def build_frame_lookup(
    source_dirs: Iterable[Path],
    label_lookup: Mapping[Tuple[str, str], dict],
    available_rgb_filenames: set[str],
) -> Dict[Tuple[str, str, str], List[dict]]:
    """
    用法: frame_lookup = build_frame_lookup(source_dirs, label_lookup, available_rgb_filenames)
    作用: 从 placement 输出目录构建每条样本的基础记录，并按帧分组，仅保留 RGB 实际存在的样本
    输入: source_dirs: Iterable[Path]；label_lookup: 标签映射；
         available_rgb_filenames: RGB 文件名集合
    输出: dict，键为 (source_name, scene_id, frame_id)
    """
    frame_lookup: Dict[Tuple[str, str, str], List[dict]] = {}
    for source_dir in sorted(source_dirs):
        source_name = source_dir.name
        samples_dir = source_dir / "samples"
        point_cloud_dir = source_dir / "point_clouds"
        grid_meta_dir = source_dir / "grid_meta"
        if not samples_dir.exists():
            raise FileNotFoundError(f"Missing samples directory: {samples_dir}")

        for sample_json, record in iter_sample_records(samples_dir):
            sample_id = str(record["sample_id"])
            scene_id = str(record["scene_id"])
            frame_id = str(record["frame_id"])
            image_filename = f"{source_name}__{sample_id}.png"
            if image_filename not in available_rgb_filenames:
                continue

            label_key = (source_name, sample_id)
            if label_key not in label_lookup:
                raise KeyError(f"Missing label for sample: {label_key}")

            point_cloud_path = point_cloud_dir / f"{scene_id}_{frame_id}.ply"
            grid_meta_path = grid_meta_dir / f"{scene_id}_{frame_id}.json"

            base_record = {
                "sample_id": sample_id,
                "source_name": source_name,
                "scene_id": scene_id,
                "frame_id": frame_id,
                "image_filename": image_filename,
                "rgb_path": to_repo_relative(PROJECT_ROOT / "outputs/placement_rgb_bbox_vis" / image_filename),
                "point_cloud_path": to_repo_relative(point_cloud_path),
                "grid_meta_path": to_repo_relative(grid_meta_path),
                "sample_json_path": to_repo_relative(sample_json),
                "placement": record,
                "label_record": label_lookup[label_key],
            }
            frame_key = (source_name, scene_id, frame_id)
            frame_lookup.setdefault(frame_key, []).append(base_record)
    return frame_lookup


def enrich_records_with_frame_meta(
    frame_lookup: Mapping[Tuple[str, str, str], List[dict]],
    rgb_dir: Path,
) -> List[dict]:
    """
    用法: records = enrich_records_with_frame_meta(frame_lookup, rgb_dir)
    作用: 为每条样本补齐 placement grid_meta 中的相机参数
    输入: frame_lookup: 按帧分组的样本；rgb_dir: RGB 目录
    输出: list[dict]，完整样本列表
    """
    enriched_records: List[dict] = []
    for frame_key in sorted(frame_lookup):
        sample_group = frame_lookup[frame_key]
        point_cloud_rel = sample_group[0]["point_cloud_path"]
        point_cloud_path = PROJECT_ROOT / point_cloud_rel
        grid_meta_path = PROJECT_ROOT / sample_group[0]["grid_meta_path"]

        frame_required_paths = {
            "point_cloud": point_cloud_path,
            "grid_meta": grid_meta_path,
        }
        for key, path in frame_required_paths.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {key} file for frame {frame_key}: {path}"
                )
        camera_dict = load_camera_from_grid_meta(grid_meta_path)

        for item in sample_group:
            rgb_path = rgb_dir / item["image_filename"]
            sample_json_path = PROJECT_ROOT / item["sample_json_path"]
            label_record = item["label_record"]

            required_paths = {
                "rgb": rgb_path,
                "sample_json": sample_json_path,
            }
            for key, path in required_paths.items():
                if not path.exists():
                    raise FileNotFoundError(f"Missing {key} file for {item['sample_id']}: {path}")

            raw_label = str(label_record.get("label", "")).strip()
            polished_label = str(label_record.get("polished_label", "") or "").strip()
            if not raw_label:
                raise ValueError(f"Empty label for sample: {item['sample_id']}")

            placement_record = item["placement"]
            enriched_record = {
                "sample_id": item["sample_id"],
                "source_name": item["source_name"],
                "scene_id": item["scene_id"],
                "frame_id": item["frame_id"],
                "rgb_path": to_repo_relative(rgb_path),
                "point_cloud_path": item["point_cloud_path"],
                "prompt": raw_label,
                "polished_prompt": polished_label,
                "placement": build_minimal_placement(placement_record),
                "camera": camera_dict,
            }
            if placement_record.get("object_id") is not None:
                enriched_record["object_id"] = str(placement_record["object_id"])
            if placement_record.get("class_name") is not None:
                enriched_record["class_name"] = str(placement_record["class_name"])
            spatial_relation = label_record.get("spatial_relation")
            if isinstance(spatial_relation, dict):
                enriched_record["spatial_relation"] = spatial_relation
            enriched_records.append(enriched_record)
    return enriched_records


def stratified_split_by_source(
    records: Iterable[dict],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    用法: train_records, valid_records, test_records = stratified_split_by_source(records, 0.8, 0.1, 42)
    作用: 按 source_name 分层随机切分 train/valid/test
    输入: records: 样本列表；train_ratio: float；valid_ratio: float；seed: int
    输出: tuple[list[dict], list[dict], list[dict]]，训练集、验证集与测试集
    """
    grouped: Dict[str, List[dict]] = {}
    for record in records:
        grouped.setdefault(record["source_name"], []).append(record)

    train_records: List[dict] = []
    valid_records: List[dict] = []
    test_records: List[dict] = []
    for source_name, source_records in sorted(grouped.items()):
        sorted_records = sorted(source_records, key=lambda x: x["sample_id"])
        rng = random.Random(f"{seed}:{source_name}")
        rng.shuffle(sorted_records)
        train_end = int(len(sorted_records) * train_ratio)
        valid_end = int(len(sorted_records) * (train_ratio + valid_ratio))
        train_records.extend(sorted_records[:train_end])
        valid_records.extend(sorted_records[train_end:valid_end])
        test_records.extend(sorted_records[valid_end:])

    sorter = lambda x: (x["source_name"], x["sample_id"])
    return (
        sorted(train_records, key=sorter),
        sorted(valid_records, key=sorter),
        sorted(test_records, key=sorter),
    )


def build_split_payload(
    split_name: str,
    records: List[dict],
    seed: int,
    train_ratio: float,
    valid_ratio: float,
) -> dict:
    """
    用法: payload = build_split_payload("train", records, 42, 0.8, 0.1)
    作用: 构造 train/valid/test 标注 JSON 顶层结构
    输入: split_name: str；records: list[dict]；seed: int；train_ratio: float；valid_ratio: float
    输出: dict，可直接写入 JSON
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "split": split_name,
        "seed": int(seed),
        "split_strategy": {
            "type": "stratified_by_source_random_sample",
            "train_ratio": float(train_ratio),
            "valid_ratio": float(valid_ratio),
            "test_ratio": float(1.0 - train_ratio - valid_ratio),
        },
        "sample_count": len(records),
        "samples": records,
    }


def build_minimal_placement(record: Mapping[str, object]) -> dict:
    """
    用法: placement = build_minimal_placement(sample_record)
    作用: 从原始 placement sample 中提取精简监督字段
    输入: record: Mapping[str, object]，原始 placement 样本
    输出: dict，仅保留训练需要的目标框监督信息
    """
    center_world = np.asarray(record["center_world"], dtype=np.float64)
    canonical_aabb_object = np.asarray(record["canonical_aabb_object"], dtype=np.float64)
    yaw_degrees = float(record["yaw_degrees"])
    transform_world = np.asarray(record["transform_world"], dtype=np.float64)
    box_size = compute_yaw_aligned_box_size(
        canonical_aabb_object=canonical_aabb_object,
        transform_world=transform_world,
        yaw_degrees=yaw_degrees,
    )
    target_box = [
        float(center_world[0]),
        float(center_world[1]),
        float(center_world[2]),
        float(box_size[0]),
        float(box_size[1]),
        float(box_size[2]),
        yaw_degrees,
    ]
    return {
        "target_box": target_box,
    }


def compute_yaw_aligned_box_size(
    canonical_aabb_object: np.ndarray,
    transform_world: np.ndarray,
    yaw_degrees: float,
) -> np.ndarray:
    """
    用法: size = compute_yaw_aligned_box_size(canonical_aabb_object, transform_world, yaw_degrees)
    作用: 由 canonical AABB 和 object→world 姿态计算 yaw 对齐后的 3D box 尺寸
    输入: canonical_aabb_object: ndarray(6,)；transform_world: ndarray(4,4)；yaw_degrees: float
    输出: ndarray(3,)，格式为 [size_x, size_y, size_z]
    """
    canonical_aabb_object = np.asarray(canonical_aabb_object, dtype=np.float64)
    transform_world = np.asarray(transform_world, dtype=np.float64)
    if canonical_aabb_object.shape != (6,):
        raise ValueError(
            f"canonical_aabb_object must have shape (6,), got {canonical_aabb_object.shape}"
        )
    if transform_world.shape != (4, 4):
        raise ValueError(f"transform_world must have shape (4, 4), got {transform_world.shape}")

    corners_object = get_bbox_corners(canonical_aabb_object)
    corners_world = transform_points(corners_object, transform_world)
    center_world = corners_world.mean(axis=0, dtype=np.float64)

    # 先剥离 yaw，避免 yaw 对世界 AABB 的扩张重复进入尺寸监督。
    inverse_yaw = rotation_z_3x3(-np.deg2rad(float(yaw_degrees)))
    corners_yaw_aligned = (corners_world - center_world[None, :]) @ inverse_yaw.T
    box_size = corners_yaw_aligned.max(axis=0) - corners_yaw_aligned.min(axis=0)
    if np.any(box_size <= 0.0):
        raise ValueError(f"computed box size must be positive, got {box_size.tolist()}")
    return box_size.astype(np.float64)


def build_summary(train_records: List[dict], valid_records: List[dict], test_records: List[dict]) -> dict:
    """
    用法: summary = build_summary(train_records, valid_records, test_records)
    作用: 汇总 train/valid/test 的样本数与来源分布
    输入: train_records/valid_records/test_records: list[dict]
    输出: dict，summary 信息
    """
    def count_by_source(items: Iterable[dict]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in items:
            counts[item["source_name"]] = counts.get(item["source_name"], 0) + 1
        return dict(sorted(counts.items()))

    all_records = list(train_records) + list(valid_records) + list(test_records)
    return {
        "schema_version": SCHEMA_VERSION,
        "total_samples": len(all_records),
        "train_samples": len(train_records),
        "valid_samples": len(valid_records),
        "test_samples": len(test_records),
        "by_source": {
            "all": count_by_source(all_records),
            "train": count_by_source(train_records),
            "valid": count_by_source(valid_records),
            "test": count_by_source(test_records),
        },
    }


def save_json(output_path: Path, payload: dict) -> None:
    """
    用法: save_json(Path("data/annotations/train.json"), payload)
    作用: 将 JSON 写入磁盘
    输入: output_path: Path；payload: dict
    输出: None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def validate_args(args: argparse.Namespace) -> None:
    """
    用法: validate_args(args)
    作用: 校验命令行参数的基本合法性
    输入: args: argparse.Namespace
    输出: None，非法时抛出异常
    """
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be in (0, 1)")
    if not 0.0 < args.valid_ratio < 1.0:
        raise ValueError("--valid-ratio must be in (0, 1)")
    if args.train_ratio + args.valid_ratio >= 1.0:
        raise ValueError("--train-ratio + --valid-ratio must be less than 1")


def build_multimodal_dataset(
    rgb_dir: Path,
    label_json: Path,
    source_dirs: Iterable[Path],
    output_dir: Path,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> dict:
    """
    用法: summary = build_multimodal_dataset(rgb_dir, label_json, source_dirs, output_dir, 0.8, 0.1, 42)
    作用: 构建 train/valid/test 多模态数据集并写出 JSON
    输入: rgb_dir: Path；label_json: Path；source_dirs: Iterable[Path]；
         output_dir: Path；train_ratio: float；valid_ratio: float；seed: int
    输出: dict，summary 信息
    """
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not label_json.exists():
        raise FileNotFoundError(f"Label JSON not found: {label_json}")

    label_lookup = build_label_lookup(load_json(label_json))
    available_rgb_filenames = collect_available_rgb_filenames(rgb_dir)
    frame_lookup = build_frame_lookup(
        source_dirs=source_dirs,
        label_lookup=label_lookup,
        available_rgb_filenames=available_rgb_filenames,
    )
    all_records = enrich_records_with_frame_meta(
        frame_lookup=frame_lookup,
        rgb_dir=rgb_dir,
    )
    train_records, valid_records, test_records = stratified_split_by_source(
        records=all_records,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        seed=seed,
    )

    train_payload = build_split_payload("train", train_records, seed, train_ratio, valid_ratio)
    valid_payload = build_split_payload("valid", valid_records, seed, train_ratio, valid_ratio)
    test_payload = build_split_payload("test", test_records, seed, train_ratio, valid_ratio)
    summary = build_summary(train_records, valid_records, test_records)

    save_json(output_dir / "train.json", train_payload)
    save_json(output_dir / "valid.json", valid_payload)
    save_json(output_dir / "test.json", test_payload)
    save_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    """
    用法: main()
    作用: 命令行入口，构建并保存多模态数据集
    输入: 无
    输出: None
    """
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    summary = build_multimodal_dataset(
        rgb_dir=args.rgb_dir,
        label_json=args.label_json,
        source_dirs=args.source_dirs,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
