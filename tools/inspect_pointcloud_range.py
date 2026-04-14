"""
tools/inspect_pointcloud_range.py
---------------------------------
职责：统计项目内点云文件的空间范围，并输出推荐的固定体素范围配置。

功能：
- 扫描给定 glob 下的点云文件
- 读取点云 xyz 范围并汇总全局最小/最大值
- 输出推荐的 point_cloud_range_cm 到项目内 outputs 目录

输入：
    --glob: str 点云文件 glob，如 "outputs/**/*.npy"
    --limit: int 最多统计文件数
    --padding-cm: float 输出范围额外扩边
    --output-dir: str 统计结果目录

输出：
    summary.json 统计摘要

用法：
    python tools/inspect_pointcloud_range.py \
        --glob "outputs/**/*.npy" \
        --padding-cm 10.0 \
        --output-dir outputs/voxelnet_stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d


def load_pointcloud_xyz(path: Path) -> np.ndarray:
    """
    从常见点云文件读取 xyz。

    输入:
        path: Path 点云文件路径，支持 .npy/.npz/.ply/.pcd/.xyz
    输出:
        (N, 3) ndarray 点云坐标，单位沿用文件本身
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        if "points" in data:
            points = data["points"]
        else:
            first_key = list(data.keys())[0]
            points = data[first_key]
    elif suffix in {".ply", ".pcd", ".xyz"}:
        pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pcd.points, dtype=np.float64)
    else:
        raise ValueError(f"unsupported point cloud format: {path}")

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"invalid point cloud shape: {path} -> {points.shape}")
    return points[:, :3]


def inspect_pointcloud_range(
        paths: Iterable[Path],
        padding_cm: float) -> dict:
    """
    汇总点云文件的全局空间范围。

    输入:
        paths: 可迭代的点云文件路径
        padding_cm: float 输出范围的额外扩边
    输出:
        dict 统计摘要
    """
    global_min = None
    global_max = None
    num_files = 0
    num_points = 0

    for path in paths:
        xyz = load_pointcloud_xyz(path)
        if xyz.shape[0] == 0:
            continue
        xyz_min = xyz.min(axis=0)
        xyz_max = xyz.max(axis=0)
        global_min = xyz_min if global_min is None else np.minimum(global_min, xyz_min)
        global_max = xyz_max if global_max is None else np.maximum(global_max, xyz_max)
        num_files += 1
        num_points += int(xyz.shape[0])

    if global_min is None or global_max is None:
        raise ValueError("no valid point clouds found")

    padded_min = global_min - float(padding_cm)
    padded_max = global_max + float(padding_cm)

    return {
        "num_files": num_files,
        "num_points": num_points,
        "global_min_cm": global_min.tolist(),
        "global_max_cm": global_max.tolist(),
        "recommended_point_cloud_range_cm": np.concatenate(
            [padded_min, padded_max]).tolist(),
        "padding_cm": float(padding_cm),
    }


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    输入:
        无
    输出:
        argparse.Namespace 参数对象
    """
    parser = argparse.ArgumentParser(description="Inspect point cloud range in cm")
    parser.add_argument("--glob", required=True, help="点云文件 glob 模式")
    parser.add_argument("--limit", type=int, default=0, help="最多统计的文件数，0 表示不限制")
    parser.add_argument("--padding-cm", type=float, default=10.0, help="推荐范围的额外扩边")
    parser.add_argument(
        "--output-dir",
        default="outputs/voxelnet_stats",
        help="结果输出目录",
    )
    return parser.parse_args()


def main() -> None:
    """
    CLI 入口。

    输入:
        无，读取命令行参数
    输出:
        无，将统计结果写入 summary.json
    """
    args = parse_args()
    paths = sorted(Path(".").glob(args.glob))
    if args.limit > 0:
        paths = paths[:args.limit]

    summary = inspect_pointcloud_range(paths, padding_cm=args.padding_cm)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"[OK] wrote summary to {output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
