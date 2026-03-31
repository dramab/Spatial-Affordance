"""
src/utils/placement_stats.py
----------------------------
Placement 输出统计工具：读取 placement samples JSON，汇总场景、类别、
候选框样本数等信息，并导出表格与图表。

用法:
    from pathlib import Path
    from src.utils.placement_stats import (
        aggregate_statistics,
        export_statistics,
    )

    summary = aggregate_statistics([
        Path("outputs/housecat6d_placement10"),
        Path("outputs/placement_hope5"),
    ])
    export_statistics(summary, Path("outputs/placement_stats_combined"))
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class CategorySummary:
    """
    用法: CategorySummary(...)
    作用: 表示单个类别的统计结果
    输入: category/sample_count/object_count/scene_count/ratio
    输出: CategorySummary 实例
    """

    category: str
    sample_count: int
    object_count: int
    scene_count: int
    ratio: float


def collect_sample_files(directories: Iterable[Path]) -> List[Path]:
    """
    用法: collect_sample_files([Path("outputs/placement")])
    作用: 收集一个或多个 placement 输出目录下 samples 子目录中的 JSON 文件
    输入: directories: Iterable[Path]，placement 输出根目录或 samples 目录
    输出: List[Path]，按路径排序后的样本文件列表
    """
    sample_files: List[Path] = []
    for directory in directories:
        directory = Path(directory)
        sample_dir = directory if directory.name == "samples" else directory / "samples"
        if not sample_dir.is_dir():
            continue
        sample_files.extend(sorted(sample_dir.glob("*.json")))
    return sorted(sample_files)


def normalize_category_name(name: str) -> str:
    """
    用法: normalize_category_name("  Butter ")
    作用: 统一类别名格式，避免空白差异导致同类被拆分
    输入: name: str，原始类别名
    输出: str，规范化后的类别名
    """
    return str(name).strip()


def load_sample_payload(sample_file: Path) -> Dict:
    """
    用法: load_sample_payload(Path("outputs/.../samples/scene_x.json"))
    作用: 读取单个样本 JSON 文件
    输入: sample_file: Path，样本文件路径
    输出: Dict，解析后的 JSON 数据
    """
    with sample_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _infer_scene_id(payload: Dict, sample_file: Path) -> str:
    """
    用法: _infer_scene_id(payload, sample_file)
    作用: 从 payload 或文件名中提取场景标识
    输入: payload: Dict；sample_file: Path
    输出: str，场景 ID
    """
    scene_id = payload.get("scene_id")
    if scene_id:
        return str(scene_id)

    stem_parts = sample_file.stem.split("_")
    if len(stem_parts) >= 2 and stem_parts[0] == "scene":
        return "_".join(stem_parts[:2])
    return sample_file.stem


def _infer_frame_id(payload: Dict, sample_file: Path) -> str:
    """
    用法: _infer_frame_id(payload, sample_file)
    作用: 从 payload 或文件名中提取帧标识
    输入: payload: Dict；sample_file: Path
    输出: str，帧 ID
    """
    frame_id = payload.get("frame_id")
    if frame_id is not None:
        return str(frame_id)

    stem_parts = sample_file.stem.split("_")
    if len(stem_parts) >= 3 and stem_parts[0] == "scene":
        return stem_parts[-1]
    return sample_file.stem


def aggregate_statistics(directories: Iterable[Path]) -> Dict:
    """
    用法: aggregate_statistics([Path("dir1"), Path("dir2")])
    作用: 汇总多个 placement 输出目录的场景、类别、候选框样本统计
    输入: directories: Iterable[Path]，待统计目录列表
    输出: Dict，包含总览统计、类别统计和场景分布
    """
    sample_files = collect_sample_files(directories)
    category_samples: Counter = Counter()
    category_objects: Dict[str, set] = defaultdict(set)
    category_scenes: Dict[str, set] = defaultdict(set)
    scene_samples: Counter = Counter()
    all_scenes = set()
    source_file_count = 0

    for sample_file in sample_files:
        try:
            payload = load_sample_payload(sample_file)
        except (json.JSONDecodeError, OSError) as err:
            print(f"[WARN] Skip invalid sample file: {sample_file} ({err})", file=sys.stderr)
            continue

        source_file_count += 1
        scene_id = _infer_scene_id(payload, sample_file)
        frame_id = _infer_frame_id(payload, sample_file)
        all_scenes.add(scene_id)

        for sample in payload.get("samples", []):
            class_name = normalize_category_name(sample.get("class_name", "unknown"))
            object_id = str(sample.get("object_id", "unknown_object"))
            category_samples[class_name] += 1
            category_objects[class_name].add((scene_id, frame_id, object_id))
            category_scenes[class_name].add(scene_id)
            scene_samples[scene_id] += 1

    total_samples = int(sum(category_samples.values()))
    total_scenes = len(all_scenes)

    category_stats = []
    for category, sample_count in category_samples.most_common():
        ratio = (sample_count / total_samples) if total_samples else 0.0
        category_stats.append(
            CategorySummary(
                category=category,
                sample_count=int(sample_count),
                object_count=len(category_objects[category]),
                scene_count=len(category_scenes[category]),
                ratio=ratio,
            )
        )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_directories": [str(Path(directory)) for directory in directories],
        "total_scenes": total_scenes,
        "total_categories": len(category_stats),
        "total_samples": total_samples,
        "source_file_count": source_file_count,
        "avg_samples_per_scene": (total_samples / total_scenes) if total_scenes else 0.0,
        "category_stats": [
            {
                "category": item.category,
                "sample_count": item.sample_count,
                "object_count": item.object_count,
                "scene_count": item.scene_count,
                "ratio": item.ratio,
            }
            for item in category_stats
        ],
        "scene_distribution": [
            {"scene_id": scene_id, "sample_count": int(sample_count)}
            for scene_id, sample_count in sorted(
                scene_samples.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
    }


def write_summary_json(summary: Dict, output_dir: Path) -> Path:
    """
    用法: write_summary_json(summary, Path("outputs/stats"))
    作用: 导出汇总 JSON
    输入: summary: Dict；output_dir: Path
    输出: Path，生成的 JSON 路径
    """
    path = output_dir / "summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return path


def write_category_csv(summary: Dict, output_dir: Path) -> Path:
    """
    用法: write_category_csv(summary, Path("outputs/stats"))
    作用: 导出类别统计 CSV
    输入: summary: Dict；output_dir: Path
    输出: Path，生成的 CSV 路径
    """
    path = output_dir / "category_counts.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["category", "sample_count", "object_count", "scene_count", "ratio"],
        )
        writer.writeheader()
        for row in summary["category_stats"]:
            writer.writerow(row)
    return path


def write_scene_csv(summary: Dict, output_dir: Path) -> Path:
    """
    用法: write_scene_csv(summary, Path("outputs/stats"))
    作用: 导出按场景聚合的样本分布 CSV
    输入: summary: Dict；output_dir: Path
    输出: Path，生成的 CSV 路径
    """
    path = output_dir / "scene_distribution.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene_id", "sample_count"])
        writer.writeheader()
        for row in summary["scene_distribution"]:
            writer.writerow(row)
    return path


def write_summary_text(summary: Dict, output_dir: Path, top_k: int = 10) -> Path:
    """
    用法: write_summary_text(summary, Path("outputs/stats"), top_k=10)
    作用: 导出便于人工阅读的文本摘要
    输入: summary: Dict；output_dir: Path；top_k: int，展示前多少个类别
    输出: Path，生成的文本路径
    """
    path = output_dir / "summary.txt"
    lines = [
        "Placement 统计摘要",
        f"生成时间: {summary['generated_at']}",
        f"输入目录: {', '.join(summary['input_directories'])}",
        f"总场景数: {summary['total_scenes']}",
        f"总类别数: {summary['total_categories']}",
        f"总样本数: {summary['total_samples']}",
        f"样本文件数: {summary['source_file_count']}",
        f"平均每场景样本数: {summary['avg_samples_per_scene']:.2f}",
        "",
        f"Top {top_k} 类别:",
    ]
    for row in summary["category_stats"][:top_k]:
        lines.append(
            f"{row['category']}: samples={row['sample_count']}, "
            f"objects={row['object_count']}, scenes={row['scene_count']}, "
            f"ratio={row['ratio']:.2%}"
        )

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def plot_category_bar(summary: Dict, output_dir: Path, top_k: int = 20) -> Path:
    """
    用法: plot_category_bar(summary, Path("outputs/stats"), top_k=20)
    作用: 绘制类别样本数柱状图
    输入: summary: Dict；output_dir: Path；top_k: int，展示前多少个类别
    输出: Path，生成的图片路径
    """
    rows = summary["category_stats"][:top_k]
    categories = [row["category"] for row in rows]
    counts = [row["sample_count"] for row in rows]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(categories, counts, color="#2f6f9f")
    ax.set_title(f"Top {top_k} Categories by Candidate Count")
    ax.set_xlabel("Category")
    ax.set_ylabel("Candidate Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    path = output_dir / f"category_bar_top{top_k}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_category_pie(summary: Dict, output_dir: Path, top_k: int = 10) -> Path:
    """
    用法: plot_category_pie(summary, Path("outputs/stats"), top_k=10)
    作用: 绘制类别样本占比饼图
    输入: summary: Dict；output_dir: Path；top_k: int，展示前多少个类别
    输出: Path，生成的图片路径
    """
    rows = summary["category_stats"]
    if not rows:
        labels = ["empty"]
        sizes = [1]
    else:
        top_rows = rows[:top_k]
        other_count = sum(row["sample_count"] for row in rows[top_k:])
        labels = [row["category"] for row in top_rows]
        sizes = [row["sample_count"] for row in top_rows]
        if other_count > 0:
            labels.append("others")
            sizes.append(other_count)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Top {top_k} Category Share")
    fig.tight_layout()

    path = output_dir / f"category_pie_top{top_k}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_scene_hist(summary: Dict, output_dir: Path) -> Path:
    """
    用法: plot_scene_hist(summary, Path("outputs/stats"))
    作用: 绘制每场景样本数直方图
    输入: summary: Dict；output_dir: Path
    输出: Path，生成的图片路径
    """
    values = [row["sample_count"] for row in summary["scene_distribution"]]
    if not values:
        values = [0]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(values, bins=min(20, max(1, len(set(values)))), color="#db8b2b", edgecolor="black")
    ax.set_title("Scene-level Candidate Count Distribution")
    ax.set_xlabel("Candidates per Scene")
    ax.set_ylabel("Number of Scenes")
    fig.tight_layout()

    path = output_dir / "scene_sample_hist.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def export_statistics(summary: Dict, output_dir: Path, top_k: int = 20) -> Dict[str, Path]:
    """
    用法: export_statistics(summary, Path("outputs/stats"), top_k=20)
    作用: 将统计结果导出为 JSON、CSV、TXT 和 PNG 图表
    输入: summary: Dict；output_dir: Path；top_k: int，柱状图展示前多少类
    输出: Dict[str, Path]，各输出文件路径
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pie_top_k = min(10, top_k)
    return {
        "summary_json": write_summary_json(summary, output_dir),
        "category_csv": write_category_csv(summary, output_dir),
        "scene_csv": write_scene_csv(summary, output_dir),
        "summary_txt": write_summary_text(summary, output_dir, top_k=min(10, top_k)),
        "category_bar": plot_category_bar(summary, output_dir, top_k=top_k),
        "category_pie": plot_category_pie(summary, output_dir, top_k=pie_top_k),
        "scene_hist": plot_scene_hist(summary, output_dir),
    }
