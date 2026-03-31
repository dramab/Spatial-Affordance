#!/usr/bin/env python3
"""
tools/merge_category_stats.py
-----------------------------
按预定义的严格映射规则归并 placement 类别统计，并导出映射表、归并结果和箱形图。

用法:
    python tools/merge_category_stats.py \
        --summary-json outputs/placement_stats_combined/summary.json \
        --output-dir outputs/placement_stats_combined

作用:
    - 读取现有 summary.json 中的 category_stats
    - 按严格版命名归并规则合并类别
    - 导出原类别到归并类别的映射表
    - 导出归并后的类别样本数 JSON / CSV
    - 生成归并后类别样本数分布的箱形图

输入:
    --summary-json: placement 统计结果中的 summary.json 路径
    --output-dir: 输出目录

输出:
    在输出目录下生成:
        - merged_category_mapping.json
        - merged_category_counts.json
        - merged_category_counts.csv
        - merged_category_boxplot.png

使用示例:
    python tools/merge_category_stats.py \
        --summary-json outputs/placement_stats_combined/summary.json \
        --output-dir outputs/placement_stats_combined
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / "outputs" / ".matplotlib_cache"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CONFIG_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STRICT_CATEGORY_MAPPING: Dict[str, str] = {
    "glass-small": "glass",
    "glass-small_3": "glass",
    "glass-small_4": "glass",
    "glass-new_1": "glass",
    "glass-new_6": "glass",
    "glass-new_7": "glass",
    "glass-new_8": "glass",
    "glass-new_12": "glass",
    "cutlery-spoon_1": "cutlery-spoon",
    "cutlery-spoon_2": "cutlery-spoon",
    "cutlery-spoon_4_new": "cutlery-spoon",
    "cutlery-spoon_5_new": "cutlery-spoon",
    "cutlery-spoon_6_new": "cutlery-spoon",
    "cutlery-fork_1": "cutlery-fork",
    "cutlery-fork_1_new": "cutlery-fork",
    "cutlery-knife_1": "cutlery-knife",
    "cutlery-knife_2": "cutlery-knife",
    "remote-black": "remote",
    "remote-grey": "remote",
    "remote-silver": "remote",
    "remote-thin_silver": "remote",
    "remote-led_2": "remote",
    "remote-comfee": "remote",
    "remote-infini_fun": "remote",
    "remote-jaxster": "remote",
    "remote-jaxster_d1170": "remote",
    "remote-factory_svc": "remote",
    "remote-tv_white_quelle": "remote",
    "teapot-white_rectangle": "teapot",
    "teapot-white_rectangle_sprout": "teapot",
    "teapot-white_was_brand": "teapot",
    "teapot-ambition_brand": "teapot",
    "teapot-wooden_color": "teapot",
    "teapot-new_chinese": "teapot",
    "cup-green_actys": "cup",
    "cup-yellow_handle": "cup",
    "cup-red_heart": "cup",
    "cup-white_whisker": "cup",
    "cup-new_york_big": "cup",
    "cup-stanford": "cup",
    "cup-white_coffee_round_handle": "cup",
    "cup-mc_cafe": "cup",
    "cup-yellow_white_border": "cup",
    "bottle-evian_red": "bottle-evian",
    "bottle-evian_frozen": "bottle-evian",
    "tube-signal": "tube-signal",
    "tube-toothpaste_signal_kids_bio": "tube-signal",
}


def build_parser() -> argparse.ArgumentParser:
    """
    用法: parser = build_parser()
    作用: 构建命令行参数解析器
    输入: 无
    输出: argparse.ArgumentParser，配置完成的解析器
    """
    parser = argparse.ArgumentParser(description="归并 placement 类别统计并导出结果")
    parser.add_argument(
        "--summary-json",
        required=True,
        type=Path,
        help="原始统计 summary.json 路径",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="归并结果输出目录",
    )
    return parser


def load_summary(summary_json: Path) -> Dict:
    """
    用法: summary = load_summary(Path("outputs/placement_stats_combined/summary.json"))
    作用: 读取原始 placement 统计结果
    输入: summary_json: Path，summary.json 文件路径
    输出: Dict，解析后的汇总数据
    """
    with summary_json.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_full_mapping(category_stats: List[Dict]) -> Dict[str, str]:
    """
    用法: mapping = build_full_mapping(category_stats)
    作用: 为所有原始类别补齐完整映射，未归并的类别保持映射到自身
    输入: category_stats: List[Dict]，summary.json 中的 category_stats
    输出: Dict[str, str]，原类别到归并后类别的完整映射表
    """
    full_mapping: Dict[str, str] = {}
    for row in category_stats:
        category = str(row["category"])
        full_mapping[category] = STRICT_CATEGORY_MAPPING.get(category, category)
    return dict(sorted(full_mapping.items(), key=lambda item: item[0]))


def merge_category_counts(category_stats: List[Dict], mapping: Dict[str, str]) -> List[Dict]:
    """
    用法: merged_rows = merge_category_counts(category_stats, mapping)
    作用: 按映射表归并类别样本数、物体数和场景数
    输入:
        category_stats: List[Dict]，原始类别统计
        mapping: Dict[str, str]，原类别到归并类别的映射表
    输出: List[Dict]，归并后的类别统计列表
    """
    merged_sample_counts: Dict[str, int] = defaultdict(int)
    merged_object_counts: Dict[str, int] = defaultdict(int)
    merged_scene_counts: Dict[str, int] = defaultdict(int)

    for row in category_stats:
        source_category = str(row["category"])
        merged_category = mapping[source_category]
        merged_sample_counts[merged_category] += int(row.get("sample_count", 0))
        merged_object_counts[merged_category] += int(row.get("object_count", 0))
        merged_scene_counts[merged_category] += int(row.get("scene_count", 0))

    total_samples = sum(merged_sample_counts.values())
    merged_rows: List[Dict] = []
    for category, sample_count in merged_sample_counts.items():
        merged_rows.append(
            {
                "category": category,
                "sample_count": sample_count,
                "object_count": merged_object_counts[category],
                "scene_count": merged_scene_counts[category],
                "ratio": (sample_count / total_samples) if total_samples else 0.0,
            }
        )

    return sorted(merged_rows, key=lambda row: (-row["sample_count"], row["category"]))


def write_mapping_json(mapping: Dict[str, str], output_dir: Path) -> Path:
    """
    用法: path = write_mapping_json(mapping, Path("outputs/placement_stats_combined"))
    作用: 导出完整类别映射 JSON
    输入: mapping: Dict[str, str]；output_dir: Path
    输出: Path，生成的 JSON 路径
    """
    path = output_dir / "merged_category_mapping.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mapping": mapping,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def write_merged_counts_json(merged_rows: List[Dict], output_dir: Path) -> Path:
    """
    用法: path = write_merged_counts_json(merged_rows, Path("outputs/placement_stats_combined"))
    作用: 导出归并后的类别统计 JSON
    输入: merged_rows: List[Dict]；output_dir: Path
    输出: Path，生成的 JSON 路径
    """
    path = output_dir / "merged_category_counts.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "merged_category_count": len(merged_rows),
        "total_samples": sum(row["sample_count"] for row in merged_rows),
        "categories": merged_rows,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def write_merged_counts_csv(merged_rows: List[Dict], output_dir: Path) -> Path:
    """
    用法: path = write_merged_counts_csv(merged_rows, Path("outputs/placement_stats_combined"))
    作用: 导出归并后的类别统计 CSV
    输入: merged_rows: List[Dict]；output_dir: Path
    输出: Path，生成的 CSV 路径
    """
    path = output_dir / "merged_category_counts.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["category", "sample_count", "object_count", "scene_count", "ratio"],
        )
        writer.writeheader()
        for row in merged_rows:
            writer.writerow(row)
    return path


def write_boxplot(merged_rows: List[Dict], output_dir: Path) -> Path:
    """
    用法: path = write_boxplot(merged_rows, Path("outputs/placement_stats_combined"))
    作用: 绘制归并后类别样本数分布的箱形图
    输入: merged_rows: List[Dict]；output_dir: Path
    输出: Path，生成的 PNG 路径
    """
    path = output_dir / "merged_category_boxplot.png"
    values = [row["sample_count"] for row in merged_rows]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.boxplot(
        values,
        patch_artist=True,
        boxprops={"facecolor": "#7fb3d5", "edgecolor": "#1f618d"},
        medianprops={"color": "#922b21", "linewidth": 2},
        whiskerprops={"color": "#1f618d"},
        capprops={"color": "#1f618d"},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#cd6155",
            "markeredgecolor": "#922b21",
            "markersize": 5,
            "alpha": 0.8,
        },
    )
    ax.set_title("Merged Category Sample Count Distribution")
    ax.set_ylabel("Sample Count")
    ax.set_xticks([1])
    ax.set_xticklabels(["Merged Categories"])
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    """
    用法: main()
    作用: 执行类别归并导出 CLI 主流程
    输入: 无，参数来自命令行
    输出: None，在终端打印结果并写出文件
    """
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(args.summary_json)
    category_stats = summary.get("category_stats", [])
    mapping = build_full_mapping(category_stats)
    merged_rows = merge_category_counts(category_stats, mapping)

    mapping_path = write_mapping_json(mapping, args.output_dir)
    merged_json_path = write_merged_counts_json(merged_rows, args.output_dir)
    merged_csv_path = write_merged_counts_csv(merged_rows, args.output_dir)
    boxplot_path = write_boxplot(merged_rows, args.output_dir)

    print("类别归并完成")
    print(f"原始类别数: {len(category_stats)}")
    print(f"归并后类别数: {len(merged_rows)}")
    print(f"总样本数: {sum(row['sample_count'] for row in merged_rows)}")
    print("生成文件:")
    print(f"- mapping_json: {mapping_path}")
    print(f"- merged_counts_json: {merged_json_path}")
    print(f"- merged_counts_csv: {merged_csv_path}")
    print(f"- boxplot_png: {boxplot_path}")


if __name__ == "__main__":
    main()
