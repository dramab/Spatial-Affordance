#!/usr/bin/env python3
"""
tools/placement_stats.py
------------------------
统计一个或多个 placement 输出目录中的 AABB 候选框样本，并生成汇总文件和图表。

用法:
    python tools/placement_stats.py \
        --inputs outputs/housecat6d_placement10 outputs/placement_hope5 \
        --output-dir outputs/placement_stats_combined

作用:
    - 统计总场景数、总类别数、总样本数
    - 统计每个类别的候选框样本数、物体数、覆盖场景数
    - 导出 JSON / CSV / TXT
    - 生成柱状图、饼图和场景分布直方图

输入:
    --inputs: 一个或多个 placement 输出根目录
    --output-dir: 统计结果输出目录
    --top-k: 柱状图展示的前 K 个类别，默认 20

输出:
    在输出目录下生成:
        - summary.json
        - category_counts.csv
        - scene_distribution.csv
        - summary.txt
        - category_bar_topK.png
        - category_pie_top10.png
        - scene_sample_hist.png

使用示例:
    python tools/placement_stats.py \
        --inputs outputs/housecat6d_placement10 outputs/placement_hope5 \
        --output-dir outputs/placement_stats_combined \
        --top-k 20
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.placement_stats import aggregate_statistics, export_statistics


def build_parser() -> argparse.ArgumentParser:
    """
    用法: parser = build_parser()
    作用: 构建命令行参数解析器
    输入: 无
    输出: argparse.ArgumentParser，配置完成的解析器
    """
    parser = argparse.ArgumentParser(description="统计 placement 输出目录中的候选框样本")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        type=Path,
        help="一个或多个 placement 输出根目录",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="统计结果输出目录",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="柱状图展示前 K 个类别，默认 20",
    )
    return parser


def format_top_categories(summary: dict, top_k: int = 10) -> str:
    """
    用法: format_top_categories(summary, top_k=10)
    作用: 生成终端打印用的 Top 类别摘要
    输入: summary: dict，统计结果；top_k: int，展示条数
    输出: str，多行摘要文本
    """
    rows = summary["category_stats"][:top_k]
    if not rows:
        return "没有可用的类别统计结果。"

    lines = []
    for idx, row in enumerate(rows, start=1):
        lines.append(
            f"{idx:>2}. {row['category']}: "
            f"samples={row['sample_count']}, "
            f"objects={row['object_count']}, "
            f"scenes={row['scene_count']}, "
            f"ratio={row['ratio']:.2%}"
        )
    return "\n".join(lines)


def main() -> None:
    """
    用法: main()
    作用: 执行 placement 统计 CLI 主流程
    输入: 无，参数来自命令行
    输出: None，在终端打印摘要并写出统计文件
    """
    args = build_parser().parse_args()
    summary = aggregate_statistics(args.inputs)
    outputs = export_statistics(summary, args.output_dir, top_k=args.top_k)

    print("Placement 统计完成")
    print(f"输出目录: {args.output_dir}")
    print(f"总场景数: {summary['total_scenes']}")
    print(f"总类别数: {summary['total_categories']}")
    print(f"总样本数: {summary['total_samples']}")
    print(f"样本文件数: {summary['source_file_count']}")
    print(f"平均每场景样本数: {summary['avg_samples_per_scene']:.2f}")
    print("")
    print("Top 类别:")
    print(format_top_categories(summary, top_k=min(10, args.top_k)))
    print("")
    print("生成文件:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
