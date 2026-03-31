#!/usr/bin/env python3
"""根据每个 samples json 统计一个目录总样本数。"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List
import sys


def collect_sample_files(directory: Path) -> List[Path]:
    """
    用法: collect_sample_files(Path('...'))
    作用: 收集目录下 samples 子目录中的所有 json 文件
    输入: directory: Path，目标输出目录
    输出: List[Path]，样本 json 文件列表
    """
    samples_dir = directory / "samples"
    if not samples_dir.is_dir():
        return []
    return list(samples_dir.glob("*.json"))


def count_samples_in_file(sample_file: Path) -> int:
    """
    用法: count_samples_in_file(Path('.../scene01_000000.json'))
    作用: 返回单文件中 samples 数量
    输入: sample_file: Path
    输出: int，samples 列表长度
    """
    try:
        with sample_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as err:
        print(f"跳过非 JSON 文件 {sample_file}: {err}", file=sys.stderr)
        return 0
    samples = payload.get("samples", [])
    return len(samples)


def count_samples_in_directory(directory: Path) -> int:
    """
    用法: count_samples_in_directory(Path('.../housecat6d_placement10'))
    作用: 统计给定目录下所有 samples 文件的总框数
    输入: directory: Path
    输出: int，总样本数
    """
    files = collect_sample_files(directory)
    return sum(count_samples_in_file(file_path) for file_path in files)


def main(directories: Iterable[Path]) -> None:
    """
    用法: main([Path('dir1'), Path('dir2')])
    作用: 输出每个目录及总体的样本计数
    输入: directories: Iterable[Path]
    输出: None，打印结果
    """
    totals = {}
    for directory in directories:
        totals[directory] = count_samples_in_directory(directory)
        print("{} -> {} samples".format(directory, totals[directory]))
    overall = sum(totals.values())
    print("总样本数: {}".format(overall))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计 placement 输出的样本数")
    parser.add_argument(
        "dirs",
        nargs="+",
        type=Path,
        help="需要统计的 placement 输出目录",
    )
    args = parser.parse_args()
    main(args.dirs)
