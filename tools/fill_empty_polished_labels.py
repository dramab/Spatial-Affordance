"""
fill_empty_polished_labels.py
-----------------------------
补全 JSON 文件中为空的 polished_label 字段。

当 auto-label 生成的 polished_label 为空（None 或空字符串）时，
用对应条目的原始 label 字段进行回退填充，确保下游训练不会读到空文本。

用法:
    python tools/fill_empty_polished_labels.py \
        --input outputs/auto_labels/all_labels_polished.json

输入:
    --input: JSON 文件路径，格式为对象列表，每个对象包含 "label" 和 "polished_label" 字段

输出:
    直接覆盖原文件，并在终端打印修改条数统计。
"""

import argparse
import json
import os


def fill_empty_polished_labels(input_path: str) -> int:
    """
    读取 JSON 文件，将 polished_label 为空的条目用 label 补全，并写回文件。

    参数:
        input_path: JSON 文件路径
    返回:
        被修改的条目数量
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    modified = 0
    for item in data:
        polished = item.get("polished_label")
        if polished is None or polished == "":
            item["polished_label"] = item.get("label", "")
            modified += 1

    # 先写入临时文件，再替换原文件，防止写中断导致数据损坏
    tmp_path = input_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, input_path)

    return modified


def main():
    parser = argparse.ArgumentParser(
        description="补全 JSON 中为空的 polished_label 字段"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/auto_labels/all_labels_polished.json",
        help="输入 JSON 文件路径（默认: outputs/auto_labels/all_labels_polished.json）",
    )
    args = parser.parse_args()

    modified = fill_empty_polished_labels(args.input)
    print(f"处理完成，共修改 {modified} 条记录。")


if __name__ == "__main__":
    main()
