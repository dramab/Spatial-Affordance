#!/usr/bin/env python3
"""
visualize_train_logs.py
-----------------------
解析并可视化多模态训练日志。

用法:
    conda run -n spatial python visualize_train_logs.py
    conda run -n spatial python visualize_train_logs.py -l outputs/multimodal_train/train.log -o out.png
    conda run -n spatial python visualize_train_logs.py -l train.log -l trainv1.log -o compare.png

输出:
    默认保存到 outputs/training_curves.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_epoch_summary_line(line: str) -> dict[str, Any] | None:
    """
    用法: data = parse_epoch_summary_line(line)
    作用: 从 Epoch 汇总行提取指标
    输入: line: str, 日志行
    输出: dict 或 None
    """
    pattern = (
        r"\[Epoch\s+(\d+)/(\d+)\]\s+\|"
        r"\s+train_time=([\d:]+\.?\d*)\s+\|"
        r"\s+train:\s+(.+?)\s+\|"
        r"\s+valid_time=([\d:]+\.?\d*)\s+\|"
        r"\s+valid:\s+(.+?)\s+\|"
        r"\s+epoch_time=([\d:]+\.?\d*)\s+\|"
        r"\s+best_val_loss=([\d.e+-]+)"
    )
    m = re.search(pattern, line)
    if not m:
        return None

    epoch, total_epochs, train_time, train_str, valid_time, valid_str, epoch_time, best_val = m.groups()

    def parse_metrics(s: str) -> dict[str, float]:
        metrics = {}
        for part in s.split(", "):
            if "=" in part:
                k, v = part.split("=", 1)
                metrics[k.strip()] = float(v.strip())
        return metrics

    def time_to_seconds(t: str) -> float:
        """将 HH:MM:SS.s 或 MM:SS.s 转为秒"""
        parts = t.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        else:
            return float(parts[0])

    return {
        "epoch": int(epoch),
        "total_epochs": int(total_epochs),
        "train_time": time_to_seconds(train_time),
        "valid_time": time_to_seconds(valid_time),
        "epoch_time": time_to_seconds(epoch_time),
        "best_val_loss": float(best_val),
        "train_metrics": parse_metrics(train_str),
        "valid_metrics": parse_metrics(valid_str),
    }


def parse_lr_from_log(log_path: Path) -> dict[int, float]:
    """
    用法: lr_map = parse_lr_from_log(path)
    作用: 从日志中提取每个 epoch 最后一步的学习率
    输入: log_path: Path
    输出: dict[epoch, lr]
    """
    lr_map = {}
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            # 匹配训练最后一步的 lr
            m = re.search(r"\[Train\]\s+epoch\s+(\d+)/\d+\s+step\s+\d+/\d+\s+-\s+.*lr=([\d.e+-]+)", line)
            if m:
                epoch = int(m.group(1))
                lr = float(m.group(2))
                lr_map[epoch] = lr
    return lr_map


def parse_log_file(log_path: Path) -> dict[str, Any]:
    """
    用法: data = parse_log_file(path)
    作用: 解析完整日志文件
    输入: log_path: Path
    输出: dict，包含 epochs 列表和标签
    """
    epochs = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            data = parse_epoch_summary_line(line)
            if data:
                epochs.append(data)

    lr_map = parse_lr_from_log(log_path)
    for ep in epochs:
        ep["lr"] = lr_map.get(ep["epoch"], np.nan)

    # 从第一行提取标签信息
    label = log_path.stem
    with log_path.open("r", encoding="utf-8") as f:
        first_lines = [f.readline() for _ in range(10)]
        for line in first_lines:
            if "world_size=" in line:
                m = re.search(r"world_size=(\d+)", line)
                if m:
                    ws = int(m.group(1))
                    label += f" (DDP {ws} GPUs)"
                break
            elif "DataLoader" in line and "train_batch_size=" in line:
                label += " (Single GPU)"
                break

    return {"label": label, "epochs": epochs}


def plot_training_curves(logs: list[dict[str, Any]], output_path: Path) -> None:
    """
    用法: plot_training_curves(logs, Path("out.png"))
    作用: 绘制训练曲线对比图
    输入: logs: list[dict], 各日志解析结果; output_path: Path
    输出: None，保存图片
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Training Log Visualization", fontsize=14, fontweight="bold")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # 1. Total Loss
    ax = axes[0, 0]
    for idx, log in enumerate(logs):
        c = colors[idx % len(colors)]
        epochs = [e["epoch"] for e in log["epochs"]]
        train_loss = [e["train_metrics"]["loss"] for e in log["epochs"]]
        valid_loss = [e["valid_metrics"]["loss"] for e in log["epochs"]]
        ax.plot(epochs, train_loss, "-o", color=c, label=f"{log['label']} train", alpha=0.8)
        ax.plot(epochs, valid_loss, "--s", color=c, label=f"{log['label']} valid", alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2. Center Loss
    ax = axes[0, 1]
    for idx, log in enumerate(logs):
        c = colors[idx % len(colors)]
        epochs = [e["epoch"] for e in log["epochs"]]
        train_v = [e["train_metrics"].get("center_loss", np.nan) for e in log["epochs"]]
        valid_v = [e["valid_metrics"].get("center_loss", np.nan) for e in log["epochs"]]
        ax.plot(epochs, train_v, "-o", color=c, label=f"{log['label']} train", alpha=0.8)
        ax.plot(epochs, valid_v, "--s", color=c, label=f"{log['label']} valid", alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Center Loss")
    ax.set_title("Center Loss")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 3. Size Loss
    ax = axes[0, 2]
    for idx, log in enumerate(logs):
        c = colors[idx % len(colors)]
        epochs = [e["epoch"] for e in log["epochs"]]
        train_v = [e["train_metrics"].get("size_loss", np.nan) for e in log["epochs"]]
        valid_v = [e["valid_metrics"].get("size_loss", np.nan) for e in log["epochs"]]
        ax.plot(epochs, train_v, "-o", color=c, label=f"{log['label']} train", alpha=0.8)
        ax.plot(epochs, valid_v, "--s", color=c, label=f"{log['label']} valid", alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Size Loss")
    ax.set_title("Size Loss")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 4. Yaw Loss
    ax = axes[1, 0]
    for idx, log in enumerate(logs):
        c = colors[idx % len(colors)]
        epochs = [e["epoch"] for e in log["epochs"]]
        train_v = [e["train_metrics"].get("yaw_loss", np.nan) for e in log["epochs"]]
        valid_v = [e["valid_metrics"].get("yaw_loss", np.nan) for e in log["epochs"]]
        ax.plot(epochs, train_v, "-o", color=c, label=f"{log['label']} train", alpha=0.8)
        ax.plot(epochs, valid_v, "--s", color=c, label=f"{log['label']} valid", alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Yaw Loss")
    ax.set_title("Yaw Loss")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 5. Learning Rate
    ax = axes[1, 1]
    for idx, log in enumerate(logs):
        c = colors[idx % len(colors)]
        epochs = [e["epoch"] for e in log["epochs"]]
        lrs = [e["lr"] for e in log["epochs"]]
        ax.plot(epochs, lrs, "-o", color=c, label=log["label"], alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Epoch Time
    ax = axes[1, 2]
    for idx, log in enumerate(logs):
        c = colors[idx % len(colors)]
        epochs = [e["epoch"] for e in log["epochs"]]
        times = [e["epoch_time"] / 60 for e in log["epochs"]]  # 转为分钟
        ax.bar([e + idx * 0.15 for e in epochs], times, width=0.15, color=c, label=log["label"], alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time (minutes)")
    ax.set_title("Epoch Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存: {output_path}")


def print_summary_table(logs: list[dict[str, Any]]) -> None:
    """
    用法: print_summary_table(logs)
    作用: 打印各日志的关键指标汇总
    输入: logs: list[dict]
    输出: None
    """
    print("\n" + "=" * 80)
    print("训练日志汇总")
    print("=" * 80)
    for log in logs:
        print(f"\n{log['label']}:")
        print(f"  {'Epoch':>6} {'Train Loss':>12} {'Valid Loss':>12} {'Best Val':>12} {'Epoch Time':>12} {'LR':>12}")
        print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        for e in log["epochs"]:
            print(
                f"  {e['epoch']:>6} "
                f"{e['train_metrics']['loss']:>12.6f} "
                f"{e['valid_metrics']['loss']:>12.6f} "
                f"{e['best_val_loss']:>12.6f} "
                f"{e['epoch_time']/60:>11.1f}m "
                f"{e['lr']:>12.6f}"
            )


def main() -> None:
    """
    用法: main()
    作用: 命令行入口，解析参数并绘制训练曲线
    输入: 命令行参数（通过 argparse）
    输出: None
    """
    parser = argparse.ArgumentParser(description="可视化训练日志")
    parser.add_argument(
        "-l", "--log",
        type=Path,
        action="append",
        help="日志文件路径，可多次传入以对比多个日志"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("outputs/training_curves.png"),
        help="输出图片路径（默认: outputs/training_curves.png）"
    )
    args = parser.parse_args()

    if args.log:
        log_files = args.log
    else:
        # 未传参数时使用默认路径
        base_dir = Path("outputs/multimodal_train")
        log_files = [base_dir / "train.log", base_dir / "trainv1.log"]

    logs = []
    for path in log_files:
        if path.exists():
            logs.append(parse_log_file(path))
        else:
            print(f"警告: 文件不存在 {path}")

    if not logs:
        print("没有找到可解析的日志文件")
        return

    print_summary_table(logs)
    plot_training_curves(logs, args.output)


if __name__ == "__main__":
    main()
