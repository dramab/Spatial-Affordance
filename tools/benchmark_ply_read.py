#!/usr/bin/env python3
"""
benchmark_ply_read.py
---------------------
测试 ASCII PLY 点云文件读取耗时

用法:
    conda run -n spatial python benchmark_ply_read.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.multimodal_dataset import (
    PlacementMultimodalDataset,
    _read_ascii_ply_points,
)


def benchmark_single_ply(ply_path: Path, num_runs: int = 10):
    """测试单个 PLY 文件读取耗时"""
    print(f"测试文件: {ply_path}")
    print(f"文件大小: {ply_path.stat().st_size / 1024 / 1024:.2f} MB")

    # 先读一次看总点数
    points = _read_ascii_ply_points(ply_path)
    print(f"点云数量: {points.shape[0]:,}")
    print()

    # 基准测试
    times = []
    for i in range(num_runs):
        t0 = time.perf_counter()
        _ = _read_ascii_ply_points(ply_path)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)
    print(f"重复读取 {num_runs} 次:")
    print(f"  平均: {avg*1000:.1f} ms")
    print(f"  最快: {min_t*1000:.1f} ms")
    print(f"  最慢: {max_t*1000:.1f} ms")
    print()

    # 换算每千点耗时
    ms_per_k = avg * 1000 / (points.shape[0] / 1000)
    print(f"每千点耗时: {ms_per_k:.2f} ms")
    print()


def benchmark_dataset_iteration():
    """测试 Dataset __getitem__ 整体耗时"""
    dataset = PlacementMultimodalDataset(
        annotation_dir="data/annotations/placement_multimodal",
        split="train",
        prompt_key="prompt",
        image_size=(480, 640),
    )
    print(f"数据集样本数: {len(dataset)}")
    print()

    # 测试单样本读取（含图片+点云+文本）
    num_samples = min(10, len(dataset))
    times = []
    for i in range(num_samples):
        t0 = time.perf_counter()
        _ = dataset[i]
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = sum(times) / len(times)
    print(f"Dataset.__getitem__ 单样本耗时 (含图+点云+文本):")
    print(f"  平均: {avg*1000:.1f} ms")
    print(f"  最快: {min(times)*1000:.1f} ms")
    print(f"  最慢: {max(times)*1000:.1f} ms")
    print()

    # 单独测试点云读取
    sample = dataset.samples[0]
    ply_path = PROJECT_ROOT / sample["point_cloud_path"]
    print("--- 单独测试点云读取 ---")
    benchmark_single_ply(ply_path, num_runs=10)


def benchmark_dataloader_workers():
    """测试不同 num_workers 下的 DataLoader 速度"""
    from torch.utils.data import DataLoader
    from src.datasets.multimodal_dataset import placement_multimodal_collate_fn

    dataset = PlacementMultimodalDataset(
        annotation_dir="data/annotations/placement_multimodal",
        split="train",
        prompt_key="prompt",
        image_size=(480, 640),
    )

    batch_size = 4
    num_batches = 5

    for num_workers in [0, 2, 4]:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            collate_fn=placement_multimodal_collate_fn,
        )

        # warmup
        for i, batch in enumerate(loader):
            if i >= 1:
                break

        # 正式测试
        times = []
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            t0 = time.perf_counter()
            _ = batch
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg = sum(times) / len(times)
        throughput = batch_size * len(times) / sum(times)
        print(f"num_workers={num_workers}: batch 平均 {avg*1000:.1f} ms, 吞吐 {throughput:.1f} samples/s")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("PLY 点云读取基准测试")
    print("=" * 60)
    print()

    benchmark_dataset_iteration()

    print("=" * 60)
    print("DataLoader 不同 worker 数对比")
    print("=" * 60)
    print()
    benchmark_dataloader_workers()
