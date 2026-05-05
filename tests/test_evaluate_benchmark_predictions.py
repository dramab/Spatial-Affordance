"""
tests/test_evaluate_benchmark_predictions.py
--------------------------------------------
职责：测试只依赖 benchmark 包的 prediction 评测脚本。

测试内容：
- test_evaluate_benchmark_predictions_uses_only_manifest：
  验证 evaluator 只用 manifest、occupancy 和 predictions 完成三类 metric

用法：
    pytest tests/test_evaluate_benchmark_predictions.py -v
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "tools" / "evaluate_benchmark_predictions.py"
SPEC = importlib.util.spec_from_file_location("evaluate_benchmark_predictions", MODULE_PATH)
evaluator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluator)


def _write_json(path: Path, payload: dict | list) -> None:
    """
    用法: _write_json(path, payload)
    作用: 写入测试 JSON
    输入: path: Path；payload: dict | list
    输出: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def test_evaluate_benchmark_predictions_uses_only_manifest(tmp_path, monkeypatch):
    """
    作用：验证 benchmark evaluator 不需要 annotation、outputs-base、mapping 或原始数据集。

    输入：
        tmp_path: pytest 临时目录
        monkeypatch: pytest monkeypatch 工具
    输出：
        无，通过断言验证评测结果
    """
    monkeypatch.setattr(evaluator, "PROJECT_ROOT", tmp_path)
    benchmark_dir = tmp_path / "benchmark/placement_v1"
    predictions_path = tmp_path / "outputs/infer/predictions.json"
    output_dir = tmp_path / "outputs/eval"
    occupancy_rel = "occupancy_grids/hope/scene_0000_0000.npy"
    (benchmark_dir / "occupancy_grids/hope").mkdir(parents=True, exist_ok=True)
    np.save(benchmark_dir / occupancy_rel, np.zeros((8, 8, 8), dtype=np.uint8))
    camera = {
        "fx": 100.0,
        "fy": 100.0,
        "cx": 0.0,
        "cy": 0.0,
        "img_w": 64,
        "img_h": 48,
        "E_c2w": np.eye(4, dtype=np.float64).tolist(),
    }
    _write_json(
        benchmark_dir / "manifest.json",
        {
            "schema_version": "placement_benchmark_manifest/v1",
            "split": "test",
            "sample_count": 1,
            "samples": [
                {
                    "sample_id": "sample_0",
                    "source_name": "hope",
                    "scene_id": "scene_0000",
                    "frame_id": "0000",
                    "object_id": "obj_0",
                    "target_box_world": [4.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0],
                    "camera": camera,
                    "occupancy": {
                        "path": occupancy_rel,
                        "voxel_params": {"origin": [0.0, 0.0, 0.0], "voxel_size": 1.0},
                        "grid_shape": [8, 8, 8],
                    },
                    "direction": {
                        "expected_relation": "the right of",
                        "reference_object_id": "obj_1",
                        "reference_class_name": "Reference",
                        "reference_name": "Reference",
                        "reference_corners_world": [
                            [-1, -1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, 1],
                            [-1, -1, 3], [-1, 1, 3], [1, -1, 3], [1, 1, 3],
                        ],
                    },
                }
            ],
        },
    )
    _write_json(
        predictions_path,
        {
            "split": "test",
            "predictions": [
                {
                    "sample_id": "sample_0",
                    "source_name": "hope",
                    "pred_box_world": [4.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0],
                }
            ],
        },
    )
    args = argparse.Namespace(
        benchmark_dir=benchmark_dir,
        predictions=predictions_path,
        output_dir=output_dir,
        sample_ids=None,
        limit=None,
        progress_interval=0,
        collision_ratio_threshold=0.01,
        support_ignore_layers=2,
        volume_error_ratio_threshold=0.05,
        direction_center_l2_threshold_cm=1.0,
        write_csv=True,
    )

    summary = evaluator.run_evaluation(args)

    per_sample = json.loads((output_dir / "per_sample_metrics.json").read_text(encoding="utf-8"))["samples"][0]
    assert summary["sample_count"] == 1
    assert per_sample["collision"]["evaluated"] is True
    assert per_sample["collision"]["collision_free"] is True
    assert per_sample["collision"]["support_ignore_layers"] == 2
    assert per_sample["size"]["size_consistent"] is True
    assert per_sample["size"]["volume_error_ratio_threshold"] == 0.05
    assert per_sample["size"]["volume_error_ratio"] == 0.0
    assert per_sample["direction"]["direction_correct"] is True
    assert per_sample["direction"]["center_match"] is True
    assert per_sample["status"]["placement_success"] is True
    assert (output_dir / "per_sample_metrics.csv").exists()
