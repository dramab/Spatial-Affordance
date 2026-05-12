"""
tests/test_backfill_benchmark_manifest_target_object.py
-------------------------------------------------------
职责：测试旧 benchmark manifest 升级为新版 target_object 格式的 backfill 脚本。

测试内容：
- test_backfill_benchmark_manifest_adds_target_object：
  验证 backfill 会删除 object_center_world 并补齐 target_object.corners_world
- test_backfill_benchmark_manifest_requires_target_object：
  验证 scene_objects 缺少目标物体时明确报错

用法：
    pytest tests/test_backfill_benchmark_manifest_target_object.py -v
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "tools" / "backfill_benchmark_manifest_target_object.py"
SPEC = importlib.util.spec_from_file_location("backfill_benchmark_manifest_target_object", MODULE_PATH)
backfiller = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(backfiller)


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


def _write_old_benchmark(tmp_path: Path) -> tuple[Path, Path]:
    """
    用法: benchmark_dir, outputs_base = _write_old_benchmark(tmp_path)
    作用: 写入旧 benchmark 和对应 scene_objects 测试输入
    输入: tmp_path: pytest 临时目录
    输出: tuple，旧 benchmark 目录和 outputs 根目录
    """
    benchmark_dir = tmp_path / "benchmark/placement_v1"
    outputs_base = tmp_path / "outputs"
    occupancy_rel = "occupancy_grids/hope/scene_0000_0000.npy"
    _write_json(
        benchmark_dir / "manifest.json",
        {
            "schema_version": "placement_benchmark_manifest/v1",
            "split": "test",
            "sample_count": 1,
            "inputs": {"annotation_dir": "data/annotations"},
            "samples": [
                {
                    "sample_id": "scene_0000_0000_obj_0_p000",
                    "source_name": "hope",
                    "scene_id": "scene_0000",
                    "frame_id": "0000",
                    "object_id": "obj_0",
                    "class_name": "Target",
                    "target_box_world": [4.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0],
                    "object_center_world": [1.0, 2.0, 3.0],
                    "occupancy": {"path": occupancy_rel},
                    "direction": {"expected_relation": "the right of"},
                }
            ],
        },
    )
    (benchmark_dir / "occupancy_grids/hope").mkdir(parents=True, exist_ok=True)
    np.save(benchmark_dir / occupancy_rel, np.zeros((2, 2, 2), dtype=np.uint8))
    _write_json(
        outputs_base / "hope/scene_objects/scene_0000_0000.json",
        {
            "schema_version": "placement_scene_objects/v1",
            "objects": [
                {
                    "object_id": "obj_0",
                    "class_name": "Target",
                    "canonical_aabb_object": [-1, -1, -1, 1, 1, 1],
                    "pose_world": np.eye(4, dtype=np.float64).tolist(),
                    "corners_world": [],
                }
            ],
        },
    )
    return benchmark_dir, outputs_base


def test_backfill_benchmark_manifest_adds_target_object(tmp_path, monkeypatch):
    """
    作用：验证 backfill 会写出新版 manifest、删除旧中心字段并复制 occupancy。

    输入：
        tmp_path: pytest 临时目录
        monkeypatch: pytest monkeypatch 工具
    输出：
        无，通过断言验证新版 manifest 内容
    """
    monkeypatch.setattr(backfiller, "PROJECT_ROOT", tmp_path)
    benchmark_dir, outputs_base = _write_old_benchmark(tmp_path)
    output_dir = tmp_path / "benchmark/placement_v2"

    payload = backfiller.backfill_benchmark_manifest(
        benchmark_dir=benchmark_dir,
        outputs_base=outputs_base,
        output_dir=output_dir,
        overwrite=False,
    )

    sample = payload["samples"][0]
    assert payload["schema_version"] == "placement_benchmark_manifest/v2"
    assert "object_center_world" not in sample
    assert sample["target_object"]["object_id"] == "obj_0"
    assert sample["target_object"]["class_name"] == "Target"
    assert len(sample["target_object"]["corners_world"]) == 8
    assert (output_dir / sample["occupancy"]["path"]).exists()
    saved_payload = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert saved_payload["samples"][0]["target_object"]["object_id"] == "obj_0"


def test_backfill_benchmark_manifest_requires_target_object(tmp_path, monkeypatch):
    """
    作用：验证 scene_objects 缺少目标物体时会抛出明确异常。

    输入：
        tmp_path: pytest 临时目录
        monkeypatch: pytest monkeypatch 工具
    输出：
        无，通过断言验证异常
    """
    monkeypatch.setattr(backfiller, "PROJECT_ROOT", tmp_path)
    benchmark_dir, outputs_base = _write_old_benchmark(tmp_path)
    _write_json(
        outputs_base / "hope/scene_objects/scene_0000_0000.json",
        {"schema_version": "placement_scene_objects/v1", "objects": []},
    )

    with pytest.raises(KeyError, match="target object"):
        backfiller.backfill_benchmark_manifest(
            benchmark_dir=benchmark_dir,
            outputs_base=outputs_base,
            output_dir=tmp_path / "benchmark/placement_v2",
            overwrite=False,
        )
