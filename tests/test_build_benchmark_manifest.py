"""
tests/test_build_benchmark_manifest.py
--------------------------------------
职责：测试 benchmark manifest 构建脚本。

测试内容：
- test_build_benchmark_manifest_writes_self_contained_metric_inputs：
  验证 manifest 固化 target、camera、occupancy 和 reference corners
- test_build_benchmark_manifest_requires_reference_object：
  验证缺少 reference object 时明确报错

用法：
    pytest tests/test_build_benchmark_manifest.py -v
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "tools" / "build_benchmark_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_benchmark_manifest", MODULE_PATH)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(builder)


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


def _make_camera() -> dict:
    """
    用法: camera = _make_camera()
    作用: 构造测试用相机字段
    输入: 无
    输出: dict
    """
    return {
        "fx": 100.0,
        "fy": 100.0,
        "cx": 0.0,
        "cy": 0.0,
        "img_w": 64,
        "img_h": 48,
        "E_c2w": np.eye(4, dtype=np.float64).tolist(),
    }


def _write_minimal_inputs(tmp_path: Path, reference_id: str = "obj_1") -> tuple[Path, Path, Path]:
    """
    用法: annotation_dir, label_json, outputs_base = _write_minimal_inputs(tmp_path)
    作用: 写入构建 benchmark 所需的最小输入文件
    输入: tmp_path: Path；reference_id: str
    输出: tuple，annotation_dir、label_json、outputs_base
    """
    annotation_dir = tmp_path / "data/annotations"
    label_json = tmp_path / "outputs/prompt_merged/all_labels.json"
    outputs_base = tmp_path / "outputs"
    camera = _make_camera()
    _write_json(
        annotation_dir / "test.json",
        {
            "samples": [
                {
                    "sample_id": "scene_0000_0000_obj_0_p000",
                    "source_name": "hope",
                    "scene_id": "scene_0000",
                    "frame_id": "0000",
                    "object_id": "obj_0",
                    "class_name": "Target",
                    "rgb_path": "outputs/rgb.png",
                    "point_cloud_path": "outputs/pc.ply",
                    "prompt": "Move Target to the right of Reference.",
                    "polished_prompt": "",
                    "placement": {
                        "target_box": [4.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0],
                        "object_center": [1.0, 2.0, 3.0],
                    },
                    "camera": camera,
                }
            ]
        },
    )
    _write_json(
        label_json,
        [
            {
                "sample_id": "scene_0000_0000_obj_0_p000",
                "source_name": "hope",
                "label": "Move Target to the right of Reference.",
                "spatial_relation": {
                    "placement": {
                        "relation": "the right of",
                        "reference_object_id": reference_id,
                        "reference_class_name": "Reference",
                        "reference_name": "Reference",
                    }
                },
            }
        ],
    )
    _write_json(
        outputs_base / "hope/grid_meta/scene_0000_0000.json",
        {
            "voxel_params": {"origin": [0.0, 0.0, 0.0], "voxel_size": 1.0},
            "grid_shape": [8, 8, 8],
            "camera": camera,
        },
    )
    _write_json(
        outputs_base / "hope/scene_objects/scene_0000_0000.json",
        {
            "schema_version": "placement_scene_objects/v1",
            "scene_id": "scene_0000",
            "frame_id": "0000",
            "unit": "cm",
            "objects": [
                {
                    "object_id": "obj_0",
                    "class_name": "Target",
                    "canonical_aabb_object": [-1, -1, -1, 1, 1, 1],
                    "pose_world": np.eye(4, dtype=np.float64).tolist(),
                    "corners_world": [],
                    "aabb_world": [],
                },
                {
                    "object_id": "obj_1",
                    "class_name": "Reference",
                    "canonical_aabb_object": [-1, -1, -1, 1, 1, 1],
                    "pose_world": np.eye(4, dtype=np.float64).tolist(),
                    "corners_world": [[-1, -1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, 1], [-1, -1, 3], [-1, 1, 3], [1, -1, 3], [1, 1, 3]],
                    "aabb_world": [-1, -1, 1, 1, 1, 3],
                },
            ],
        },
    )
    grid_path = outputs_base / "hope/occupancy_grids/scene_0000_0000.npy"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(grid_path, np.zeros((8, 8, 8), dtype=np.uint8))
    return annotation_dir, label_json, outputs_base


def test_build_benchmark_manifest_writes_self_contained_metric_inputs(tmp_path, monkeypatch):
    """
    作用：验证 manifest 包含后续 metric 所需字段并复制 occupancy grid。

    输入：
        tmp_path: pytest 临时目录
        monkeypatch: pytest monkeypatch 工具
    输出：
        无，通过断言验证结果
    """
    monkeypatch.setattr(builder, "PROJECT_ROOT", tmp_path)
    annotation_dir, label_json, outputs_base = _write_minimal_inputs(tmp_path)
    output_dir = tmp_path / "benchmark/placement_v1"

    payload = builder.build_benchmark_manifest(
        annotation_dir=annotation_dir,
        label_json=label_json,
        outputs_base=outputs_base,
        output_dir=output_dir,
        split="test",
        overwrite=False,
    )

    sample = payload["samples"][0]
    assert payload["sample_count"] == 1
    assert sample["target_box_world"] == [4.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0]
    assert sample["object_center_world"] == [1.0, 2.0, 3.0]
    assert sample["direction"]["expected_relation"] == "the right of"
    assert sample["direction"]["reference_object_id"] == "obj_1"
    assert len(sample["direction"]["reference_corners_world"]) == 8
    assert (output_dir / sample["occupancy"]["path"]).exists()
    assert json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))["sample_count"] == 1


def test_build_benchmark_manifest_requires_reference_object(tmp_path, monkeypatch):
    """
    作用：验证 spatial_relation 指向不存在 reference object 时会报错。

    输入：
        tmp_path: pytest 临时目录
        monkeypatch: pytest monkeypatch 工具
    输出：
        无，通过断言验证异常
    """
    monkeypatch.setattr(builder, "PROJECT_ROOT", tmp_path)
    annotation_dir, label_json, outputs_base = _write_minimal_inputs(tmp_path, reference_id="missing_obj")

    with pytest.raises(KeyError, match="reference object"):
        builder.build_benchmark_manifest(
            annotation_dir=annotation_dir,
            label_json=label_json,
            outputs_base=outputs_base,
            output_dir=tmp_path / "benchmark/placement_v1",
            split="test",
            overwrite=False,
        )


def test_parse_sample_identity_supports_simple_annotation_sample_ids():
    """
    作用：验证 simple annotation 缺少 scene/frame/object 字段时可从 sample_id 解析。

    输入：
        无，内部构造 HOPE 与 DoPose 风格 sample_id
    输出：
        无，通过断言验证解析结果
    """
    assert builder.parse_sample_identity(
        {"sample_id": "scene_0000_0155_obj_7_p000"}
    ) == ("scene_0000", "0155", "obj_7")
    assert builder.parse_sample_identity(
        {"sample_id": "test_bin_000005_000000_obj_000002_2_p000"}
    ) == ("test_bin_000005", "000000", "obj_000002_2")
