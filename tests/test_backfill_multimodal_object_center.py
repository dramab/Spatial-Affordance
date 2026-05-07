"""
tests/test_backfill_multimodal_object_center.py
-----------------------------------------------
职责：测试旧多模态 annotation 补全 placement.object_center 的脚本。

测试内容：
- test_backfill_multimodal_object_center_writes_v2_annotation：
  验证脚本会从原始 placement sample 计算 object_center 并写出 v2 annotation
- test_backfill_multimodal_object_center_raises_on_missing_raw_sample：
  验证 annotation 样本找不到原始 placement sample 时会报错
- test_backfill_multimodal_object_center_protects_existing_output_dir：
  验证输出目录已存在时必须显式 overwrite
- test_backfill_multimodal_object_center_parses_simple_dopose_sample_id：
  验证缺少 scene/frame 字段时可从 DoPose 风格 sample_id 定位 raw sample

用法：
    pytest tests/test_backfill_multimodal_object_center.py -v
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "tools" / "backfill_multimodal_object_center.py"
SPEC = importlib.util.spec_from_file_location("backfill_multimodal_object_center", MODULE_PATH)
backfill = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(backfill)


def _write_json(path: Path, payload: dict | list) -> None:
    """
    用法: _write_json(path, payload)
    作用: 为测试写入 JSON 文件
    输入: path: Path；payload: dict | list
    输出: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _make_transform(translation: tuple[float, float, float]) -> list[list[float]]:
    """
    用法: transform = _make_transform((1.0, 2.0, 3.0))
    作用: 构造测试用 object→world 齐次变换
    输入: translation: tuple[float,float,float]
    输出: list[list[float]]，可序列化的 4x4 矩阵
    """
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform.tolist()


def _write_annotation_split(annotation_dir: Path, sample_id: str, source_name: str) -> None:
    """
    用法: _write_annotation_split(annotation_dir, "scene_0000_0000_obj_0_p000", "hope")
    作用: 写入只包含 target_box 的旧版 train annotation
    输入: annotation_dir: Path；sample_id/source_name: str
    输出: None
    """
    _write_json(
        annotation_dir / "train.json",
        {
            "schema_version": "placement_multimodal_dataset/v1",
            "split": "train",
            "sample_count": 1,
            "samples": [
                {
                    "sample_id": sample_id,
                    "source_name": source_name,
                    "placement": {
                        "target_box": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0],
                    },
                }
            ],
        },
    )


def _write_raw_sample(
        source_dir: Path,
        scene_frame: str,
        sample_id: str,
        translation: tuple[float, float, float] = (10.0, 20.0, 30.0)) -> None:
    """
    用法: _write_raw_sample(source_dir, "scene_0000_0000", sample_id)
    作用: 写入包含 canonical_aabb_object 与 original_pose_world 的原始 placement sample
    输入: source_dir: Path；scene_frame/sample_id: str；translation: 原始物体位姿平移
    输出: None
    """
    _write_json(
        source_dir / "samples" / f"{scene_frame}.json",
        {
            "samples": [
                {
                    "sample_id": sample_id,
                    "canonical_aabb_object": [0.0, 0.0, 0.0, 2.0, 4.0, 6.0],
                    "original_pose_world": _make_transform(translation),
                }
            ]
        },
    )


def test_backfill_multimodal_object_center_writes_v2_annotation(tmp_path):
    """
    作用：验证脚本会补齐 placement.object_center 并写出 v2 annotation。

    输入：
        tmp_path: pytest 临时目录
    输出：
        无，通过断言验证结果
    """
    annotation_dir = tmp_path / "data/annotations/placement_multimodal"
    output_dir = tmp_path / "data/annotations/placement_multimodal_v2"
    hope_dir = tmp_path / "outputs/hope"
    sample_id = "scene_0000_0000_obj_0_p000"

    _write_annotation_split(annotation_dir, sample_id, "hope")
    _write_raw_sample(hope_dir, "scene_0000_0000", sample_id)

    summary = backfill.backfill_annotation_dir(
        annotation_dir=annotation_dir,
        output_dir=output_dir,
        source_dirs=[hope_dir],
        splits=["train"],
    )

    train_payload = json.loads((output_dir / "train.json").read_text(encoding="utf-8"))
    source_payload = json.loads((annotation_dir / "train.json").read_text(encoding="utf-8"))
    sample_v2 = train_payload["samples"][0]

    assert summary["total_samples"] == 1
    assert summary["raw_files_read"] == 1
    assert train_payload["schema_version"] == backfill.SCHEMA_VERSION
    assert train_payload["sample_count"] == 1
    assert sample_v2["placement"]["target_box"] == [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0]
    assert np.allclose(sample_v2["placement"]["object_center"], [11.0, 22.0, 33.0], atol=1e-6)
    assert "object_center" not in source_payload["samples"][0]["placement"]
    assert (output_dir / "summary.json").exists()


def test_backfill_multimodal_object_center_raises_on_missing_raw_sample(tmp_path):
    """
    作用：验证 raw samples 文件中缺少对应 sample_id 时会抛出 KeyError。

    输入：
        tmp_path: pytest 临时目录
    输出：
        无，通过断言验证结果
    """
    annotation_dir = tmp_path / "data/annotations/placement_multimodal"
    output_dir = tmp_path / "data/annotations/placement_multimodal_v2"
    hope_dir = tmp_path / "outputs/hope"

    _write_annotation_split(annotation_dir, "scene_0000_0000_obj_1_p000", "hope")
    _write_raw_sample(hope_dir, "scene_0000_0000", "scene_0000_0000_obj_0_p000")

    with pytest.raises(KeyError, match="not found in raw placement file"):
        backfill.backfill_annotation_dir(
            annotation_dir=annotation_dir,
            output_dir=output_dir,
            source_dirs=[hope_dir],
            splits=["train"],
        )


def test_backfill_multimodal_object_center_protects_existing_output_dir(tmp_path):
    """
    作用：验证输出目录已存在且未指定 overwrite 时会被拒绝。

    输入：
        tmp_path: pytest 临时目录
    输出：
        无，通过断言验证结果
    """
    annotation_dir = tmp_path / "data/annotations/placement_multimodal"
    output_dir = tmp_path / "data/annotations/placement_multimodal_v2"
    hope_dir = tmp_path / "outputs/hope"
    sample_id = "scene_0000_0000_obj_0_p000"

    _write_annotation_split(annotation_dir, sample_id, "hope")
    _write_raw_sample(hope_dir, "scene_0000_0000", sample_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileExistsError, match="use --overwrite"):
        backfill.backfill_annotation_dir(
            annotation_dir=annotation_dir,
            output_dir=output_dir,
            source_dirs=[hope_dir],
            splits=["train"],
        )


def test_backfill_multimodal_object_center_parses_simple_dopose_sample_id(tmp_path):
    """
    作用：验证 simple annotation 缺少 scene/frame 字段时，可通过 DoPose 风格 sample_id 定位 raw JSON。

    输入：
        tmp_path: pytest 临时目录
    输出：
        无，通过断言验证结果
    """
    annotation_dir = tmp_path / "data/annotations/placement_multimodal_simple"
    output_dir = tmp_path / "data/annotations/placement_multimodal_simple_v2"
    dopose_dir = tmp_path / "outputs/dopose"
    sample_id = "test_bin_000005_000000_obj_000002_2_p000"

    _write_annotation_split(annotation_dir, sample_id, "dopose")
    _write_raw_sample(
        source_dir=dopose_dir,
        scene_frame="test_bin_000005_000000",
        sample_id=sample_id,
        translation=(1.0, 2.0, 3.0),
    )

    summary = backfill.backfill_annotation_dir(
        annotation_dir=annotation_dir,
        output_dir=output_dir,
        source_dirs=[dopose_dir],
        splits=["train"],
    )
    train_payload = json.loads((output_dir / "train.json").read_text(encoding="utf-8"))

    assert summary["total_samples"] == 1
    assert summary["by_source"]["train"] == {"dopose": 1}
    assert np.allclose(
        train_payload["samples"][0]["placement"]["object_center"],
        [2.0, 4.0, 6.0],
        atol=1e-6,
    )
