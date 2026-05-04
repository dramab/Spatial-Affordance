"""
tests/test_backfill_placement_scene_objects.py
----------------------------------------------
职责：测试旧 placement 输出 scene_objects 补全脚本的核心行为。

测试内容：
- test_backfill_source_dir_writes_all_scene_objects：
  验证补全结果包含 adapter 返回的全部物体
- test_backfill_source_dir_skips_existing_by_default：
  验证默认不会覆盖已有 scene_objects 文件

用法：
    pytest tests/test_backfill_placement_scene_objects.py -v
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "tools" / "backfill_placement_scene_objects.py"
SPEC = importlib.util.spec_from_file_location("backfill_placement_scene_objects", MODULE_PATH)
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


def _read_json(path: Path) -> dict:
    """
    用法: payload = _read_json(path)
    作用: 为测试读取 JSON 文件
    输入: path: Path
    输出: dict，解析后的 JSON 对象
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_minimal_source(source_dir: Path) -> None:
    """
    用法: _write_minimal_source(source_dir)
    作用: 写入包含 scene/frame 的最小旧 placement 输出
    输入: source_dir: Path
    输出: None
    """
    _write_json(
        source_dir / "samples/scene_0000_0000.json",
        {
            "scene_id": "scene_0000",
            "frame_id": "0000",
            "samples": [
                {
                    "sample_id": "scene_0000_0000_obj_0_p000",
                    "scene_id": "scene_0000",
                    "frame_id": "0000",
                    "object_id": "obj_0",
                }
            ],
        },
    )
    _write_json(source_dir / "grid_meta/scene_0000_0000.json", {})


class _FakeAdapter:
    """
    作用：记录 load_scene 调用并返回包含两个物体的假 SceneData。
    """

    def __init__(self):
        """
        用法: adapter = _FakeAdapter()
        作用: 初始化测试 adapter
        输入: 无
        输出: None
        """
        self.calls = []

    def load_scene(self, scene_path: str, frame_id: str):
        """
        用法: scene = adapter.load_scene(scene_path, frame_id)
        作用: 返回带两个物体的假 SceneData
        输入: scene_path: str；frame_id: str
        输出: SimpleNamespace，包含 scene objects
        """
        self.calls.append((scene_path, frame_id))
        objects = [
            SimpleNamespace(
                obj_id="obj_0",
                class_name="Target",
                bbox3d_canonical=np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float64),
                pose_world=np.eye(4, dtype=np.float64),
            ),
            SimpleNamespace(
                obj_id="obj_1",
                class_name="ReferenceOnly",
                bbox3d_canonical=np.array([-2.0, -1.0, -1.0, 2.0, 1.0, 1.0], dtype=np.float64),
                pose_world=np.array(
                    [
                        [1.0, 0.0, 0.0, 5.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                ),
            ),
        ]
        return SimpleNamespace(
            scene_id="scene_0000",
            frame_id="0000",
            unit="cm",
            objects=objects,
        )


def test_backfill_source_dir_writes_all_scene_objects(tmp_path, monkeypatch):
    """
    作用：验证补全文件包含当前帧所有物体，而不是仅包含 samples 中的目标物体。

    输入：
        tmp_path: pytest 临时目录
        monkeypatch: pytest monkeypatch 工具
    输出：
        无，通过断言验证结果
    """
    source_dir = tmp_path / "outputs/hope"
    dataset_root = tmp_path / "dataset"
    adapter = _FakeAdapter()
    _write_minimal_source(source_dir)

    monkeypatch.setattr(backfill, "infer_config_path", lambda _source_dir: tmp_path / "config.yaml")
    monkeypatch.setattr(
        backfill,
        "load_config",
        lambda _config_path: {"dataset": {"root_dir": str(dataset_root)}},
    )
    monkeypatch.setattr(backfill, "build_adapter", lambda _config: adapter)

    summary = backfill.backfill_source_dir(source_dir, overwrite=False, dry_run=False)

    output_path = source_dir / "scene_objects/scene_0000_0000.json"
    payload = _read_json(output_path)
    assert summary["total_frames"] == 1
    assert summary["updated"] == 1
    assert summary["failed"] == 0
    assert [item["object_id"] for item in payload["objects"]] == ["obj_0", "obj_1"]
    assert payload["objects"][1]["class_name"] == "ReferenceOnly"
    assert len(payload["objects"][1]["corners_world"]) == 8
    assert adapter.calls == [(str(dataset_root / "scene_0000"), "0000")]


def test_backfill_source_dir_skips_existing_by_default(tmp_path, monkeypatch):
    """
    作用：验证默认不会覆盖已有 scene_objects 文件。

    输入：
        tmp_path: pytest 临时目录
        monkeypatch: pytest monkeypatch 工具
    输出：
        无，通过断言验证结果
    """
    source_dir = tmp_path / "outputs/hope"
    adapter = _FakeAdapter()
    existing_payload = {"schema_version": "placement_scene_objects/v1", "objects": []}
    _write_minimal_source(source_dir)
    _write_json(source_dir / "scene_objects/scene_0000_0000.json", existing_payload)

    monkeypatch.setattr(backfill, "infer_config_path", lambda _source_dir: tmp_path / "config.yaml")
    monkeypatch.setattr(
        backfill,
        "load_config",
        lambda _config_path: {"dataset": {"root_dir": str(tmp_path / "dataset")}},
    )
    monkeypatch.setattr(backfill, "build_adapter", lambda _config: adapter)

    summary = backfill.backfill_source_dir(source_dir, overwrite=False, dry_run=False)

    payload = _read_json(source_dir / "scene_objects/scene_0000_0000.json")
    assert summary["total_frames"] == 1
    assert summary["updated"] == 0
    assert summary["skipped_existing"] == 1
    assert payload == existing_payload
    assert adapter.calls == []
