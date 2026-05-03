"""
tests/test_backfill_placement_camera_meta.py
--------------------------------------------
职责：测试旧 placement 输出 camera 回填脚本的核心行为。

测试内容：
- test_backfill_source_dir_writes_missing_camera：
  验证缺失 camera 的 grid_meta 会被回填
- test_backfill_source_dir_skips_existing_camera_by_default：
  验证默认不会覆盖已有 camera

用法：
    pytest tests/test_backfill_placement_camera_meta.py -v
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "tools" / "backfill_placement_camera_meta.py"
SPEC = importlib.util.spec_from_file_location("backfill_placement_camera_meta", MODULE_PATH)
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


def _write_minimal_sample(source_dir: Path) -> None:
    """
    用法: _write_minimal_sample(source_dir)
    作用: 写入包含 scene_id/frame_id 的最小 samples JSON
    输入: source_dir: Path，placement 输出根目录
    输出: None
    """
    _write_json(
        source_dir / "samples/scene_0000_0000.json",
        {
            "scene_id": "scene_0000",
            "frame_id": "0000",
            "samples": [],
        },
    )


class _FakeAdapter:
    """
    作用：记录 load_scene 调用并返回固定相机参数。
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
        作用: 返回带固定 camera 的假 SceneData
        输入: scene_path: str；frame_id: str
        输出: SimpleNamespace，包含 camera 字段
        """
        self.calls.append((scene_path, frame_id))
        camera = SimpleNamespace(
            fx=100.0,
            fy=101.0,
            cx=32.0,
            cy=24.0,
            img_w=64,
            img_h=48,
            E_c2w=np.eye(4, dtype=np.float64),
        )
        return SimpleNamespace(camera=camera)


def test_backfill_source_dir_writes_missing_camera(tmp_path, monkeypatch):
    """
    作用：验证缺失 camera 的 grid_meta 会被回填。

    输入：
        tmp_path: pytest 临时目录
        monkeypatch: pytest monkeypatch 工具
    输出：
        无，通过断言验证结果
    """
    source_dir = tmp_path / "outputs/hope"
    dataset_root = tmp_path / "dataset"
    grid_meta_path = source_dir / "grid_meta/scene_0000_0000.json"
    adapter = _FakeAdapter()

    _write_minimal_sample(source_dir)
    _write_json(grid_meta_path, {"scene_id": "scene_0000", "frame_id": "0000"})

    monkeypatch.setattr(backfill, "infer_config_path", lambda _source_dir: tmp_path / "config.yaml")
    monkeypatch.setattr(
        backfill,
        "load_config",
        lambda _config_path: {"dataset": {"root_dir": str(dataset_root)}},
    )
    monkeypatch.setattr(backfill, "build_adapter", lambda _config: adapter)

    summary = backfill.backfill_source_dir(source_dir, overwrite=False, dry_run=False)

    payload = _read_json(grid_meta_path)
    assert summary["total_frames"] == 1
    assert summary["updated"] == 1
    assert summary["failed"] == 0
    assert payload["camera"]["fx"] == 100.0
    assert payload["camera"]["img_w"] == 64
    assert len(payload["camera"]["E_c2w"]) == 4
    assert adapter.calls == [(str(dataset_root / "scene_0000"), "0000")]


def test_backfill_source_dir_skips_existing_camera_by_default(tmp_path, monkeypatch):
    """
    作用：验证默认不会覆盖已有 grid_meta.camera。

    输入：
        tmp_path: pytest 临时目录
        monkeypatch: pytest monkeypatch 工具
    输出：
        无，通过断言验证结果
    """
    source_dir = tmp_path / "outputs/hope"
    grid_meta_path = source_dir / "grid_meta/scene_0000_0000.json"
    adapter = _FakeAdapter()
    old_camera = {
        "fx": 1.0,
        "fy": 2.0,
        "cx": 3.0,
        "cy": 4.0,
        "img_w": 5,
        "img_h": 6,
        "E_c2w": np.eye(4, dtype=np.float64).tolist(),
    }

    _write_minimal_sample(source_dir)
    _write_json(grid_meta_path, {"camera": old_camera})

    monkeypatch.setattr(backfill, "infer_config_path", lambda _source_dir: tmp_path / "config.yaml")
    monkeypatch.setattr(
        backfill,
        "load_config",
        lambda _config_path: {"dataset": {"root_dir": str(tmp_path / "dataset")}},
    )
    monkeypatch.setattr(backfill, "build_adapter", lambda _config: adapter)

    summary = backfill.backfill_source_dir(source_dir, overwrite=False, dry_run=False)

    payload = _read_json(grid_meta_path)
    assert summary["total_frames"] == 1
    assert summary["updated"] == 0
    assert summary["skipped_existing"] == 1
    assert payload["camera"] == old_camera
    assert adapter.calls == []
