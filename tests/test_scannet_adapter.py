"""
tests/test_scannet_adapter.py
-----------------------------
职责：测试 ScanNet extracted_scans adapter 的格式解析、单位转换和可见实例过滤。

测试内容：
- test_scannet_adapter_loads_visible_non_structural_objects：
  验证 RGBD、相机位姿、mesh 尺寸、2D instance 可见过滤和结构类过滤。
- test_scannet_adapter_lists_common_frames：
  验证 adapter 只枚举 RGB/depth/pose/instance 都存在且符合 frame_step 的帧。

用法：
    pytest tests/test_scannet_adapter.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.scannet_adapter import ScanNetAdapter


def _write_json(path: Path, payload: dict) -> None:
    """
    用法: _write_json(path, payload)
    作用: 写入测试 JSON 文件
    输入: path: Path，输出路径；payload: dict，JSON 内容
    输出: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _write_matrix(path: Path, matrix: np.ndarray) -> None:
    """
    用法: _write_matrix(path, np.eye(4))
    作用: 写入 ScanNet 风格 4x4 文本矩阵
    输入: path: Path；matrix: ndarray(4,4)
    输出: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, fmt="%.6f")


def _write_rgb(path: Path, width: int = 4, height: int = 4) -> None:
    """
    用法: _write_rgb(path)
    作用: 写入测试 RGB 图片
    输入: path: Path，图片路径；width/height: int，图片尺寸
    输出: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(width, dtype=np.uint8)[None, :]
    rgb[..., 1] = 20
    rgb[..., 2] = 30
    Image.fromarray(rgb, mode="RGB").save(path)


def _write_depth(path: Path, value: int = 1000, width: int = 2, height: int = 2) -> None:
    """
    用法: _write_depth(path, value=1000)
    作用: 写入测试深度图
    输入: path: Path；value: int，raw mm 深度；width/height: int
    输出: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    depth = np.full((height, width), value, dtype=np.uint16)
    Image.fromarray(depth).save(path)


def _write_instance(path: Path) -> None:
    """
    用法: _write_instance(path)
    作用: 写入测试 2D instance 图，value=objectId+1
    输入: path: Path，输出路径
    输出: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    instance = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [2, 2, 0, 0],
            [2, 2, 0, 0],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(instance).save(path)


def _write_ply(path: Path) -> None:
    """
    用法: _write_ply(path)
    作用: 写入最小 ScanNet 风格 binary_little_endian PLY
    输入: path: Path，PLY 输出路径
    输出: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.array(
        [
            (0.0, 0.0, 0.0, 10, 20, 30, 255),
            (0.2, 0.4, 0.6, 10, 20, 30, 255),
            (1.0, 1.0, 0.0, 40, 50, 60, 255),
            (1.5, 1.5, 0.1, 40, 50, 60, 255),
        ],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("alpha", "u1"),
        ],
    )
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property uchar alpha\n"
        "element face 0\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        vertices.tofile(f)


def _make_scene(scene_dir: Path) -> None:
    """
    用法: _make_scene(tmp_path / "scene0000_00")
    作用: 构造最小 ScanNet extracted_scans 风格场景
    输入: scene_dir: Path，场景目录
    输出: None
    """
    scene_id = scene_dir.name
    for frame_id in ("0", "1", "2"):
        _write_rgb(scene_dir / "color" / f"{frame_id}.jpg")
        _write_depth(scene_dir / "depth" / f"{frame_id}.png", value=1000 + int(frame_id))
        _write_instance(scene_dir / "2d-instance" / f"{frame_id}.png")

        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = [1.0, 2.0, 3.0]
        _write_matrix(scene_dir / "pose" / f"{frame_id}.txt", pose)

    intrinsic = np.array(
        [
            [100.0, 0.0, 1.0, 0.0],
            [0.0, 101.0, 1.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    _write_matrix(scene_dir / "intrinsic" / "intrinsic_depth.txt", intrinsic)
    _write_ply(scene_dir / f"{scene_id}_vh_clean_2.ply")
    _write_json(scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json", {"segIndices": [10, 10, 20, 20]})
    _write_json(
        scene_dir / f"{scene_id}.aggregation.json",
        {
            "sceneId": f"scannet.{scene_id}",
            "segGroups": [
                {"id": 0, "objectId": 0, "segments": [10], "label": "table"},
                {"id": 1, "objectId": 1, "segments": [20], "label": "floor"},
            ],
        },
    )


def test_scannet_adapter_loads_visible_non_structural_objects(tmp_path: Path) -> None:
    """
    作用：验证 adapter 正确加载可见非结构类实例并完成 cm 单位转换。

    输入：
        tmp_path: pytest 临时目录
    输出：
        无，通过断言验证结果
    """
    root_dir = tmp_path / "extracted_scans"
    scene_dir = root_dir / "scene0000_00"
    _make_scene(scene_dir)

    adapter = ScanNetAdapter(root_dir=str(root_dir))
    scene = adapter.load_scene(str(scene_dir), "0")

    assert scene.scene_id == "scene0000_00"
    assert scene.frame_id == "0"
    assert scene.rgb.shape == (2, 2, 3)
    assert scene.depth.shape == (2, 2)
    assert np.allclose(scene.depth, 100.0)
    assert scene.camera.fx == 100.0
    assert scene.camera.fy == 101.0
    assert np.allclose(scene.camera.E_c2w[:3, 3], [100.0, 200.0, 300.0])
    assert len(scene.objects) == 1

    obj = scene.objects[0]
    assert obj.obj_id == "obj_0"
    assert obj.class_name == "table"
    assert np.allclose(obj.bbox3d_canonical, [-10.0, -20.0, -30.0, 10.0, 20.0, 30.0])
    assert np.allclose(obj.pose_world[:3, 3], [10.0, 20.0, 30.0])


def test_scannet_adapter_lists_common_frames(tmp_path: Path) -> None:
    """
    作用：验证 list_scenes 会按 frame_step 枚举可用帧。

    输入：
        tmp_path: pytest 临时目录
    输出：
        无，通过断言验证结果
    """
    root_dir = tmp_path / "extracted_scans"
    scene_dir = root_dir / "scene0000_00"
    _make_scene(scene_dir)

    adapter = ScanNetAdapter(root_dir=str(root_dir), frame_step=2)

    assert adapter.list_scenes() == [(str(scene_dir), ["0", "2"])]
