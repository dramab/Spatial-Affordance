"""
tests/test_dopose_adapter.py
----------------------------
职责：测试 DoPose adapter 的路径解析、单位转换和命名规范。

测试内容：
- test_dopose_adapter_loads_synthetic_scene_id：
  验证合成 scene_id、RGBD、相机外参、物体位姿、canonical AABB 和类别名。
- test_dopose_adapter_lists_subsets_with_synthetic_paths：
  验证 adapter 会枚举 test_bin/test_table 并返回无斜杠合成 scene 路径。

用法：
    pytest tests/test_dopose_adapter.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dopose_adapter import DoPoseAdapter


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


def _write_rgb(path: Path, width: int = 4, height: int = 3) -> None:
    """
    用法: _write_rgb(path)
    作用: 写入测试 RGB 图片
    输入: path: Path，图片路径；width/height: int，图片尺寸
    输出: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = 10
    rgb[..., 1] = 20
    rgb[..., 2] = 30
    Image.fromarray(rgb, mode="RGB").save(path)


def _write_depth(path: Path, value: int = 1000, width: int = 4, height: int = 3) -> None:
    """
    用法: _write_depth(path, value=1000)
    作用: 写入测试深度图
    输入: path: Path，深度图路径；value: int，raw uint16 深度值；width/height: int，图片尺寸
    输出: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    depth = np.full((height, width), value, dtype=np.uint16)
    Image.fromarray(depth).save(path)


def _make_models(root_dir: Path) -> tuple[Path, Path]:
    """
    用法: models_info_path, models_names_path = _make_models(root_dir)
    作用: 构造最小 DoPose 模型信息文件
    输入: root_dir: Path，DoPose 根目录
    输出: tuple[Path, Path]，模型尺寸和名称文件路径
    """
    models_info_path = root_dir / "models" / "models_info.json"
    models_names_path = root_dir / "models_names.json"
    _write_json(
        models_info_path,
        {
            "1": {
                "min_x": -5.0,
                "min_y": -6.0,
                "min_z": -7.0,
                "size_x": 10.0,
                "size_y": 12.0,
                "size_z": 14.0,
            }
        },
    )
    _write_json(models_names_path, {"1": {"name": "choco_box"}})
    return models_info_path, models_names_path


def _make_scene(scene_dir: Path) -> None:
    """
    用法: _make_scene(scene_dir)
    作用: 构造最小 DoPose scene
    输入: scene_dir: Path，真实 scene 目录
    输出: None
    """
    _write_rgb(scene_dir / "rgb" / "000000.png")
    _write_rgb(scene_dir / "rgb" / "000001.png")
    _write_depth(scene_dir / "depth" / "000000.png", value=1000)
    _write_depth(scene_dir / "depth" / "000001.png", value=2000)
    _write_json(
        scene_dir / "scene_camera.json",
        {
            "0": {
                "cam_K": [100.0, 0.0, 2.0, 0.0, 101.0, 1.0, 0.0, 0.0, 1.0],
                "depth_scale": 2.0,
            },
            "1": {
                "cam_K": [100.0, 0.0, 2.0, 0.0, 101.0, 1.0, 0.0, 0.0, 1.0],
                "depth_scale": 2.0,
            },
        },
    )
    _write_json(
        scene_dir / "scene_gt.json",
        {
            "0": [
                {
                    "obj_id": 1,
                    "cam_R_m2c": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    "cam_t_m2c": [10.0, 20.0, 300.0],
                },
                {
                    "obj_id": 1,
                    "cam_R_m2c": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    "cam_t_m2c": [10.0, 20.0, 300.0],
                },
            ],
            "1": [],
        },
    )
    _write_json(
        scene_dir / "scene_gt_info.json",
        {
            "0": [
                {"px_count_visib": 20, "visib_fract": 0.5},
                {"px_count_visib": 0, "visib_fract": 0.0},
            ],
            "1": [],
        },
    )
    _write_json(
        scene_dir / "scene_transformations.json",
        {
            "0": [
                {
                    "source_frame": "zivid_optical_frame",
                    "target_frame": "scene_link",
                    "translation": {"x": 0.0, "y": 0.0, "z": 1.0},
                    "rotation_quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                }
            ],
            "1": [
                {
                    "source_frame": "zivid_optical_frame",
                    "target_frame": "scene_link",
                    "translation": {"x": 0.0, "y": 0.0, "z": 1.0},
                    "rotation_quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                }
            ],
        },
    )


def test_dopose_adapter_loads_synthetic_scene_id(tmp_path: Path) -> None:
    """
    作用：验证 DoPose adapter 能通过合成 scene_id 加载数据并完成单位转换。

    输入：
        tmp_path: pytest 临时目录
    输出：
        无，通过断言验证结果
    """
    root_dir = tmp_path / "dopose"
    scene_dir = root_dir / "test_bin" / "000001"
    models_info_path, models_names_path = _make_models(root_dir)
    _make_scene(scene_dir)

    adapter = DoPoseAdapter(
        root_dir=str(root_dir),
        models_info_path=str(models_info_path),
        models_names_path=str(models_names_path),
        min_visib_fract=0.1,
    )
    scene = adapter.load_scene(str(root_dir / "test_bin_000001"), "000000")

    assert scene.scene_id == "test_bin_000001"
    assert scene.frame_id == "000000"
    assert scene.rgb.shape == (3, 4, 3)
    assert np.allclose(scene.depth, 200.0)
    assert scene.camera.fx == 100.0
    assert np.allclose(scene.camera.E_c2w[:3, 3], [0.0, 0.0, -100.0])
    assert len(scene.objects) == 1

    obj = scene.objects[0]
    assert obj.obj_id == "obj_000001_0"
    assert obj.class_name == "choco_box"
    assert np.allclose(obj.bbox3d_canonical, [-0.5, -0.6, -0.7, 0.5, 0.6, 0.7])
    assert np.allclose(obj.pose_world[:3, 3], [1.0, 2.0, -70.0])


def test_dopose_adapter_lists_subsets_with_synthetic_paths(tmp_path: Path) -> None:
    """
    作用：验证 list_scenes 返回后处理兼容的合成 scene 路径。

    输入：
        tmp_path: pytest 临时目录
    输出：
        无，通过断言验证结果
    """
    root_dir = tmp_path / "dopose"
    models_info_path, models_names_path = _make_models(root_dir)
    _make_scene(root_dir / "test_bin" / "000001")
    _make_scene(root_dir / "test_table" / "000002")

    adapter = DoPoseAdapter(
        root_dir=str(root_dir),
        models_info_path=str(models_info_path),
        models_names_path=str(models_names_path),
        frame_step=2,
    )

    assert adapter.list_scenes() == [
        (str(root_dir / "test_bin_000001"), ["000000"]),
        (str(root_dir / "test_table_000002"), ["000000"]),
    ]
