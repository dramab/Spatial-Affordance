"""
tests/test_ycb_video_adapter.py
-------------------------------
职责：测试 YCB-Video BOP test adapter 的格式解析与单位转换。

测试内容：
- test_ycb_video_adapter_loads_test_frame：
  验证 RGBD、相机外参、物体位姿、canonical AABB 和可见性过滤。
- test_ycb_video_adapter_lists_png_frames：
  验证 adapter 只枚举 RGB/Depth 都存在且符合 frame_step 的帧。

用法：
    pytest tests/test_ycb_video_adapter.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.datasets.ycb_video_adapter import YCBVideoAdapter


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


def _make_models_info(path: Path) -> None:
    """
    用法: _make_models_info(path)
    作用: 写入最小 YCB models_info.json
    输入: path: Path，models_info 路径
    输出: None
    """
    _write_json(
        path,
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


def _make_scene(scene_dir: Path) -> None:
    """
    用法: _make_scene(scene_dir)
    作用: 构造最小可加载的 BOP test scene
    输入: scene_dir: Path，scene 输出目录
    输出: None
    """
    _write_rgb(scene_dir / "rgb" / "000001.png")
    _write_rgb(scene_dir / "rgb" / "000002.png")
    _write_depth(scene_dir / "depth" / "000001.png", value=1000)
    _write_depth(scene_dir / "depth" / "000002.png", value=2000)
    _write_json(
        scene_dir / "scene_camera.json",
        {
            "1": {
                "cam_K": [100.0, 0.0, 2.0, 0.0, 101.0, 1.0, 0.0, 0.0, 1.0],
                "cam_R_w2c": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "cam_t_w2c": [0.0, 0.0, 1000.0],
                "depth_scale": 0.1,
            },
            "2": {
                "cam_K": [100.0, 0.0, 2.0, 0.0, 101.0, 1.0, 0.0, 0.0, 1.0],
                "cam_R_w2c": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "cam_t_w2c": [0.0, 0.0, 1000.0],
                "depth_scale": 0.1,
            },
        },
    )
    _write_json(
        scene_dir / "scene_gt.json",
        {
            "1": [
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
            "2": [],
        },
    )
    _write_json(
        scene_dir / "scene_gt_info.json",
        {
            "1": [
                {"px_count_visib": 20, "visib_fract": 0.5},
                {"px_count_visib": 0, "visib_fract": 0.0},
            ],
            "2": [],
        },
    )


def test_ycb_video_adapter_loads_test_frame(tmp_path: Path) -> None:
    """
    作用：验证 adapter 正确加载 test 帧并完成 cm 单位转换。

    输入：
        tmp_path: pytest 临时目录
    输出：
        无，通过断言验证结果
    """
    root_dir = tmp_path / "test"
    scene_dir = root_dir / "000048"
    models_info_path = tmp_path / "models_info.json"
    _make_models_info(models_info_path)
    _make_scene(scene_dir)

    adapter = YCBVideoAdapter(
        root_dir=str(root_dir),
        models_info_path=str(models_info_path),
        min_visib_fract=0.1,
    )
    scene = adapter.load_scene(str(scene_dir), "000001")

    assert scene.scene_id == "000048"
    assert scene.frame_id == "000001"
    assert scene.rgb.shape == (3, 4, 3)
    assert np.allclose(scene.depth, 10.0)
    assert scene.camera.fx == 100.0
    assert np.allclose(scene.camera.E_c2w[:3, 3], [0.0, 0.0, -100.0])
    assert len(scene.objects) == 1

    obj = scene.objects[0]
    assert obj.obj_id == "obj_000001_0"
    assert obj.class_name == "002_master_chef_can"
    assert np.allclose(obj.bbox3d_canonical, [-0.5, -0.6, -0.7, 0.5, 0.6, 0.7])
    assert np.allclose(obj.pose_world[:3, 3], [1.0, 2.0, -70.0])


def test_ycb_video_adapter_lists_png_frames(tmp_path: Path) -> None:
    """
    作用：验证 list_scenes 会按 frame_step 枚举可用 PNG 帧。

    输入：
        tmp_path: pytest 临时目录
    输出：
        无，通过断言验证结果
    """
    root_dir = tmp_path / "test"
    scene_dir = root_dir / "000048"
    models_info_path = tmp_path / "models_info.json"
    _make_models_info(models_info_path)
    _make_scene(scene_dir)

    adapter = YCBVideoAdapter(
        root_dir=str(root_dir),
        models_info_path=str(models_info_path),
        frame_step=2,
    )

    assert adapter.list_scenes() == [(str(scene_dir), ["000002"])]
