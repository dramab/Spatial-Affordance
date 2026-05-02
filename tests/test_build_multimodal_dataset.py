"""
tests/test_build_multimodal_dataset.py
--------------------------------------
职责：测试多模态数据集构建脚本的关键行为。

测试内容：
- test_build_dataset_aligns_modalities_and_writes_splits：
  验证脚本会正确对齐多模态数据并写出 train/test/summary
- test_build_dataset_raises_on_missing_label：
  验证缺失文本标签时会抛出明确错误
- test_build_dataset_allows_missing_polished_label：
  验证缺失 polished_label 时会写入空字符串
- test_build_frame_lookup_ignores_samples_without_rgb_before_label_check：
  验证无 RGB 的样本不会要求存在文本标签

用法：
    pytest tests/test_build_multimodal_dataset.py -v
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "tools" / "build_multimodal_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_multimodal_dataset", MODULE_PATH)
build_multimodal_dataset = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(build_multimodal_dataset)


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


def _write_ascii_ply(path: Path, points: np.ndarray) -> None:
    """
    用法: _write_ascii_ply(path, points)
    作用: 为测试写入最小可读的 ASCII PLY 点云
    输入: path: Path；points: ndarray(N, 3)
    输出: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        for x, y, z in points:
            f.write(f"{x:.4f} {y:.4f} {z:.4f} 255 0 0\n")


def _make_fake_camera_dict(fx: float = 100.0) -> dict:
    """
    用法: camera_dict = _make_fake_camera_dict()
    作用: 构造可序列化的假相机参数
    输入: fx: float，相机焦距
    输出: dict，相机字段
    """
    return {
        "fx": fx,
        "fy": fx + 1.0,
        "cx": 32.0,
        "cy": 24.0,
        "img_w": 64,
        "img_h": 48,
        "E_c2w": np.eye(4, dtype=np.float64).tolist(),
    }


def _make_transform(
    translation: tuple[float, float, float],
    rotation: np.ndarray | None = None,
) -> list[list[float]]:
    """
    用法: transform = _make_transform((1.0, 0.0, 0.0))
    作用: 构造测试用 object→world 齐次变换
    输入: translation: tuple[float,float,float]；rotation: 可选 ndarray(3,3)
    输出: list[list[float]]，可写入 JSON 的 4x4 变换
    """
    transform = np.eye(4, dtype=np.float64)
    if rotation is not None:
        transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform.tolist()


def _make_yaw_transform(
    translation: tuple[float, float, float],
    yaw_degrees: float,
) -> list[list[float]]:
    """
    用法: transform = _make_yaw_transform((1.0, 0.0, 0.0), 30.0)
    作用: 构造带 Z 轴 yaw 的测试用 object→world 变换
    输入: translation: tuple[float,float,float]；yaw_degrees: float
    输出: list[list[float]]，可写入 JSON 的 4x4 变换
    """
    rotation = build_multimodal_dataset.rotation_z_3x3(np.deg2rad(float(yaw_degrees)))
    return _make_transform(translation, rotation)


def test_compute_yaw_aligned_box_size_uses_transform_without_yaw_inflation():
    """
    作用：验证 whl 来自 canonical AABB 与 transform_world，且不会重复计入 yaw 膨胀。

    输入：
        无
    输出：
        无，通过断言验证结果
    """
    canonical_aabb = np.array([-1.0, -2.0, -3.0, 1.0, 2.0, 3.0], dtype=np.float64)
    yaw_rad = np.deg2rad(45.0)
    cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
    yaw_rotation = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    size = build_multimodal_dataset.compute_yaw_aligned_box_size(
        canonical_aabb_object=canonical_aabb,
        transform_world=np.asarray(
            _make_transform((10.0, 20.0, 30.0), yaw_rotation),
            dtype=np.float64,
        ),
        yaw_degrees=45.0,
    )

    assert np.allclose(size, [2.0, 4.0, 6.0], atol=1e-6)


def test_compute_yaw_aligned_box_size_reflects_sideways_pose():
    """
    作用：验证侧放姿态会通过 transform_world 改变训练监督的 whl。

    输入：
        无
    输出：
        无，通过断言验证结果
    """
    canonical_aabb = np.array([-1.0, -2.0, -3.0, 1.0, 2.0, 3.0], dtype=np.float64)
    roll_rad = np.deg2rad(90.0)
    cos_roll, sin_roll = np.cos(roll_rad), np.sin(roll_rad)
    roll_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_roll, -sin_roll],
            [0.0, sin_roll, cos_roll],
        ],
        dtype=np.float64,
    )

    size = build_multimodal_dataset.compute_yaw_aligned_box_size(
        canonical_aabb_object=canonical_aabb,
        transform_world=np.asarray(
            _make_transform((0.0, 0.0, 0.0), roll_rotation),
            dtype=np.float64,
        ),
        yaw_degrees=0.0,
    )

    assert np.allclose(size, [2.0, 6.0, 4.0], atol=1e-6)


def test_build_dataset_aligns_modalities_and_writes_splits(tmp_path, monkeypatch):
    """
    作用：验证脚本会正确对齐多模态数据并写出 train/test/summary。

    输入：
        tmp_path: pytest 临时目录
        monkeypatch: pytest monkeypatch 工具
    输出：
        无，通过断言验证结果
    """
    rgb_dir = tmp_path / "outputs/placement_rgb_bbox_vis"
    hope_dir = tmp_path / "outputs/hope"
    house_dir = tmp_path / "outputs/housecat6d"
    label_json = tmp_path / "outputs/auto_labels/all_labels_polished.json"
    output_dir = tmp_path / "data/annotations/placement_multimodal"

    (rgb_dir).mkdir(parents=True, exist_ok=True)
    for image_name in [
        "hope__scene_0000_0000_obj_0_p000.png",
        "hope__scene_0000_0000_obj_1_p000.png",
        "housecat6d__scene01_000000_obj_2_p000.png",
        "housecat6d__scene01_000000_obj_3_p000.png",
    ]:
        (rgb_dir / image_name).write_bytes(b"png")

    _write_ascii_ply(
        hope_dir / "point_clouds/scene_0000_0000.ply",
        np.array([[0.0, 0.0, 10.0], [2.0, 4.0, 14.0]], dtype=np.float64),
    )
    _write_ascii_ply(
        house_dir / "point_clouds/scene01_000000.ply",
        np.array([[-1.0, 1.0, 5.0], [3.0, 5.0, 9.0]], dtype=np.float64),
    )
    _write_json(hope_dir / "grid_meta/scene_0000_0000.json", {"ok": True})
    _write_json(house_dir / "grid_meta/scene01_000000.json", {"ok": True})

    _write_json(
        hope_dir / "samples/scene_0000_0000.json",
        {
            "samples": [
                {
                    "sample_id": "scene_0000_0000_obj_0_p000",
                    "scene_id": "scene_0000",
                    "frame_id": "0000",
                    "canonical_aabb_object": [-1, -2, -3, 1, 2, 3],
                    "transform_world": _make_yaw_transform((0, 0, 0), 15.0),
                    "center_world": [0, 0, 0],
                    "aabb_world": [-1, -2, -3, 1, 2, 3],
                    "yaw_degrees": 15.0,
                },
                {
                    "sample_id": "scene_0000_0000_obj_1_p000",
                    "scene_id": "scene_0000",
                    "frame_id": "0000",
                    "canonical_aabb_object": [-1, -2, -3, 1, 2, 3],
                    "transform_world": _make_yaw_transform((1, 0, 0), 30.0),
                    "center_world": [1, 0, 0],
                    "aabb_world": [0, -2, -3, 2, 2, 3],
                    "yaw_degrees": 30.0,
                },
            ]
        },
    )
    _write_json(
        house_dir / "samples/scene01_000000.json",
        {
            "samples": [
                {
                    "sample_id": "scene01_000000_obj_2_p000",
                    "scene_id": "scene01",
                    "frame_id": "000000",
                    "canonical_aabb_object": [-1, -2, -2, 1, 2, 2],
                    "transform_world": _make_yaw_transform((0, 1, 0), 45.0),
                    "center_world": [0, 1, 0],
                    "aabb_world": [-1, -1, -2, 1, 3, 2],
                    "yaw_degrees": 45.0,
                },
                {
                    "sample_id": "scene01_000000_obj_3_p000",
                    "scene_id": "scene01",
                    "frame_id": "000000",
                    "canonical_aabb_object": [-1, -2, -2, 1, 2, 2],
                    "transform_world": _make_yaw_transform((0, 2, 0), 60.0),
                    "center_world": [0, 2, 0],
                    "aabb_world": [-1, 0, -2, 1, 4, 2],
                    "yaw_degrees": 60.0,
                },
            ]
        },
    )
    _write_json(
        label_json,
        [
            {
                "image_filename": "hope__scene_0000_0000_obj_0_p000.png",
                "sample_id": "scene_0000_0000_obj_0_p000",
                "source_name": "hope",
                "target_object_name": "objA",
                "is_found_target": True,
                "label": "raw hope 0",
                "polished_label": "polished hope 0",
            },
            {
                "image_filename": "hope__scene_0000_0000_obj_1_p000.png",
                "sample_id": "scene_0000_0000_obj_1_p000",
                "source_name": "hope",
                "target_object_name": "objB",
                "is_found_target": True,
                "label": "raw hope 1",
                "polished_label": "polished hope 1",
            },
            {
                "image_filename": "housecat6d__scene01_000000_obj_2_p000.png",
                "sample_id": "scene01_000000_obj_2_p000",
                "source_name": "housecat6d",
                "target_object_name": "objC",
                "is_found_target": True,
                "label": "raw house 2",
                "polished_label": "polished house 2",
            },
            {
                "image_filename": "housecat6d__scene01_000000_obj_3_p000.png",
                "sample_id": "scene01_000000_obj_3_p000",
                "source_name": "housecat6d",
                "target_object_name": "objD",
                "is_found_target": True,
                "label": "raw house 3",
                "polished_label": "polished house 3",
            },
        ],
    )

    def _fake_load_camera_for_frame(source_name, source_cfg, scene_id, frame_id):
        del source_name, source_cfg, scene_id
        if frame_id == "0000":
            return _make_fake_camera_dict(100.0)
        return _make_fake_camera_dict(200.0)

    monkeypatch.setattr(build_multimodal_dataset, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        build_multimodal_dataset,
        "build_source_configs",
        lambda _source_names: {"hope": {"dataset": {"root_dir": "/unused"}}, "housecat6d": {"dataset": {"root_dir": "/unused"}}},
    )
    monkeypatch.setattr(build_multimodal_dataset, "load_camera_for_frame", _fake_load_camera_for_frame)

    summary = build_multimodal_dataset.build_multimodal_dataset(
        rgb_dir=rgb_dir,
        label_json=label_json,
        source_dirs=[hope_dir, house_dir],
        output_dir=output_dir,
        train_ratio=0.5,
        valid_ratio=0.25,
        seed=7,
    )

    assert summary["total_samples"] == 4
    assert summary["train_samples"] == 2
    assert summary["test_samples"] == 2
    assert summary["by_source"]["all"] == {"hope": 2, "housecat6d": 2}
    assert summary["by_source"]["train"] == {"hope": 1, "housecat6d": 1}
    assert summary["by_source"]["test"] == {"hope": 1, "housecat6d": 1}

    train_payload = json.loads((output_dir / "train.json").read_text(encoding="utf-8"))
    valid_payload = json.loads((output_dir / "valid.json").read_text(encoding="utf-8"))
    test_payload = json.loads((output_dir / "test.json").read_text(encoding="utf-8"))
    assert train_payload["schema_version"] == build_multimodal_dataset.SCHEMA_VERSION
    assert valid_payload["schema_version"] == build_multimodal_dataset.SCHEMA_VERSION
    assert test_payload["schema_version"] == build_multimodal_dataset.SCHEMA_VERSION

    sample = train_payload["samples"][0]
    assert sorted(sample.keys()) == [
        "camera",
        "placement",
        "point_cloud_path",
        "polished_prompt",
        "prompt",
        "rgb_path",
        "sample_id",
        "source_name",
    ]
    assert sample["prompt"].startswith("raw")
    assert sample["polished_prompt"].startswith("polished")
    assert sample["camera"]["img_w"] == 64
    assert len(sample["camera"]["E_c2w"]) == 4
    assert sorted(sample["placement"].keys()) == ["target_box"]
    assert len(sample["placement"]["target_box"]) == 7
    assert sample["placement"]["target_box"][-1] in {15.0, 30.0, 45.0, 60.0}

    samples_by_id = {
        item["sample_id"]: item
        for payload in (train_payload, valid_payload, test_payload)
        for item in payload["samples"]
    }
    assert np.allclose(
        samples_by_id["scene_0000_0000_obj_0_p000"]["placement"]["target_box"][3:6],
        [2.0, 4.0, 6.0],
        atol=1e-6,
    )
    assert np.allclose(
        samples_by_id["scene01_000000_obj_2_p000"]["placement"]["target_box"][3:6],
        [2.0, 4.0, 4.0],
        atol=1e-6,
    )


def test_build_dataset_allows_missing_polished_label(tmp_path, monkeypatch):
    """
    作用：验证 polished_label 缺失时不会报错，并会写入空字符串。

    输入：
        tmp_path: pytest 临时目录
        monkeypatch: pytest monkeypatch 工具
    输出：
        无，通过断言验证结果
    """
    rgb_dir = tmp_path / "outputs/placement_rgb_bbox_vis"
    hope_dir = tmp_path / "outputs/hope"
    output_dir = tmp_path / "data/annotations/placement_multimodal"
    label_json = tmp_path / "outputs/auto_labels/all_labels.json"

    rgb_dir.mkdir(parents=True, exist_ok=True)
    (rgb_dir / "hope__scene_0000_0000_obj_0_p000.png").write_bytes(b"png")
    _write_ascii_ply(
        hope_dir / "point_clouds/scene_0000_0000.ply",
        np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
    )
    _write_json(hope_dir / "grid_meta/scene_0000_0000.json", {"grid": "meta"})
    _write_json(
        hope_dir / "samples/scene_0000_0000.json",
        {
            "samples": [
                {
                    "sample_id": "scene_0000_0000_obj_0_p000",
                    "scene_id": "scene_0000",
                    "frame_id": "0000",
                    "canonical_aabb_object": [-1, -1, -1, 1, 1, 1],
                    "transform_world": _make_transform((0.0, 0.0, 0.0)),
                    "center_world": [0.0, 0.0, 0.0],
                    "aabb_world": [-1, -1, -1, 1, 1, 1],
                    "yaw_degrees": 0.0,
                }
            ]
        },
    )
    _write_json(
        label_json,
        [
            {
                "image_filename": "hope__scene_0000_0000_obj_0_p000.png",
                "sample_id": "scene_0000_0000_obj_0_p000",
                "source_name": "hope",
                "label": "raw prompt",
            }
        ],
    )

    monkeypatch.setattr(build_multimodal_dataset, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        build_multimodal_dataset,
        "build_source_configs",
        lambda _source_names: {"hope": {"dataset": {"root_dir": "/unused"}}},
    )

    def _fake_load_camera_for_frame(source_name, source_cfg, scene_id, frame_id):
        """
        作用：返回固定测试相机参数，避免读取真实数据集。
        """
        del source_name, source_cfg, scene_id, frame_id
        return _make_fake_camera_dict()

    monkeypatch.setattr(
        build_multimodal_dataset,
        "load_camera_for_frame",
        _fake_load_camera_for_frame,
    )

    summary = build_multimodal_dataset.build_multimodal_dataset(
        rgb_dir=rgb_dir,
        label_json=label_json,
        source_dirs=[hope_dir],
        output_dir=output_dir,
        train_ratio=0.8,
        valid_ratio=0.1,
        seed=42,
    )

    assert summary["total_samples"] == 1
    payloads = [
        json.loads((output_dir / split_name).read_text(encoding="utf-8"))
        for split_name in ("train.json", "valid.json", "test.json")
    ]
    samples = [sample for payload in payloads for sample in payload["samples"]]
    assert samples[0]["prompt"] == "raw prompt"
    assert samples[0]["polished_prompt"] == ""


def test_build_frame_lookup_ignores_samples_without_rgb_before_label_check(tmp_path, monkeypatch):
    """
    作用：验证缺少 RGB 的样本会先被过滤，不会触发缺失 label 错误。

    输入：
        tmp_path: pytest 临时目录
        monkeypatch: pytest monkeypatch 工具
    输出：
        无，通过断言验证结果
    """
    hope_dir = tmp_path / "outputs/hope"
    _write_json(
        hope_dir / "samples/scene_0000_0000.json",
        {
            "samples": [
                {
                    "sample_id": "scene_0000_0000_obj_0_p000",
                    "scene_id": "scene_0000",
                    "frame_id": "0000",
                },
                {
                    "sample_id": "scene_0000_0000_obj_1_p000",
                    "scene_id": "scene_0000",
                    "frame_id": "0000",
                },
            ]
        },
    )
    label_lookup = {
        ("hope", "scene_0000_0000_obj_0_p000"): {
            "sample_id": "scene_0000_0000_obj_0_p000",
            "source_name": "hope",
            "label": "raw prompt",
        }
    }

    monkeypatch.setattr(build_multimodal_dataset, "PROJECT_ROOT", tmp_path)
    frame_lookup = build_multimodal_dataset.build_frame_lookup(
        source_dirs=[hope_dir],
        label_lookup=label_lookup,
        available_rgb_filenames={"hope__scene_0000_0000_obj_0_p000.png"},
    )

    assert list(frame_lookup.keys()) == [("hope", "scene_0000", "0000")]
    assert len(frame_lookup[("hope", "scene_0000", "0000")]) == 1
    assert frame_lookup[("hope", "scene_0000", "0000")][0]["sample_id"] == "scene_0000_0000_obj_0_p000"


def test_build_dataset_raises_on_missing_label(tmp_path, monkeypatch):
    """
    作用：验证缺失文本标签时会抛出明确错误。

    输入：
        tmp_path: pytest 临时目录
        monkeypatch: pytest monkeypatch 工具
    输出：
        无，通过断言验证结果
    """
    rgb_dir = tmp_path / "outputs/placement_rgb_bbox_vis"
    hope_dir = tmp_path / "outputs/hope"
    label_json = tmp_path / "outputs/auto_labels/all_labels_polished.json"

    rgb_dir.mkdir(parents=True, exist_ok=True)
    (rgb_dir / "hope__scene_0000_0000_obj_0_p000.png").write_bytes(b"png")
    _write_json(
        hope_dir / "samples/scene_0000_0000.json",
        {
            "samples": [
                {
                    "sample_id": "scene_0000_0000_obj_0_p000",
                    "scene_id": "scene_0000",
                    "frame_id": "0000",
                }
            ]
        },
    )
    _write_json(label_json, [])

    monkeypatch.setattr(build_multimodal_dataset, "PROJECT_ROOT", tmp_path)

    try:
        build_multimodal_dataset.build_multimodal_dataset(
            rgb_dir=rgb_dir,
            label_json=label_json,
            source_dirs=[hope_dir],
            output_dir=tmp_path / "data/annotations/placement_multimodal",
            train_ratio=0.8,
            valid_ratio=0.1,
            seed=42,
        )
    except KeyError as exc:
        assert "Missing label for sample" in str(exc)
    else:
        raise AssertionError("expected KeyError when label is missing")
