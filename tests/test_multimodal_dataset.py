"""
tests/test_multimodal_dataset.py
--------------------------------
职责：测试 placement 多模态 Dataset 的加载、归一化与 batch 拼装行为。

测试内容：
- test_multimodal_dataset_returns_empty_valid_split_when_file_missing：
  验证 valid.json 缺失时会返回空数据集
- test_multimodal_dataset_normalizes_points_and_box：
  验证点云、3D box 与 yaw 会按约定规则完成归一化
- test_multimodal_dataset_collate_batch_can_feed_model：
  验证 collate_fn 输出的 batch 可直接喂给多模态模型

用法：
    pytest tests/test_multimodal_dataset.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from src.datasets import (
    PlacementMultimodalDataset,
    placement_multimodal_collate_fn,
)
from src.models import MultimodalModel


def _write_json(path: Path, payload: dict | list) -> None:
    """
    用法: _write_json(path, payload)
    作用: 为测试写入 JSON 文件
    输入: path: Path；payload: dict 或 list
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


def _write_rgb_image(path: Path, width: int = 32, height: int = 24) -> None:
    """
    用法: _write_rgb_image(path)
    作用: 为测试写入简单 RGB 图片
    输入: path: Path；width: int；height: int
    输出: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = 128
    image[..., 1] = 64
    image[..., 2] = 32
    Image.fromarray(image, mode="RGB").save(path)


def _make_model_cfg() -> dict:
    """
    用法: cfg = _make_model_cfg()
    作用: 构造轻量化多模态模型测试配置
    输入: 无
    输出: dict，模型配置
    """
    return {
        "image_backbone": {
            "type": "resnet50",
            "pretrained": False,
            "out_channels": 32,
        },
        "pc_backbone": {
            "type": "voxelnet",
            "voxel_size": [0.25, 0.25, 0.25],
            "point_cloud_range": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            "max_points_per_voxel": 8,
            "max_voxels": 256,
            "input_feature_dim": 6,
            "svfe_hidden_channels": 16,
            "svfe_out_channels": 32,
            "cml_channels": [32],
        },
        "text_encoder": {
            "type": "roberta-base",
            "out_channels": 32,
            "max_length": 16,
        },
        "fusion": {
            "hidden_dim": 32,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.0,
        },
        "decoder": {
            "hidden_dim": 32,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.0,
            "num_queries": 1,
        },
        "bbox3d_head": {
            "hidden_dim": 32,
            "num_layers": 2,
            "out_dim": 7,
        },
    }


class _FakeTokenizer:
    """
    作用：在单元测试中模拟 HuggingFace tokenizer。

    输入：
        texts: list[str] 原始文本
    输出：
        dict[str, Tensor]，包含 input_ids 与 attention_mask
    """

    def __call__(
            self,
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"):
        del padding, truncation, max_length, return_tensors
        batch_size = len(texts)
        seq_len = 6
        input_ids = torch.arange(batch_size * seq_len).view(batch_size, seq_len)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class _FakeModel(torch.nn.Module):
    """
    作用：在单元测试中模拟 HuggingFace Transformer 主干。

    输入：
        input_ids: Tensor(B, L)
        attention_mask: Tensor(B, L)
    输出：
        具有 last_hidden_state 属性的对象
    """

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = torch.nn.Embedding(256, hidden_size)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        del attention_mask, position_ids
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def _build_fake_multimodal_annotation_root(tmp_path: Path) -> Path:
    """
    用法: annotation_dir = _build_fake_multimodal_annotation_root(tmp_path)
    作用: 为测试构造最小可读的多模态标注目录
    输入: tmp_path: Path，pytest 临时目录
    输出: Path，标注根目录
    """
    annotation_dir = tmp_path / "data/annotations/placement_multimodal"
    outputs_dir = tmp_path / "outputs"
    rgb_dir = outputs_dir / "placement_rgb_bbox_vis"
    point_cloud_dir = outputs_dir / "demo/point_clouds"

    _write_rgb_image(rgb_dir / "demo__sample_a.png", width=32, height=24)
    _write_rgb_image(rgb_dir / "demo__sample_b.png", width=40, height=30)

    _write_ascii_ply(
        point_cloud_dir / "sample_a.ply",
        np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )
    _write_ascii_ply(
        point_cloud_dir / "sample_b.ply",
        np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                [3.0, 1.0, 1.0],
                [4.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )

    train_payload = {
        "schema_version": "placement_multimodal_dataset/v1",
        "split": "train",
        "sample_count": 2,
        "samples": [
            {
                "sample_id": "sample_a",
                "source_name": "demo",
                "rgb_path": "outputs/placement_rgb_bbox_vis/demo__sample_a.png",
                "point_cloud_path": "outputs/demo/point_clouds/sample_a.ply",
                "prompt": "raw prompt a",
                "polished_prompt": "polished prompt a",
                "placement": {
                    "target_box": [1.0, 2.0, 0.0, 2.0, 4.0, 6.0, 270.0],
                },
                "camera": {
                    "fx": 100.0,
                    "fy": 100.0,
                    "cx": 16.0,
                    "cy": 12.0,
                    "img_w": 32,
                    "img_h": 24,
                    "E_c2w": np.eye(4, dtype=np.float64).tolist(),
                },
            },
            {
                "sample_id": "sample_b",
                "source_name": "demo",
                "rgb_path": "outputs/placement_rgb_bbox_vis/demo__sample_b.png",
                "point_cloud_path": "outputs/demo/point_clouds/sample_b.ply",
                "prompt": "raw prompt b",
                "polished_prompt": "polished prompt b",
                "placement": {
                    "target_box": [2.5, 1.0, 1.0, 1.0, 1.0, 2.0, 90.0],
                },
                "camera": {
                    "fx": 120.0,
                    "fy": 121.0,
                    "cx": 16.0,
                    "cy": 12.0,
                    "img_w": 32,
                    "img_h": 24,
                    "E_c2w": np.eye(4, dtype=np.float64).tolist(),
                },
            },
        ],
    }
    test_payload = {
        "schema_version": "placement_multimodal_dataset/v1",
        "split": "test",
        "sample_count": 1,
        "samples": [train_payload["samples"][0]],
    }

    _write_json(annotation_dir / "train.json", train_payload)
    _write_json(annotation_dir / "test.json", test_payload)
    return annotation_dir


def test_multimodal_dataset_returns_empty_valid_split_when_file_missing(tmp_path, monkeypatch):
    """
    作用：验证 valid.json 缺失时会返回空数据集。

    输入：
        tmp_path: pytest 临时目录
        monkeypatch: pytest monkeypatch 工具
    输出：
        无，通过断言验证结果
    """
    annotation_dir = _build_fake_multimodal_annotation_root(tmp_path)
    monkeypatch.setattr("src.datasets.multimodal_dataset.PROJECT_ROOT", tmp_path)

    dataset = PlacementMultimodalDataset(annotation_dir=annotation_dir, split="valid")

    assert len(dataset) == 0
    assert dataset.samples == []


def test_multimodal_dataset_normalizes_points_and_box(tmp_path, monkeypatch):
    """
    作用：验证点云、3D box 与 yaw 会按约定规则完成归一化。

    输入：
        tmp_path: pytest 临时目录
        monkeypatch: pytest monkeypatch 工具
    输出：
        无，通过断言验证结果
    """
    annotation_dir = _build_fake_multimodal_annotation_root(tmp_path)
    monkeypatch.setattr("src.datasets.multimodal_dataset.PROJECT_ROOT", tmp_path)

    dataset = PlacementMultimodalDataset(annotation_dir=annotation_dir, split="train")
    sample = dataset[0]

    expected_center = torch.tensor([2.0 / 3.0, 4.0 / 3.0, 0.0], dtype=torch.float32)
    expected_scale = torch.tensor(4.0, dtype=torch.float32)
    expected_points = torch.tensor(
        [
            [-1.0 / 6.0, -1.0 / 3.0, 0.0],
            [1.0 / 3.0, -1.0 / 3.0, 0.0],
            [-1.0 / 6.0, 2.0 / 3.0, 0.0],
        ],
        dtype=torch.float32,
    )
    expected_box = torch.tensor(
        [
            1.0 / 12.0,
            1.0 / 6.0,
            0.0,
            0.5,
            1.0,
            1.5,
            -0.5,
        ],
        dtype=torch.float32,
    )

    assert sample["sample_id"] == "sample_a"
    assert sample["image"].shape == (3, 480, 640)
    assert torch.allclose(sample["norm_meta"]["scene_center"], expected_center, atol=1e-6)
    assert torch.allclose(sample["norm_meta"]["scene_scale"], expected_scale, atol=1e-6)
    assert torch.allclose(sample["points_xyz_norm"], expected_points, atol=1e-6)
    assert torch.allclose(sample["target_box_norm"], expected_box, atol=1e-6)
    assert sample["text_input"] == "polished prompt a"
    assert "source_name" not in sample
    assert "target_box" not in sample
    assert "points_xyz" not in sample


def test_multimodal_dataset_collate_batch_can_feed_model(tmp_path, monkeypatch):
    """
    作用：验证 collate_fn 输出的 batch 可直接喂给多模态模型。

    输入：
        tmp_path: pytest 临时目录
        monkeypatch: pytest monkeypatch 工具
    输出：
        无，通过断言验证结果
    """
    annotation_dir = _build_fake_multimodal_annotation_root(tmp_path)
    monkeypatch.setattr("src.datasets.multimodal_dataset.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "src.models.encoders.text_encoder.AutoTokenizer.from_pretrained",
        lambda _: _FakeTokenizer(),
    )
    monkeypatch.setattr(
        "src.models.encoders.text_encoder.AutoModel.from_pretrained",
        lambda _: _FakeModel(),
    )

    dataset = PlacementMultimodalDataset(
        annotation_dir=annotation_dir,
        split="train",
    )
    batch = placement_multimodal_collate_fn([dataset[0], dataset[1]])

    assert batch["images"].shape == (2, 3, 480, 640)
    assert batch["points_xyz"].shape == (2, 4, 3)
    assert batch["target_boxes_norm"].shape == (2, 7)
    assert torch.isnan(batch["points_xyz"][0, -1]).all()
    assert not torch.isnan(batch["points_xyz"][1]).any()
    assert batch["norm_meta"]["scene_center"].shape == (2, 3)
    assert batch["norm_meta"]["scene_scale"].shape == (2,)
    assert batch["text_inputs"] == ["polished prompt a", "polished prompt b"]

    model = MultimodalModel(_make_model_cfg())
    outputs = model(
        points_xyz=batch["points_xyz"],
        images=batch["images"],
        text_inputs=batch["text_inputs"],
    )

    assert outputs["pred_boxes_norm"].shape == (2, 1, 7)
