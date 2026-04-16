"""
tests/test_multimodal_model.py
------------------------------
职责：测试统一多模态模型的前向接口与输出格式。

测试内容：
- test_multimodal_model_forward_returns_single_query_boxes：
  验证三模态输入可以被统一编码并输出单 query 3D BBox
- test_multimodal_model_requires_at_least_one_modality：
  验证所有模态都为空时会抛出错误
- test_multimodal_model_denormalizes_boxes_with_scene_meta：
  验证传入场景中心和尺度后，模型会额外输出去归一化 box

用法：
    pytest tests/test_multimodal_model.py -v
"""

from types import SimpleNamespace

import torch

from src.models import MultimodalModel


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


def _make_model_cfg() -> dict:
    """
    作用：构造轻量化多模态模型测试配置。

    输入：
        无
    输出：
        dict 模型配置
    """
    return {
        "image_backbone": {
            "type": "resnet50",
            "pretrained": False,
            "out_channels": 32,
        },
        "pc_backbone": {
            "type": "voxelnet",
            "voxel_size_cm": [5.0, 5.0, 5.0],
            "point_cloud_range_cm": [-20.0, -20.0, -20.0, 20.0, 20.0, 20.0],
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


def _make_points_batch() -> torch.Tensor:
    """
    作用：构造一组简单的测试点云 batch。

    输入：
        无
    输出：
        Tensor(B, N, 3) 点云坐标
    """
    sample_a = torch.tensor([
        [-10.0, -10.0, -10.0],
        [-5.0, -5.0, -5.0],
        [0.0, 0.0, 0.0],
        [5.0, 5.0, 5.0],
    ])
    sample_b = torch.tensor([
        [-8.0, -4.0, -2.0],
        [-4.0, -2.0, 0.0],
        [4.0, 2.0, 6.0],
        [8.0, 4.0, 8.0],
    ])
    return torch.stack([sample_a, sample_b], dim=0).to(torch.float32)


def test_multimodal_model_forward_returns_single_query_boxes(monkeypatch):
    """
    作用：验证多模态模型会输出统一 memory 与单 query 3D BBox。

    输入：
        无，内部构造 mock 文本编码主干、点云、图像与文本
    输出：
        无，通过断言验证结果
    """
    monkeypatch.setattr(
        "src.models.encoders.text_encoder.AutoTokenizer.from_pretrained",
        lambda _: _FakeTokenizer(),
    )
    monkeypatch.setattr(
        "src.models.encoders.text_encoder.AutoModel.from_pretrained",
        lambda _: _FakeModel(),
    )

    model = MultimodalModel(_make_model_cfg())
    points_xyz = _make_points_batch()
    images = torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8)
    text_inputs = [
        "place the mug on the table",
        "move the bowl beside the chair",
    ]

    outputs = model(
        points_xyz=points_xyz,
        images=images,
        text_inputs=text_inputs,
    )

    assert outputs["memory"].ndim == 3
    assert outputs["memory_mask"].dtype == torch.bool
    assert outputs["decoder_tokens"].shape == (2, 1, 32)
    assert outputs["pred_boxes"].shape == (2, 1, 7)
    assert outputs["pred_boxes_norm"].shape == (2, 1, 7)
    assert torch.all(outputs["pred_boxes"][..., 3:6] > 0)
    assert outputs["modality_lengths"]["point"] > 0
    assert outputs["modality_lengths"]["image"] > 0
    assert outputs["modality_lengths"]["text"] == 6


def test_multimodal_model_requires_at_least_one_modality(monkeypatch):
    """
    作用：验证所有模态为空时会抛出错误。

    输入：
        无，内部构造 mock 文本主干并调用空输入前向
    输出：
        无，通过断言验证结果
    """
    monkeypatch.setattr(
        "src.models.encoders.text_encoder.AutoTokenizer.from_pretrained",
        lambda _: _FakeTokenizer(),
    )
    monkeypatch.setattr(
        "src.models.encoders.text_encoder.AutoModel.from_pretrained",
        lambda _: _FakeModel(),
    )

    model = MultimodalModel(_make_model_cfg())

    try:
        model()
    except ValueError as exc:
        assert "at least one modality input" in str(exc)
    else:
        raise AssertionError("expected ValueError when all modality inputs are None")


def test_multimodal_model_denormalizes_boxes_with_scene_meta(monkeypatch):
    """
    作用：验证传入场景标准化元数据后，模型会输出去归一化 box。

    输入：
        无，内部构造 mock 文本主干与场景标准化参数
    输出：
        无，通过断言验证结果
    """
    monkeypatch.setattr(
        "src.models.encoders.text_encoder.AutoTokenizer.from_pretrained",
        lambda _: _FakeTokenizer(),
    )
    monkeypatch.setattr(
        "src.models.encoders.text_encoder.AutoModel.from_pretrained",
        lambda _: _FakeModel(),
    )

    model = MultimodalModel(_make_model_cfg())
    points_xyz = _make_points_batch()
    images = torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8)
    text_inputs = [
        "place the mug on the table",
        "move the bowl beside the chair",
    ]
    box_norm_meta = {
        "scene_center": torch.tensor([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]], dtype=torch.float32),
        "scene_scale": torch.tensor([5.0, 2.0], dtype=torch.float32),
    }

    outputs = model(
        points_xyz=points_xyz,
        images=images,
        text_inputs=text_inputs,
        box_norm_meta=box_norm_meta,
    )

    assert outputs["pred_boxes_norm"].shape == (2, 1, 7)
    assert outputs["pred_boxes"].shape == (2, 1, 7)
    assert not torch.allclose(outputs["pred_boxes"], outputs["pred_boxes_norm"])
    expected_sizes = outputs["pred_boxes_norm"][..., 3:6] * box_norm_meta["scene_scale"].view(2, 1, 1)
    assert torch.allclose(outputs["pred_boxes"][..., 3:6], expected_sizes, atol=1e-5)
