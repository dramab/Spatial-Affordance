"""
tests/test_multimodal_model.py
------------------------------
职责：测试统一多模态模型的前向接口与输出格式。

测试内容：
- test_multimodal_model_forward_returns_single_query_boxes：
  验证三模态输入可以被统一编码并输出单 query 3D BBox
- test_multimodal_model_requires_at_least_one_modality：
  验证所有模态都为空时会抛出错误
- test_multimodal_model_forward_outputs_normalized_boxes_only：
  验证模型只输出 normalized box，由外部后处理负责坐标恢复

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


class _FakeTokenPointBackbone(torch.nn.Module):
    """
    作用：在单元测试中模拟直接返回 token 的点云 backbone。

    输入：
        points_xyz: Tensor(B, N, 3)
        point_feats: 可选点特征
    输出：
        dict，包含 tokens、token_mask、token_pos
    """

    out_channels = 32

    def forward(self, points_xyz, point_feats=None):
        """
        作用：返回固定长度的 batch-first 点云 token。

        输入：
            points_xyz: Tensor(B, N, 3) 点云坐标
            point_feats: 可选点特征
        输出：
            dict，包含 tokens、token_mask、token_pos
        """
        del point_feats
        batch_size = int(points_xyz.shape[0])
        device = points_xyz.device
        tokens = torch.ones((batch_size, 3, self.out_channels), dtype=torch.float32, device=device)
        token_mask = torch.ones((batch_size, 3), dtype=torch.bool, device=device)
        token_pos = points_xyz[:, :3].to(torch.float32)
        return {
            "tokens": tokens,
            "token_mask": token_mask,
            "token_pos": token_pos,
        }


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
        "object_center_head": {
            "hidden_dim": 32,
            "num_layers": 2,
            "out_dim": 3,
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
        [-1.0, -1.0, -1.0],
        [-0.5, -0.5, -0.5],
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    ])
    sample_b = torch.tensor([
        [-0.8, -0.4, -0.2],
        [-0.4, -0.2, 0.0],
        [0.4, 0.2, 0.6],
        [0.8, 0.4, 0.8],
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
    assert outputs["pred_boxes_norm"].shape == (2, 1, 7)
    assert outputs["pred_object_centers_norm"].shape == (2, 1, 3)
    assert torch.isfinite(outputs["pred_boxes_norm"]).all()
    assert torch.isfinite(outputs["pred_object_centers_norm"]).all()
    assert "pred_boxes" not in outputs
    assert outputs["modality_lengths"]["point"] > 0
    assert outputs["modality_lengths"]["image"] > 0
    assert outputs["modality_lengths"]["text"] == 6


def test_multimodal_model_accepts_direct_point_tokens(monkeypatch):
    """
    作用：验证模型可直接消费点云 backbone 返回的 token 字典。

    输入：
        无，内部替换点云 backbone 为 token 输出模拟器
    输出：
        无，通过断言验证点云分支长度
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
    model.pc_backbone = _FakeTokenPointBackbone()
    points_xyz = _make_points_batch()

    outputs = model(points_xyz=points_xyz)

    assert outputs["memory"].shape[1] == 3
    assert outputs["modality_lengths"]["point"] == 3
    assert outputs["memory_mask"].all()


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


def test_multimodal_model_forward_outputs_normalized_boxes_only(monkeypatch):
    """
    作用：验证模型前向只输出 normalized box，不在模型内部做坐标恢复。

    输入：
        无，内部构造 mock 文本主干并执行前向
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

    assert outputs["pred_boxes_norm"].shape == (2, 1, 7)
    assert outputs["pred_object_centers_norm"].shape == (2, 1, 3)
    assert torch.isfinite(outputs["pred_boxes_norm"]).all()
    assert torch.isfinite(outputs["pred_object_centers_norm"]).all()
    assert "pred_boxes" not in outputs
