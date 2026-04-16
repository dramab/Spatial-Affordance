"""
tests/test_text_encoder.py
--------------------------
职责：测试 RoBERTa 文本编码器及其 Transformer token 导出接口。

测试内容：
- test_text_encoder_accepts_raw_texts：验证原始文本输入会被正确编码
- test_text_encoder_accepts_tokenized_inputs：验证预分词输入可直接前向
- test_text_encoder_freeze_only_backbone：验证冻结时仅主干参数被冻结

用法：
    pytest tests/test_text_encoder.py -v
"""

from types import SimpleNamespace

import torch

from src.models.encoders import TextEncoder


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
        seq_len = 5
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


def test_text_encoder_accepts_raw_texts(monkeypatch):
    """
    作用：验证原始字符串列表会被 tokenizer 编码并输出 token。

    输入：
        无，内部构造 mock tokenizer 与模型
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

    encoder = TextEncoder({
        "type": "roberta-base",
        "out_channels": 24,
        "max_length": 8,
    })
    outputs = encoder(["chair near the window", "small red cup"])

    assert outputs["tokens"].shape == (2, 5, 24)
    assert outputs["token_mask"].shape == (2, 5)
    assert torch.all(outputs["token_mask"])


def test_text_encoder_accepts_tokenized_inputs(monkeypatch):
    """
    作用：验证预分词输入可直接传入编码器。

    输入：
        无，内部构造 mock 模型与 tokenized batch
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

    encoder = TextEncoder({
        "type": "roberta-base",
        "out_channels": 16,
    })
    tokenized_inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.long),
    }

    outputs = encoder(tokenized_inputs)

    assert outputs["tokens"].shape == (2, 4, 16)
    assert outputs["token_mask"].dtype == torch.bool


def test_text_encoder_freeze_only_backbone(monkeypatch):
    """
    作用：验证 freeze=True 时仅 Transformer 主干被冻结。

    输入：
        无，内部构造 mock 编码器实例
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

    encoder = TextEncoder({
        "type": "roberta-base",
        "out_channels": 16,
        "freeze": True,
    })

    assert all(not param.requires_grad for param in encoder.backbone.parameters())
    assert all(param.requires_grad for param in encoder.proj.parameters())
