"""
src/models/encoders/text_encoder.py
-----------------------------------
职责：基于 HuggingFace Transformer 的文本编码器，并输出适配 Transformer 的 token 接口。

用法：
    from src.models.encoders.text_encoder import TextEncoder

    encoder = TextEncoder({
        "type": "roberta-base",
        "out_channels": 256,
        "max_length": 64,
    })
    outputs = encoder(["the brown wooden chair"])
    text_tokens = outputs["tokens"]
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


def _cfg_get(cfg: Mapping[str, Any] | object, key: str, default: Any = None) -> Any:
    """
    作用：从 dict 或对象中统一读取配置。

    输入：
        cfg: 配置对象
        key: str 配置键
        default: 默认值
    输出：
        配置值
    """
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class TextEncoder(nn.Module):
    """
    作用：将文本编码为可直接输入 Transformer 的 token 表示。

    输入：
        text_inputs: list[str] 或 tokenizer 输出字典
    输出：
        dict，包含：
        - tokens: Tensor(B, L, C)
        - token_mask: BoolTensor(B, L)
    """

    def __init__(self, cfg: Mapping[str, Any] | object):
        super().__init__()
        self.cfg = cfg
        model_name = str(_cfg_get(cfg, "type", "roberta-base"))
        out_channels = int(_cfg_get(cfg, "out_channels", 256))
        self.max_length = int(_cfg_get(cfg, "max_length", 64))
        freeze = bool(_cfg_get(cfg, "freeze", False))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.backbone.config.hidden_size)
        self.out_channels = out_channels
        self.proj = nn.Linear(hidden_size, out_channels)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _tokenize_texts(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        """
        作用：将原始字符串列表转换为 tokenizer batch。

        输入：
            texts: 文本序列
        输出：
            dict[str, Tensor]，包含 input_ids、attention_mask 等字段
        """
        if not texts:
            raise ValueError("text_inputs must not be empty")
        return self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def _prepare_inputs(
            self,
            text_inputs: Sequence[str] | Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        作用：统一处理原始文本与预分词输入。

        输入：
            text_inputs: list[str] 或 dict[str, Tensor]
        输出：
            dict[str, Tensor]，位于当前模块设备上
        """
        if isinstance(text_inputs, Mapping):
            encoded = dict(text_inputs)
        else:
            encoded = self._tokenize_texts(text_inputs)

        device = self.proj.weight.device
        prepared = {}
        for key, value in encoded.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"tokenized input field {key} must be a torch.Tensor")
            prepared[key] = value.to(device)
        return prepared

    def forward(
            self,
            text_inputs: Sequence[str] | Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        作用：执行文本编码并导出 Transformer 友好的 token。

        输入：
            text_inputs: list[str] 或 tokenizer 输出字典
        输出：
            dict，包含 tokens、token_mask
        """
        encoded = self._prepare_inputs(text_inputs)
        backbone_outputs = self.backbone(**encoded)
        hidden_state = backbone_outputs.last_hidden_state.to(torch.float32)
        tokens = self.proj(hidden_state)

        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones(
                tokens.shape[:2],
                dtype=torch.long,
                device=tokens.device,
            )
        token_mask = attention_mask.to(torch.bool)

        return {
            "tokens": tokens,
            "token_mask": token_mask,
        }
