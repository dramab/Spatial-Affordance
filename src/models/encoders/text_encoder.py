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

from typing import Mapping, Sequence

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from src.models.common import cfg_get


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
        model_name = str(cfg_get(cfg, "type", "roberta-base"))
        out_channels = int(cfg_get(cfg, "out_channels", 256))
        self.max_length = int(cfg_get(cfg, "max_length", 64))
        freeze = bool(cfg_get(cfg, "freeze", False))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.backbone.config.hidden_size)
        self.out_channels = out_channels
        self.proj = nn.Linear(hidden_size, out_channels)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

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
        if isinstance(text_inputs, Mapping):
            encoded = dict(text_inputs)
        else:
            if not text_inputs:
                raise ValueError("text_inputs must not be empty")
            encoded = self.tokenizer(
                list(text_inputs),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        device = self.proj.weight.device
        for key, value in encoded.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"tokenized input field {key} must be a torch.Tensor")
            encoded[key] = value.to(device)

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
