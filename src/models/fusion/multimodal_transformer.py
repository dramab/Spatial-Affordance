"""
src/models/fusion/multimodal_transformer.py
-------------------------------------------
职责：统一三种模态的 token 表示，并构建共享的多模态 Transformer。

用法：
    from src.models.fusion.multimodal_transformer import (
        MultimodalDecoder,
        UnifiedMultimodalEncoder,
    )

    encoder = UnifiedMultimodalEncoder(
        point_dim=128,
        image_dim=256,
        text_dim=256,
        hidden_dim=256,
    )
    memory_dict = encoder(
        point_inputs=point_dict,
        image_inputs=image_dict,
        text_inputs=text_dict,
    )
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def _cfg_get(cfg: dict | object, key: str, default):
    """
    作用：从 dict 或对象中统一读取配置。

    输入：
        cfg: 配置字典或对象
        key: str 配置键名
        default: 默认值
    输出：
        配置值
    """
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class UnifiedMultimodalEncoder(nn.Module):
    """
    作用：将点云、图像、文本 token 投影到统一维度并编码为共享 memory。

    输入：
        point_inputs: dict 或 None，需包含 tokens、token_mask、token_pos
        image_inputs: dict 或 None，需包含 tokens、token_mask、token_pos
        text_inputs: dict 或 None，需包含 tokens、token_mask
    输出：
        dict，包含：
        - memory: Tensor(B, L, H)
        - memory_mask: BoolTensor(B, L)，True 表示有效 token
        - memory_pos: Tensor(B, L, H)
        - modality_lengths: dict[str, int]
    """

    def __init__(
            self,
            point_dim: int,
            image_dim: int,
            text_dim: int,
            hidden_dim: int = 256,
            num_layers: int = 3,
            num_heads: int = 8,
            dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.point_proj = nn.Linear(int(point_dim), self.hidden_dim)
        self.image_proj = nn.Linear(int(image_dim), self.hidden_dim)
        self.text_proj = nn.Linear(int(text_dim), self.hidden_dim)

        self.point_pos_proj = nn.Linear(3, self.hidden_dim)
        self.image_pos_proj = nn.Linear(2, self.hidden_dim)

        self.modality_embed = nn.Embedding(3, self.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=self.hidden_dim * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(num_layers),
        )
        self.norm = nn.LayerNorm(self.hidden_dim)

    def _prepare_branch(
            self,
            branch_name: str,
            branch_inputs: Optional[dict[str, torch.Tensor]],
            proj: nn.Module,
            pos_proj: Optional[nn.Module],
            modality_id: int) -> Optional[dict[str, torch.Tensor]]:
        """
        作用：统一整理单个模态的投影 token、mask 与位置编码。

        输入：
            branch_name: str 模态名称
            branch_inputs: 模态输入字典
            proj: 特征投影层
            pos_proj: 位置投影层，文本分支可为 None
            modality_id: int 模态编号
        输出：
            dict 或 None，包含 tokens、mask、length
        """
        if branch_inputs is None:
            return None

        tokens = branch_inputs["tokens"].to(torch.float32)
        token_mask = branch_inputs["token_mask"].to(torch.bool)
        if tokens.ndim != 3 or token_mask.ndim != 2:
            raise ValueError(f"{branch_name} tokens/token_mask shape is invalid")
        if tokens.shape[:2] != token_mask.shape:
            raise ValueError(f"{branch_name} tokens and token_mask must align on first two dims")

        projected_tokens = proj(tokens)
        _, seq_len, _ = projected_tokens.shape

        if branch_name == "text":
            fused_tokens = projected_tokens
        else:
            token_pos = branch_inputs["token_pos"].to(torch.float32)
            if token_pos.shape[:2] != token_mask.shape:
                raise ValueError(f"{branch_name} token_pos must align with token_mask")
            fused_tokens = projected_tokens + pos_proj(token_pos)

        modality_embed = self.modality_embed.weight[modality_id].view(1, 1, -1)
        fused_tokens = fused_tokens + modality_embed
        return {
            "tokens": fused_tokens,
            "mask": token_mask,
            "length": seq_len,
        }

    def forward(
            self,
            point_inputs: Optional[dict[str, torch.Tensor]] = None,
            image_inputs: Optional[dict[str, torch.Tensor]] = None,
            text_inputs: Optional[dict[str, torch.Tensor]] = None) -> dict[str, torch.Tensor | dict[str, int]]:
        """
        作用：执行统一多模态编码。

        输入：
            point_inputs: 点云 token 输入
            image_inputs: 图像 token 输入
            text_inputs: 文本 token 输入
        输出：
            dict，包含 memory、memory_mask、memory_pos、modality_lengths
        """
        branches = [
            ("point", point_inputs, self.point_proj, self.point_pos_proj, 0),
            ("image", image_inputs, self.image_proj, self.image_pos_proj, 1),
            ("text", text_inputs, self.text_proj, None, 2),
        ]

        prepared = []
        modality_lengths: dict[str, int] = {}
        for branch_name, branch_inputs, proj, pos_proj, modality_id in branches:
            branch = self._prepare_branch(
                branch_name=branch_name,
                branch_inputs=branch_inputs,
                proj=proj,
                pos_proj=pos_proj,
                modality_id=modality_id,
            )
            if branch is None:
                continue
            prepared.append(branch)
            modality_lengths[branch_name] = int(branch["length"])

        if not prepared:
            raise ValueError("at least one modality input must be provided")

        tokens = torch.cat([item["tokens"] for item in prepared], dim=1)
        memory_mask = torch.cat([item["mask"] for item in prepared], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~memory_mask)
        memory = self.norm(encoded)

        return {
            "memory": memory,
            "memory_mask": memory_mask,
            "memory_pos": tokens,
            "modality_lengths": modality_lengths,
        }


class MultimodalDecoder(nn.Module):
    """
    作用：基于共享 memory 执行 query 解码。

    输入：
        memory: Tensor(B, L, H) 编码后的共享 memory
        memory_mask: BoolTensor(B, L) True 表示有效 token
    输出：
        dict，包含：
        - decoder_tokens: Tensor(B, Q, H)
        - query_embed: Tensor(B, Q, H)
    """

    def __init__(
            self,
            hidden_dim: int = 256,
            num_layers: int = 3,
            num_heads: int = 8,
            dropout: float = 0.1,
            num_queries: int = 1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=self.hidden_dim * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=int(num_layers),
        )
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(
            self,
            memory: torch.Tensor,
            memory_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        作用：执行 query 解码。

        输入：
            memory: Tensor(B, L, H)
            memory_mask: BoolTensor(B, L)
        输出：
            dict，包含 decoder_tokens 与 query_embed
        """
        batch_size = int(memory.shape[0])
        query_embed = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        target = torch.zeros_like(query_embed)
        decoder_tokens = self.decoder(
            tgt=target + query_embed,
            memory=memory,
            memory_key_padding_mask=~memory_mask,
        )
        return {
            "decoder_tokens": self.norm(decoder_tokens),
            "query_embed": query_embed,
        }
