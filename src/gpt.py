"""Lesson 14: small decoder-only GPTModel."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from transformer_block import TransformerBlock


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    embed_dim: int = 128
    rope_theta: float = 10000.0
    use_rope: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GPTConfig":
        return cls(**data)


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        if not config.use_rope:
            self.position_embedding = nn.Embedding(
                config.block_size, config.embed_dim
            )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.embed_dim,
                    config.num_heads,
                    use_rope=config.use_rope,
                    rope_theta=config.rope_theta,
                    max_seq_len=config.block_size,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        if T > self.config.block_size:
            raise ValueError(
                f"Длина последовательности ({T}) превышает block_size ({self.config.block_size})"
            )
        x = self.token_embedding(input_ids)
        if not self.config.use_rope:
            pos = torch.arange(0, T, device=input_ids.device)
            x = x + self.position_embedding(pos)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            cond = input_ids[:, -self.config.block_size :]
            logits = self(cond)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_id), dim=1)
        return input_ids
