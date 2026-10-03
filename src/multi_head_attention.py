"""Lesson 12–14: multi-head causal attention (+ optional RoPE)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MultiHeadCausalAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) должно нацело делиться на num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self.rope = None

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if use_rope:
            from rope import RotaryEmbedding

            self.rope = RotaryEmbedding(
                self.head_dim, max_seq_len=max_seq_len, theta=rope_theta
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, hd = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, hd).transpose(1, 2)

        if self.use_rope:
            from rope import apply_rotary_pos_emb

            cos, sin = self.rope(T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


def run_tests() -> None:
    torch.manual_seed(42)
    B, T, D, H = 2, 4, 8, 2
    x = torch.randn(B, T, D)
    mha = MultiHeadCausalAttention(embed_dim=D, num_heads=H)

    print("config: B,T,D,H =", B, T, D, H, "head_dim =", mha.head_dim)
    print("input shape:", tuple(x.shape))

    with torch.no_grad():
        q = mha.q_proj(x).view(B, T, H, mha.head_dim).transpose(1, 2)
        k = mha.k_proj(x).view(B, T, H, mha.head_dim).transpose(1, 2)
        print("q after view/transpose:", tuple(q.shape))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(mha.head_dim)
        print("scores shape:", tuple(scores.shape))

    out = mha(x)
    assert out.shape == x.shape
    print("shapes OK", tuple(x.shape), "→", tuple(out.shape))

    try:
        MultiHeadCausalAttention(embed_dim=10, num_heads=3)
    except ValueError as e:
        print("config check OK:", e)

    print("params:", sum(p.numel() for p in mha.parameters()))


if __name__ == "__main__":
    run_tests()
