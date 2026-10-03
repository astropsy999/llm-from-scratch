"""Lesson 13: Pre-Norm Transformer block and stack."""

from __future__ import annotations

import torch
import torch.nn as nn

from multi_head_attention import MultiHeadCausalAttention


class MLP(nn.Module):
    def __init__(self, embed_dim: int, expansion_factor: int = 4):
        super().__init__()
        hidden = embed_dim * expansion_factor
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadCausalAttention(
            embed_dim,
            num_heads,
            use_rope=use_rope,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerStack(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    use_rope=use_rope,
                    rope_theta=rope_theta,
                    max_seq_len=max_seq_len,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def run_tests() -> None:
    torch.manual_seed(42)
    B, T, D, H = 2, 8, 32, 4
    x = torch.randn(B, T, D)
    block = TransformerBlock(D, H)
    out = block(x)
    assert out.shape == x.shape
    print("block OK", tuple(out.shape))
    print("params", sum(p.numel() for p in block.parameters()))

    vocab, layers = 100, 3
    emb = nn.Embedding(vocab, D)
    stack = TransformerStack(layers, D, H)
    ids = torch.randint(0, vocab, (B, T))
    y = stack(emb(ids))
    assert y.shape == (B, T, D)
    print("stack OK", tuple(y.shape))


if __name__ == "__main__":
    run_tests()
