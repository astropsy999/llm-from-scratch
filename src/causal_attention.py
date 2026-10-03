"""Lesson 11: causal (masked) Scaled Dot-Product Attention — no future peeking."""

from __future__ import annotations

import torch


def causal_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled Dot-Product Attention with a lower-triangular causal mask."""
    T, d_k = Q.size(0), Q.size(-1)
    scores = Q @ K.T
    scaled_scores = scores / (d_k**0.5)
    mask = torch.tril(torch.ones(T, T))
    masked_scores = scaled_scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(masked_scores, dim=-1)
    output = weights @ V
    return output, weights


def test_causal_attention() -> None:
    torch.manual_seed(42)
    T, d_k = 4, 2
    Q = torch.randn(T, d_k)
    K = torch.randn(T, d_k)
    V = torch.randn(T, d_k)

    print("Q shape:", tuple(Q.shape), "K shape:", tuple(K.shape), "V shape:", tuple(V.shape))

    mask = torch.tril(torch.ones(T, T))
    print("Causal mask (tril):\n", mask)

    scores = Q @ K.T
    scaled_scores = scores / (d_k**0.5)
    print("scaled_scores (before mask):\n", scaled_scores)

    masked_scores = scaled_scores.masked_fill(mask == 0, float("-inf"))
    print("masked_scores:\n", masked_scores)

    output, weights = causal_attention(Q, K, V)
    print("Attention Weights:\n", weights)
    print("Row sums:", weights.sum(dim=-1))
    print("Attention Output:\n", output)

    upper = weights.triu(diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper))
    assert torch.allclose(weights[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    print("OK: future blocked; Q1 sees only K1")
    print("output shape:", tuple(output.shape))


if __name__ == "__main__":
    test_causal_attention()
