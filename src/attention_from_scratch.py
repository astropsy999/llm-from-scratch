"""Lesson 10: Scaled Dot-Product Attention on tiny matrices (no nn.MultiheadAttention)."""

from __future__ import annotations

import torch


def attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled Dot-Product Attention without ready-made attention modules."""
    d_k = Q.size(-1)
    scores = Q @ K.T
    scaled_scores = scores / (d_k**0.5)
    weights = torch.softmax(scaled_scores, dim=-1)
    output = weights @ V
    return output, weights


def run_experiments() -> None:
    Q = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    K = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    V = torch.tensor(
        [
            [10.0, 0.0],
            [0.0, 20.0],
            [30.0, 30.0],
        ]
    )

    d_k = Q.size(-1)
    scores = Q @ K.T
    scaled_scores = scores / (d_k**0.5)
    output, weights = attention(Q, K, V)

    print("Q shape:", tuple(Q.shape), "K shape:", tuple(K.shape), "V shape:", tuple(V.shape))
    print("scores (Q @ K.T):\n", scores)
    print("scaled_scores ( / sqrt(d_k) ):\n", scaled_scores)
    print("Attention Weights:\n", weights)
    print("Attention Output:\n", output)
    print("Row sums:", weights.sum(dim=-1))

    V_new = V * 2.0
    output_new, weights_new = attention(Q, K, V_new)
    print("weights same?", torch.allclose(weights, weights_new))
    print("output changed?", not torch.allclose(output, output_new))

    simple_average = V.mean(dim=0, keepdim=True).repeat(3, 1)
    print("Uniform mean of V:\n", simple_average)
    print("Attention output:\n", output)


if __name__ == "__main__":
    run_experiments()
