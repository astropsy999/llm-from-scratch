"""Lesson 14: smoke test for GPTModel assembly + short random train."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from gpt import GPTConfig, GPTModel  # noqa: E402


def main() -> None:
    torch.manual_seed(42)
    config = GPTConfig(
        vocab_size=200,
        block_size=64,
        num_layers=2,
        num_heads=4,
        embed_dim=128,
        use_rope=True,
    )
    model = GPTModel(config)
    print("params:", sum(p.numel() for p in model.parameters()))

    x = torch.randint(0, config.vocab_size, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, config.vocab_size)
    print("logits", tuple(logits.shape))

    prompt = torch.randint(0, config.vocab_size, (1, 5))
    gen = model.generate(prompt, 10)
    print("gen before", gen.tolist())

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    model.train()
    for step in range(20):
        opt.zero_grad()
        xb = torch.randint(0, config.vocab_size, (4, 32))
        yb = torch.randint(0, config.vocab_size, (4, 32))
        logits = model(xb)
        B, T, V = logits.shape
        loss = crit(logits.view(B * T, V), yb.view(B * T))
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print(step, float(loss.detach()))

    path = Path("checkpoints")
    path.mkdir(exist_ok=True)
    ckpt = path / "gpt_checkpoint.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config.to_dict()},
        ckpt,
    )
    print("saved", ckpt)


if __name__ == "__main__":
    main()
