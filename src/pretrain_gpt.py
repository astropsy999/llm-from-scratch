"""Lesson 15: pretrain GPTModel on processed JSONL + BPE tokenizer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpt import GPTConfig, GPTModel  # noqa: E402


class TokenDataset(Dataset):
    def __init__(self, token_ids: list[int], block_size: int):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.token_ids) - self.block_size)

    def __getitem__(self, idx: int):
        chunk = self.token_ids[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def load_texts(path: Path) -> list[str]:
    texts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"])
    return texts


def texts_to_ids(tokenizer: Tokenizer, texts: list[str]) -> list[int]:
    eos_id = tokenizer.token_to_id("[EOS]")
    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer.encode(text).ids)
        if eos_id is not None:
            ids.append(eos_id)
    return ids


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    max_batches: int = 10,
) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        B, T, V = logits.shape
        total += criterion(logits.view(B * T, V), y.view(B * T)).item()
        n += 1
        if n >= max_batches:
            break
    model.train()
    return total / max(n, 1)


def train_step(model, x, y, criterion, optimizer) -> float:
    optimizer.zero_grad()
    logits = model(x)
    B, T, V = logits.shape
    loss = criterion(logits.view(B * T, V), y.view(B * T))
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def decode_ids(tokenizer: Tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids)


def main() -> None:
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size()

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=32,
        num_layers=2,
        num_heads=4,
        embed_dim=64,
        use_rope=True,
    )
    assert tokenizer.get_vocab_size() == config.vocab_size
    print("vocab_size", config.vocab_size, "device", device)

    train_texts = load_texts(ROOT / "data" / "processed" / "train.jsonl")
    val_texts = load_texts(ROOT / "data" / "processed" / "validation.jsonl")
    train_ids = texts_to_ids(tokenizer, train_texts)
    val_ids = texts_to_ids(tokenizer, val_texts)
    print("train tokens", len(train_ids), "val tokens", len(val_ids))

    train_ds = TokenDataset(train_ids, config.block_size)
    val_ds = TokenDataset(val_ids, config.block_size)
    if len(train_ds) == 0:
        raise RuntimeError("train TokenDataset пуст: увеличьте корпус или уменьшите block_size")
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False) if len(val_ds) else None

    model = GPTModel(config).to(device)
    print("params:", sum(p.numel() for p in model.parameters()))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    bos = tokenizer.token_to_id("[BOS]")
    prompt = torch.tensor(
        [[bos if bos is not None else 1]],
        dtype=torch.long,
        device=device,
    )
    model.eval()
    before = model.generate(prompt, max_new_tokens=20)[0].tolist()
    print("gen before:", decode_ids(tokenizer, before))

    max_steps = 200
    log_every = 50
    train_iter = iter(train_loader)
    model.train()
    last_loss = 0.0
    for step in range(1, max_steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)
        last_loss = train_step(model, x, y, criterion, optimizer)

        if step % log_every == 0 or step == 1:
            train_l = estimate_loss(model, train_loader, criterion, device)
            val_l = (
                estimate_loss(model, val_loader, criterion, device)
                if val_loader is not None and len(val_ds) > 0
                else float("nan")
            )
            print(f"step {step:04d} batch_loss={last_loss:.4f} train={train_l:.4f} val={val_l:.4f}")

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt = ckpt_dir / f"gpt-small-step-{max_steps:04d}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "step": max_steps,
        },
        ckpt,
    )
    print("saved", ckpt)

    model.eval()
    after = model.generate(prompt, max_new_tokens=20)[0].tolist()
    print("gen after:", decode_ids(tokenizer, after))


if __name__ == "__main__":
    main()
