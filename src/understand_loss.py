import math
import torch
import torch.nn as nn


def demonstrate_loss_mechanics() -> None:
    print("=== 1. Ручной расчёт Loss ===")
    for p in [0.99, 0.90, 0.50, 0.10, 0.01]:
        print(f"p_correct = {p:.2f} -> Loss = {-math.log(p):.4f}")

    print("\n=== 2. Сравнение с PyTorch CrossEntropyLoss ===")
    # shape [1, vocab_size]
    logits = torch.tensor([[1.2, -0.4, 2.5, 0.3]])
    # индекс символа "в"
    target = torch.tensor([2], dtype=torch.long)

    probs = torch.softmax(logits, dim=-1)
    p_correct = probs[0, target.item()].item()
    manual_ce = -math.log(p_correct)

    criterion = nn.CrossEntropyLoss()
    pytorch_ce = criterion(logits, target).item()

    print(f"Вероятность правильного класса (индекс {target.item()}): {p_correct:.4f}")
    print(f"Ручной Loss (-log):         {manual_ce:.4f}")
    print(f"PyTorch CrossEntropyLoss:   {pytorch_ce:.4f}")

    assert math.isclose(manual_ce, pytorch_ce, rel_tol=1e-5)
    print("Ручной расчёт и PyTorch CrossEntropyLoss совпали.")

    print("\n=== 3. Формы для последовательностей ===")
    torch.manual_seed(42)
    batch_size, block_size, vocab_size = 4, 8, 27
    logits_seq = torch.randn(batch_size, block_size, vocab_size)
    targets_seq = torch.randint(0, vocab_size, (batch_size, block_size))
    logits_flat = logits_seq.view(batch_size * block_size, vocab_size)
    targets_flat = targets_seq.view(batch_size * block_size)
    seq_loss = criterion(logits_flat, targets_flat)
    print(f"logits:        {tuple(logits_seq.shape)}")
    print(f"targets:       {tuple(targets_seq.shape)}")
    print(f"logits_flat:   {tuple(logits_flat.shape)}")
    print(f"targets_flat:  {tuple(targets_flat.shape)}")
    print(f"sequence loss: {seq_loss.item():.4f}")


if __name__ == "__main__":
    demonstrate_loss_mechanics()
