import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset


class CharDataset(Dataset):
    def __init__(self, text: str, block_size: int):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.data = [self.stoi[c] for c in text]
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class TinyLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: [B, T]
        x = self.embedding(idx)  # [B, T, E]
        x = self.fc1(x)  # [B, T, H]
        x = self.activation(x)  # [B, T, H]
        logits = self.fc2(x)  # [B, T, V]
        return logits


def encode(text: str, stoi: dict) -> list[int]:
    return [stoi[c] for c in text]


def decode(ids: list[int], itos: dict) -> str:
    return "".join(itos[i] for i in ids)


def predict_next_char(
    model: nn.Module,
    context_ids: list[int],
    block_size: int,
) -> int:
    model.eval()
    cond_ids = context_ids[-block_size:]
    x = torch.tensor([cond_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)  # [1, T, V]
        next_id = torch.argmax(logits[0, -1, :], dim=-1).item()
    return next_id


def generate(
    model: nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    block_size: int,
) -> list[int]:
    result = list(prompt_ids)
    for _ in range(max_new_tokens):
        next_id = predict_next_char(model, result, block_size)
        result.append(next_id)
    return result


def generate_text(
    model: nn.Module,
    dataset: CharDataset,
    prompt: str,
    max_new_tokens: int = 40,
) -> str:
    prompt_ids = encode(prompt, dataset.stoi)
    out_ids = generate(model, prompt_ids, max_new_tokens, dataset.block_size)
    return decode(out_ids, dataset.itos)


def main() -> None:
    torch.manual_seed(42)

    text = (
        "Обучение языковой модели с нуля — инженерный процесс. "
        "Сначала данные, потом модель, затем проверка генерации. "
        "Обучение требует внимания к loss и формам тензоров."
    )
    block_size = 8
    batch_size = 4
    embedding_dim = 32
    hidden_dim = 64
    epochs = 150

    dataset = CharDataset(text, block_size=block_size)
    assert decode(encode(text, dataset.stoi), dataset.itos) == text

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = TinyLanguageModel(
        vocab_size=dataset.vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Размер словаря: {dataset.vocab_size}")
    print(f"Обучающих окон: {len(dataset)}")
    print(f"Total parameters: {total_params}")

    x0, y0 = dataset[0]
    print(f"Форма X/Y одного окна: {tuple(x0.shape)} / {tuple(y0.shape)}")
    xb, yb = next(iter(dataloader))
    print(f"Форма батча X/Y: {tuple(xb.shape)} / {tuple(yb.shape)}")
    with torch.no_grad():
        logits0 = model(xb)
    print(f"Форма logits: {tuple(logits0.shape)}")

    prompt = "Обуч"
    print("\n=== 1. Генерация ДО обучения ===")
    print(f"Промпт: '{prompt}' -> '{generate_text(model, dataset, prompt)}'")

    # Один шаг на копии: доказать, что веса реально меняются
    demo_model = TinyLanguageModel(
        vocab_size=dataset.vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    )
    demo_model.load_state_dict(model.state_dict())
    demo_opt = torch.optim.AdamW(demo_model.parameters(), lr=1e-2)
    weight_before = demo_model.fc1.weight.detach().clone()
    demo_opt.zero_grad()
    logits = demo_model(xb)
    b, t, v = logits.shape
    loss = criterion(logits.view(b * t, v), yb.view(b * t))
    loss.backward()
    demo_opt.step()
    assert not torch.equal(weight_before, demo_model.fc1.weight.detach())
    print("Подтверждено: параметры изменились после optimizer.step()")

    print("\n=== 2. Старт обучения ===")
    model.train()
    loss_history: list[float] = []
    for epoch in range(epochs):
        total_loss = 0.0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            logits = model(x_batch)
            b, t, v = logits.shape
            loss = criterion(logits.view(b * t, v), y_batch.view(b * t))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        if epoch % 30 == 0:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}")

    print("\n=== График падения Loss ===")
    for i in range(0, len(loss_history), 15):
        bars = "█" * int(loss_history[i] * 5)
        print(f"Epoch {i:3d} | Loss: {loss_history[i]:.4f} | {bars}")

    print("\n=== 3. Генерация ПОСЛЕ обучения ===")
    print(f"Промпт: '{prompt}' -> '{generate_text(model, dataset, prompt)}'")

    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)
    save_path = checkpoints_dir / "tiny_lm.pt"
    torch.save(model.state_dict(), save_path)

    reloaded = TinyLanguageModel(
        vocab_size=dataset.vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    )
    reloaded.load_state_dict(torch.load(save_path, weights_only=True))
    print(f"\nМодель сохранена в {save_path}")
    print(
        "Проверка load_state_dict: "
        f"'{generate_text(reloaded, dataset, prompt, max_new_tokens=20)}'"
    )


if __name__ == "__main__":
    main()
