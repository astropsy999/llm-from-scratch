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


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        # idx: [batch_size, block_size]
        tok_emb = self.token_embedding(idx)  # [B, T, C]
        logits = self.lm_head(tok_emb)       # [B, T, vocab]
        return logits


def generate_text(model, dataset, prompt: str, max_new_tokens: int = 20) -> str:
    model.eval()
    context_ids = [dataset.stoi.get(c, 0) for c in prompt]
    x = torch.tensor([context_ids], dtype=torch.long)

    for _ in range(max_new_tokens):
        x_cond = x[:, -dataset.block_size :]
        with torch.no_grad():
            logits = model(x_cond)
            last_logits = logits[0, -1, :]
            next_id = torch.argmax(last_logits, dim=-1).item()

        x = torch.cat([x, torch.tensor([[next_id]])], dim=1)

    return "".join(dataset.itos[idx.item()] for idx in x[0])


def main():
    torch.manual_seed(42)

    text_corpus = (
        "Привет, мир! Это наша первая маленькая языковая модель. "
        "Привет всем разработчикам!"
    )
    block_size = 8
    batch_size = 4

    dataset = CharDataset(text_corpus, block_size=block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BigramLanguageModel(vocab_size=dataset.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    print(f"Размер словаря: {dataset.vocab_size}")
    print(f"Обучающих окон: {len(dataset)}")
    x0, y0 = dataset[0]
    print(f"Форма X/Y одного окна: {tuple(x0.shape)} / {tuple(y0.shape)}")

    print("\n=== 1. Генерация ДО обучения ===")
    prompt = "Прив"
    print(f"Промпт: '{prompt}' -> Результат: '{generate_text(model, dataset, prompt)}'\n")

    print("=== 2. Старт обучения ===")
    model.train()
    loss_history: list[float] = []
    for epoch in range(150):
        total_loss = 0.0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(
                logits.view(-1, dataset.vocab_size),
                y_batch.view(-1),
            )
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
    print(f"Промпт: '{prompt}' -> Результат: '{generate_text(model, dataset, prompt)}'\n")

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/char_model.pt")
    print("Модель сохранена в checkpoints/char_model.pt")


if __name__ == "__main__":
    main()
