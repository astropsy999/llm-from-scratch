from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def train_my_tokenizer() -> None:
    # [UNK] служит запасным вариантом для незнакомых элементов.
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Предварительно размечаем границы слов и пунктуации.
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=300,
        special_tokens=[
            "[UNK]",  # Неизвестный элемент
            "[PAD]",  # Выравнивание длины
            "[BOS]",  # Начало последовательности
            "[EOS]",  # Конец последовательности
        ],
    )

    corpus_files = ["data/tokenizer_corpus.txt"]
    tokenizer.train(corpus_files, trainer)

    output_dir = Path("tokenizer")
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "tokenizer.json"
    tokenizer.save(str(save_path))

    print(f"Токенизатор успешно обучен и сохранён в {save_path}")


if __name__ == "__main__":
    train_my_tokenizer()
