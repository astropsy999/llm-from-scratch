from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

SPECIAL = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
CORPUS = ["data/tokenizer_corpus.txt"]
PHRASE = "Обучение языковой модели с нуля — это увлекательный инженерный процесс."


def train_and_save(vocab_size: int, save_path: Path) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL)
    tokenizer.train(CORPUS, trainer)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path))
    return tokenizer


def report(label: str, requested: int, tokenizer: Tokenizer) -> None:
    output = tokenizer.encode(PHRASE)
    print(f"=== {label} (запрошено vocab_size={requested}) ===")
    print("фактический размер словаря:", tokenizer.get_vocab_size())
    print("фраза:", PHRASE)
    print("tokens:", output.tokens)
    print("длина ids:", len(output.ids))
    print()


def main() -> None:
    small = train_and_save(100, Path("tokenizer/tokenizer_v100.json"))
    large = train_and_save(500, Path("tokenizer/tokenizer_v500.json"))
    report("маленький словарь", 100, small)
    report("большой словарь", 500, large)


if __name__ == "__main__":
    main()
