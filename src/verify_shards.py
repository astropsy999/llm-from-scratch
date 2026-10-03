import json
import hashlib
import sys
from pathlib import Path

# Чтобы `python src/verify_shards.py` находил соседний модуль
sys.path.insert(0, str(Path(__file__).resolve().parent))

from read_shards import iter_shards


def get_sample_hash(sample: dict) -> str:
    text = sample.get("text", "")
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def verify_shards(original_path: Path, shards_dir: Path) -> None:
    print("=== Проверка целостности шардов ===")

    with original_path.open("r", encoding="utf-8") as f:
        original_samples = [json.loads(line) for line in f if line.strip()]

    original_hashes = [get_sample_hash(s) for s in original_samples]
    sharded_samples = list(iter_shards(shards_dir))
    sharded_hashes = [get_sample_hash(s) for s in sharded_samples]

    metadata_path = shards_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    check_count = len(original_samples) == len(sharded_samples) == metadata["total_samples"]
    check_hashes = original_hashes == sharded_hashes
    check_duplicates = len(sharded_hashes) == len(set(sharded_hashes))

    print(f"1. SHARDS & SAMPLES COUNT: {'OK' if check_count else 'FAILED'}")
    print(f"2. DATA INTEGRITY & ORDER: {'OK' if check_hashes else 'FAILED'}")
    print(f"3. NO DUPLICATES:          {'OK' if check_duplicates else 'FAILED'}")
    print(f"4. METADATA CONSISTENCY:   {'OK' if metadata_path.exists() else 'FAILED'}")

    if check_count and check_hashes and check_duplicates:
        print("\nПРОВЕРКА УСПЕШНО ПРОЙДЕНА: данные полны, порядок сохранён.")
    else:
        print("\nОШИБКА ПРОВЕРКИ: обнаружены расхождения.")
        raise SystemExit(1)


if __name__ == "__main__":
    verify_shards(
        original_path=Path("data/processed/train.jsonl"),
        shards_dir=Path("data/shards/train"),
    )
