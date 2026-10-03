from pathlib import Path
import json


def iter_shards(shards_dir: Path):
    """Итерируется по шардам строго по порядку и отдаёт примеры."""
    shard_files = sorted(shards_dir.glob("shard-*.jsonl"))

    for shard_path in shard_files:
        with shard_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def read_single_shard(shard_path: Path) -> list[dict]:
    samples = []
    with shard_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


if __name__ == "__main__":
    shards_dir = Path("data/shards/train")
    if shards_dir.exists():
        print("=== Чтение потока шардов ===")
        count = sum(1 for _ in iter_shards(shards_dir))
        print(f"Всего успешно прочитано из всех шардов: {count} примеров.")
