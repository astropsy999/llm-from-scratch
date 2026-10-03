import argparse
import json
from pathlib import Path


def create_shards(input_path: Path, output_dir: Path, shard_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    shard_index = 0
    buffer: list[dict] = []

    print(f"=== Старт шардирования {input_path} ===")
    print(f"Целевой размер шарда: {shard_size} примеров\n")

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            sample = json.loads(line)
            buffer.append(sample)
            total_samples += 1

            if len(buffer) == shard_size:
                _write_shard(output_dir, shard_index, buffer)
                shard_index += 1
                buffer.clear()

    # ОБЯЗАТЕЛЬНО: последний неполный шард
    if buffer:
        _write_shard(output_dir, shard_index, buffer)
        shard_index += 1
        buffer.clear()

    metadata = {
        "total_samples": total_samples,
        "shard_size": shard_size,
        "num_shards": shard_index,
        "source_file": input_path.name,
    }

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nУспешно создано шардов: {shard_index}")
    print(f"Всего обработано примеров: {total_samples}")
    print(f"Метаданные сохранены в: {metadata_path}")


def _write_shard(output_dir: Path, shard_index: int, samples: list[dict]) -> None:
    # Ведущие нули гарантируют правильную сортировку имён
    filename = f"shard-{shard_index:05d}.jsonl"
    shard_path = output_dir / filename

    with shard_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Записан {filename}: {len(samples)} примеров")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Шардирование текстового JSONL датасета")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=1000)
    args = parser.parse_args()
    create_shards(args.input, args.output, args.shard_size)
