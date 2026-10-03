import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def convert_jsonl_to_parquet() -> None:
    jsonl_path = Path("data/processed/train.jsonl")
    output_dir = Path("data/parquet")
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "train.parquet"

    # 1. Читаем исходные примеры из JSONL
    samples = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    # 2. Преобразуем в PyArrow Table
    texts = [s["text"] for s in samples]
    table = pa.Table.from_arrays([pa.array(texts)], names=["text"])

    # 3. Сохраняем в Parquet
    pq.write_table(table, parquet_path)
    print(f"Parquet создан: {parquet_path} (примеров: {len(samples)})")


if __name__ == "__main__":
    convert_jsonl_to_parquet()
