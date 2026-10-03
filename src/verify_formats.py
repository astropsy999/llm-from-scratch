import json
from pathlib import Path

import pyarrow.parquet as pq
import webdataset as wds


def _as_text(payload) -> str:
    if isinstance(payload, (bytes, bytearray, str)):
        payload = json.loads(payload)
    return payload["text"]


def verify_all_formats() -> None:
    jsonl_path = Path("data/processed/train.jsonl")
    parquet_path = Path("data/parquet/train.parquet")
    tar_path = Path("data/webdataset/train.tar")

    # 1. Читаем JSONL
    with jsonl_path.open("r", encoding="utf-8") as f:
        jsonl_texts = [
            json.loads(line)["text"] for line in f if line.strip()
        ]

    # 2. Читаем Parquet
    parquet_texts = pq.read_table(parquet_path)["text"].to_pylist()

    # 3. Читаем WebDataset по одному примеру
    wds_texts = []
    dataset = wds.WebDataset(str(tar_path), shardshuffle=False).decode()
    for sample in dataset:
        wds_texts.append(_as_text(sample["json"]))

    print(f"Примеров в JSONL:      {len(jsonl_texts)}")
    print(f"Примеров в Parquet:    {len(parquet_texts)}")
    print(f"Примеров в WebDataset: {len(wds_texts)}")

    if jsonl_texts:
        print(f"Первый текст (JSONL): {jsonl_texts[0]!r}")

    assert len(jsonl_texts) == len(parquet_texts) == len(wds_texts), (
        "Число примеров не совпало — пересоберите Parquet/WebDataset "
        "из актуального train.jsonl"
    )
    assert jsonl_texts == parquet_texts == wds_texts, (
        "Тексты или их порядок не совпали — проверьте конвертеры "
        "и shardshuffle=False"
    )

    print("\nПроверка успешна: все три формата содержат одинаковые данные!")


if __name__ == "__main__":
    verify_all_formats()
