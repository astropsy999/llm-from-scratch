from pathlib import Path

import pyarrow.parquet as pq


def read_texts(path: Path = Path("data/parquet/train.parquet")) -> list[str]:
    return pq.read_table(path)["text"].to_pylist()


if __name__ == "__main__":
    texts = read_texts()
    print(len(texts), texts[0] if texts else None)
