import json
from pathlib import Path

import webdataset as wds


def convert_jsonl_to_webdataset() -> None:
    jsonl_path = Path("data/processed/train.jsonl")
    output_dir = Path("data/webdataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    tar_path = output_dir / "train.tar"

    # Пишем примеры в tar-архив с помощью TarWriter
    with wds.TarWriter(str(tar_path)) as sink:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if line.strip():
                    sample = json.loads(line)
                    # Формируем уникальный ключ для каждого элемента
                    sink.write(
                        {
                            "__key__": f"sample_{idx:06d}",
                            "json": sample,
                        }
                    )

    print(f"WebDataset (tar) создан: {tar_path}")


if __name__ == "__main__":
    convert_jsonl_to_webdataset()
