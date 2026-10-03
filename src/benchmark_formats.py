import json
import time
from pathlib import Path

import pyarrow.parquet as pq
import webdataset as wds


def measure_jsonl(path: Path) -> tuple[int, float]:
    start = time.perf_counter()
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                _ = json.loads(line)
                count += 1
    elapsed = time.perf_counter() - start
    return count, elapsed


def measure_parquet(path: Path) -> tuple[int, float]:
    start = time.perf_counter()
    table = pq.read_table(path)
    count = table.num_rows
    _ = table["text"].to_pylist()
    elapsed = time.perf_counter() - start
    return count, elapsed


def measure_webdataset(path: Path) -> tuple[int, float]:
    start = time.perf_counter()
    count = 0
    dataset = wds.WebDataset(str(path), shardshuffle=False).decode()
    for sample in dataset:
        payload = sample["json"]
        if isinstance(payload, (bytes, bytearray, str)):
            _ = json.loads(payload)
        count += 1
    elapsed = time.perf_counter() - start
    return count, elapsed


def run_benchmark() -> None:
    jsonl_path = Path("data/processed/train.jsonl")
    parquet_path = Path("data/parquet/train.parquet")
    tar_path = Path("data/webdataset/train.tar")

    # Размеры файлов в КБ
    size_jsonl = jsonl_path.stat().st_size / 1024
    size_parquet = parquet_path.stat().st_size / 1024
    size_tar = tar_path.stat().st_size / 1024

    # 1. Холодный проход (меньше влияния кэша ОС)
    cnt_j, t_jsonl_cold = measure_jsonl(jsonl_path)
    cnt_p, t_parquet_cold = measure_parquet(parquet_path)
    cnt_w, t_wds_cold = measure_webdataset(tar_path)

    # 2. Тёплый проход (влияние кэша ОС)
    _, t_jsonl_warm = measure_jsonl(jsonl_path)
    _, t_parquet_warm = measure_parquet(parquet_path)
    _, t_wds_warm = measure_webdataset(tar_path)

    # Пропускная способность (примеров в секунду) по тёплому проходу
    tp_jsonl = cnt_j / max(t_jsonl_warm, 1e-9)
    tp_parquet = cnt_p / max(t_parquet_warm, 1e-9)
    tp_wds = cnt_w / max(t_wds_warm, 1e-9)

    report = (
        f"{'Format':<12} | {'Size (KB)':<10} | {'Cold Read (s)':<13} | "
        f"{'Warm Read (s)':<13} | {'Throughput (samples/s)':<22}\n"
        + "-" * 80
        + "\n"
        f"{'JSONL':<12} | {size_jsonl:<10.2f} | {t_jsonl_cold:<13.4f} | "
        f"{t_jsonl_warm:<13.4f} | {tp_jsonl:<22.1f}\n"
        f"{'Parquet':<12} | {size_parquet:<10.2f} | {t_parquet_cold:<13.4f} | "
        f"{t_parquet_warm:<13.4f} | {tp_parquet:<22.1f}\n"
        f"{'WebDataset':<12} | {size_tar:<10.2f} | {t_wds_cold:<13.4f} | "
        f"{t_wds_warm:<13.4f} | {tp_wds:<22.1f}\n"
    )

    print("\n=== BENCHMARK REPORT ===")
    print(report)

    exp_dir = Path("experiments")
    exp_dir.mkdir(exist_ok=True)
    with (exp_dir / "format_benchmark.txt").open("w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    run_benchmark()
