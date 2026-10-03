import json
import random
from pathlib import Path

# Параметры без «магических чисел» в середине кода
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
MIN_TEXT_LENGTH = 20        # Минимальная длина текста в символах
TRAIN_RATIO = 0.8           # 80% train, 20% validation
RANDOM_SEED = 42            # Фиксированный seed для воспроизводимости


def clean_text(text: str) -> str:
    """Шаг 1. Очистка и нормализация пробельных символов."""
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        # Схлопываем повторяющиеся пробелы внутри строки в один
        line_clean = " ".join(line.split())
        if line_clean:
            cleaned_lines.append(line_clean)

    # Объединяем очищенные строки через перенос
    return "\n".join(cleaned_lines)


def _write_jsonl(path: Path, samples: list[dict]) -> None:
    """Записывает список словарей в файл JSONL (по одному JSON на строку)."""
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def _print_report(
    files_cnt: int,
    raw_cnt: int,
    usable_cnt: int,
    train: list[dict],
    val: list[dict],
) -> None:
    """Выводит итоговый отчёт по датасету."""
    combined = train + val
    avg_len = sum(len(s["text"]) for s in combined) / max(len(combined), 1)
    print("\n" + "=" * 35)
    print("        DATASET REPORT        ")
    print("=" * 35)
    print(f"Raw documents:       {files_cnt}")
    print(f"Raw paragraphs:      {raw_cnt}")
    print(f"Usable samples:      {usable_cnt}")
    print(f"Train samples:       {len(train)}")
    print(f"Validation samples:  {len(val)}")
    print(f"Avg chars/sample:    {avg_len:.1f}")
    print("=" * 35)


def load_jsonl(path: Path) -> list[dict]:
    """Читает JSONL файл построчно."""
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def process_dataset() -> None:
    print("=== Старт предобработки датасета ===")

    raw_files = list(RAW_DATA_DIR.glob("*.txt"))
    print(f"Найдено исходных файлов: {len(raw_files)}")

    raw_texts_count = 0
    short_filtered_count = 0

    seen_hashes: set[str] = set()
    unique_samples: list[dict] = []

    # 1. Чтение, очистка, фильтрация и удаление дубликатов
    for file_path in raw_files:
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()

        # Разбиваем документ на логические абзацы/фрагменты
        paragraphs = content.split("\n\n")
        raw_texts_count += len(paragraphs)

        for p in paragraphs:
            cleaned = clean_text(p)

            # Фильтрация по длине
            if len(cleaned) < MIN_TEXT_LENGTH:
                short_filtered_count += 1
                continue

            # Точное удаление дубликатов через set()
            if cleaned in seen_hashes:
                continue

            seen_hashes.add(cleaned)
            unique_samples.append({"text": cleaned})

    duplicates_removed = (
        raw_texts_count - short_filtered_count - len(unique_samples)
    )

    print(f"Прочитано фрагментов: {raw_texts_count}")
    print(f"Отфильтровано коротких/пустых: {short_filtered_count}")
    print(f"Удалено точных дубликатов: {duplicates_removed}")
    print(f"Итого уникальных примеров: {len(unique_samples)}")

    # 2. Воспроизводимое разбиение train / validation
    random.seed(RANDOM_SEED)
    random.shuffle(unique_samples)

    train_size = int(len(unique_samples) * TRAIN_RATIO)
    train_samples = unique_samples[:train_size]
    val_samples = unique_samples[train_size:]

    # 3. Сохранение в JSONL
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DATA_DIR / "train.jsonl"
    val_path = PROCESSED_DATA_DIR / "validation.jsonl"

    _write_jsonl(train_path, train_samples)
    _write_jsonl(val_path, val_samples)

    print(f"\nСохранено train примеров: {len(train_samples)} -> {train_path}")
    print(f"Сохранено validation примеров: {len(val_samples)} -> {val_path}")

    _print_report(
        len(raw_files),
        raw_texts_count,
        len(unique_samples),
        train_samples,
        val_samples,
    )

    # Проверка построчного чтения
    loaded_train = load_jsonl(train_path)
    print(f"Успешно прочитано из JSONL: {len(loaded_train)} примеров.")
    if loaded_train:
        print("Первый пример:", loaded_train[0])


if __name__ == "__main__":
    process_dataset()
