# Обучаем LLM с нуля

Код к практической серии статей: от лаборатории PyTorch до маленькой GPT.

- Статьи: [stuzhuk.page](https://stuzhuk.page/ru/blog/) / [stuzhuklab.ru](https://stuzhuklab.ru/blog/) (серия `llm-from-scratch`)
- Состояние после каждого урока — **git-тег** `lesson-01`, `lesson-02`, …
- Актуальный снимок: [`lesson-09`](https://github.com/astropsy999/llm-from-scratch/tree/lesson-09)

## Быстрый старт

```bash
git clone https://github.com/astropsy999/llm-from-scratch.git
cd llm-from-scratch
git checkout lesson-09

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Сборку `torch` при необходимости берите со [страницы PyTorch](https://pytorch.org/get-started/locally/).

### Урок 1 — лаборатория

```bash
python src/check_environment.py
python experiments/benchmark_cpu_gpu.py
```

Подробнее: [`lessons/01.md`](lessons/01.md).

### Урок 2 — текст в числа

```bash
python src/demo_text_to_numbers.py
```

Подробнее: [`lessons/02.md`](lessons/02.md).

### Урок 3 — BPE-токенизатор

```bash
python src/train_tokenizer.py
python src/test_tokenizer.py
```

Подробнее: [`lessons/03.md`](lessons/03.md).

### Урок 4 — датасет и JSONL

```bash
python src/prepare_dataset.py
```

Подробнее: [`lessons/04.md`](lessons/04.md).

### Урок 5 — форматы JSONL / Parquet / WebDataset

```bash
python src/convert_to_parquet.py
python src/convert_to_webdataset.py
python src/verify_formats.py
python src/benchmark_formats.py
```

Подробнее: [`lessons/05.md`](lessons/05.md).

### Урок 6 — шарды

```bash
python src/create_shards.py \
  --input data/processed/train.jsonl \
  --output data/shards/train \
  --shard-size 2
python src/read_shards.py
python src/verify_shards.py
```

Подробнее: [`lessons/06.md`](lessons/06.md).

### Урок 7 — символьная модель

```bash
python src/train_char_model.py
```

Подробнее: [`lessons/07.md`](lessons/07.md).

### Урок 8 — loss и CrossEntropy

```bash
python src/understand_loss.py
python src/train_char_model.py
```

Подробнее: [`lessons/08.md`](lessons/08.md).

### Урок 9 — TinyLanguageModel

```bash
python src/train_tiny_lm.py
```

Подробнее: [`lessons/09.md`](lessons/09.md).

## Структура

```text
src/            скрипты уроков
experiments/    замеры
data/           корпуса и данные
tokenizer/      сохранённый tokenizer.json
checkpoints/    веса моделей
notebooks/      черновики
lessons/        заметки по каждому уроку
```
