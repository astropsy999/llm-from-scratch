# Обучаем LLM с нуля

Код к практической серии статей: от лаборатории PyTorch до маленькой GPT.

- Статьи: [stuzhuk.page](https://stuzhuk.page/ru/blog/) / [stuzhuklab.ru](https://stuzhuklab.ru/blog/) (серия `llm-from-scratch`)
- Состояние после каждого урока — **git-тег** `lesson-01`, `lesson-02`, …
- Актуальный снимок: [`lesson-03`](https://github.com/astropsy999/llm-from-scratch/tree/lesson-03)

## Быстрый старт

```bash
git clone https://github.com/astropsy999/llm-from-scratch.git
cd llm-from-scratch
git checkout lesson-03

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

## Структура

```text
src/            скрипты уроков
experiments/    замеры
data/           корпуса и данные
tokenizer/      сохранённый tokenizer.json
notebooks/      черновики
lessons/        заметки по каждому уроку
```
