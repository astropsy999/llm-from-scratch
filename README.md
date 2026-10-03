# Обучаем LLM с нуля

Код к практической серии статей: от лаборатории PyTorch до маленькой GPT.

- Статьи: [stuzhuk.page](https://stuzhuk.page/ru/blog/) / [stuzhuklab.ru](https://stuzhuklab.ru/blog/) (серия `llm-from-scratch`)
- Состояние после каждого урока — **git-тег** `lesson-01`, `lesson-02`, …
- Текущий снимок урока 1: [`lesson-01`](https://github.com/astropsy999/llm-from-scratch/tree/lesson-01)

## Быстрый старт (урок 1)

```bash
git clone https://github.com/astropsy999/llm-from-scratch.git
cd llm-from-scratch
git checkout lesson-01

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1

# Сборку torch берите со страницы:
# https://pytorch.org/get-started/locally/
# Затем, например:
# python -m pip install torch
```

Проверка:

```bash
python src/check_environment.py
python experiments/benchmark_cpu_gpu.py
```

Без карты NVIDIA путь на процессоре — нормальный результат урока 1 (`GPU test: SKIPPED`).

## Структура

```text
src/            скрипты уроков
experiments/    замеры
data/           данные (с урока 2+)
notebooks/      черновики
lessons/        заметки по каждому уроку
```

Подробности урока 1 — в [`lessons/01.md`](lessons/01.md).
