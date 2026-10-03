# Обучаем LLM с нуля

Код к практической серии статей: от лаборатории PyTorch до маленькой GPT.

- Статьи: [stuzhuk.page](https://stuzhuk.page/ru/blog/) / [stuzhuklab.ru](https://stuzhuklab.ru/blog/) (серия `llm-from-scratch`)
- Состояние после каждого урока — **git-тег** `lesson-01`, `lesson-02`, …
- Актуальный снимок: [`lesson-02`](https://github.com/astropsy999/llm-from-scratch/tree/lesson-02)

## Быстрый старт

```bash
git clone https://github.com/astropsy999/llm-from-scratch.git
cd llm-from-scratch
git checkout lesson-02

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
```

### Урок 1 — лаборатория

Сборку `torch` берите со [страницы PyTorch](https://pytorch.org/get-started/locally/), затем:

```bash
python src/check_environment.py
python experiments/benchmark_cpu_gpu.py
```

Без карты NVIDIA путь на процессоре — нормальный результат (`GPU test: SKIPPED`). Подробнее: [`lessons/01.md`](lessons/01.md).

### Урок 2 — текст в числа

PyTorch для этого урока не нужен.

```bash
python src/demo_text_to_numbers.py
```

Подробнее: [`lessons/02.md`](lessons/02.md).

## Структура

```text
src/            скрипты уроков
experiments/    замеры
data/           данные
notebooks/      черновики
lessons/        заметки по каждому уроку
```
