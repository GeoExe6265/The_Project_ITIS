# EduDataAnalyzer

[![CI](https://github.com/GeoExe6265/The_Project_ITIS/actions/workflows/ci.yml/badge.svg)](https://github.com/GeoExe6265/The_Project_ITIS/actions/workflows/ci.yml)

Инструмент для анализа образовательных данных: считает ключевые метрики, строит простую модель риска отчисления и генерирует отчеты, которые помогают преподавателям и администраторам вовремя замечать проблемы в успеваемости.

## Зачем это нужно
- Ручная сводка по десяткам показателей занимает часы, автоматизация экономит время.
- Простая интерпретируемая модель позволяет прозрачно обосновывать решения по поддержке студентов.
- Регулярные отчеты по расписанию дают оперативную картину без ручного запуска скриптов.

## Возможности
- Загрузка CSV с проверкой схемы и очисткой данных.
- Метрики по группе и по программам: средний балл, доля сдавших, корреляция посещаемости и оценок.
- Логистическая регрессия для оценки вероятности отчисления (балансирует классы).
- Топ студентов по риску с удобным Markdown-отчетом.
- CLI с командами summarize, predict, report.
- GitHub Actions: линтер, тесты и еженедельная автогенерация отчета-артефакта.

## Требования
- Python 3.8+
- pip

## Установка
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Быстрый старт
```bash
python -m edudataanalyzer.cli summarize data/sample_students.csv
python -m edudataanalyzer.cli predict data/sample_students.csv \
	--grade 72 --attendance_rate 0.8 --assignments_completed 8 --absences 3
python -m edudataanalyzer.cli report data/sample_students.csv --output report.md
```

Пример Python-кода:
```python
from pathlib import Path
from edudataanalyzer import load_dataset, train_risk_model, predict_risk

data = load_dataset(Path("data/sample_students.csv"))
model = train_risk_model(data)
score = predict_risk(model.model, [{
		"grade": 75,
		"attendance_rate": 0.82,
		"assignments_completed": 8,
		"absences": 2,
}])
print(f"Отчисление: {score[0]:.2%}")
```

## Структура проекта
- [src/edudataanalyzer](src/edudataanalyzer): код загрузки, метрик, модели и CLI.
- [data/sample_students.csv](data/sample_students.csv): пример набора данных.
- [tests](tests): unit-тесты для метрик и модели.
- [.github/workflows/ci.yml](.github/workflows/ci.yml): линт, тесты и генерация отчета.

## Схема данных
Ожидаемые столбцы CSV:
- student_id (str)
- program (str)
- grade (float, 0–100)
- attendance_rate (float, 0–1)
- assignments_completed (int)
- absences (int)
- risk_label (0/1)

## Проверки и CI/CD
- Ruff: `ruff check src tests`
- Pytest: `pytest -q`
- GitHub Actions: на каждый push/PR запускается линт и тесты; по расписанию (понедельник 06:00 UTC) и вручную доступен джоб weekly-report, который генерирует Markdown-отчет на базе [data/sample_students.csv](data/sample_students.csv) и сохраняет его как артефакт.

## Тесты
```bash
pytest
```

## Что дальше
- Добавить визуализации (heatmap корреляций, распределения рисков).
- Интеграция с LMS API для автоматической выгрузки данных.