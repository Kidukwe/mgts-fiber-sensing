# MGTS Fiber Sensing

Хакатонный проект для анализа данных распределённых температурных и
виброакустических волоконных сенсоров.

## Текущий статус

Добавлены read-only аудит и потоковая загрузка профилей с проверкой схемы.
Постановка ML-задачи, целевая метрика,
схема валидации и трактовка служебных значений данных пока не утверждены.

## Данные

Исходные CSV хранятся локально в `MGTS_data/`, считаются неизменяемыми и не
добавляются в Git. Правила работы с данными описаны в [`data/README.md`](data/README.md).

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,notebooks]'
```

## Проверка

```bash
pytest
ruff check .
```

Правила работы с ветками и pull request описаны в
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Загрузка и валидация CSV (JOTB-2)

```bash
python scripts/validate_data.py --data-dir MGTS_data --report-dir reports/ingestion
```

Команда только читает CSV. Производные отчёты `summary.json` и `issues.jsonl`
сохраняются отдельно и исключены из Git; SHA256 исходников проверяется до и после чтения.
Для графиков откройте `notebooks/01_data_audit.ipynb` с Python из `.venv`.

API, пример использования, флаги качества и ограничения описаны в
[`docs/ingestion.md`](docs/ingestion.md). Результат проверки шести CSV — в
[`reports/ingestion_validation.md`](reports/ingestion_validation.md).
DSP, признаки и baseline будут добавляться после утверждения соответствующих решений.
