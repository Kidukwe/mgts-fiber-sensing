> Исторический документ предыдущего этапа. Актуальные единицы, постановка и решение:
> [decisions.md](decisions.md), [solution.md](solution.md).

# Передача JOTB-7 Илье и Максиму

Контракт `jotb7-zones-v1-proposal` — **предложение**, не согласованная физическая
модель и не готовая ML-выборка. Полный API: [jotb7_contract.md](jotb7_contract.md).
Контракт нельзя менять молча: измените версию и согласуйте её с потребителями.

## Один реальный запуск

Из корня репозитория, с Python из существующего `.venv`:

```bash
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/build_features.py --input MGTS_data/2.csv --output-dir artifacts/jotb7
```

Все параметры доступны через `--help`. Defaults — исследовательский пример,
а не параметры утверждённого детектора: slice `[0,2120)`, width=40, H=5,
gap=121 секунд, min_valid_fraction=1.0. Для другого канала эти значения нельзя
переносить без проверки. Граница gap равна 121 разрешена; большие интервалы
разрывают историю. У JOTB-7 нет требования к будущему target, train/test или
интервалам 120–121 на каждом шаге: есть только параметризованная max-gap policy.

Производный выбранный участок агрегируется только после проверки x. Полный
профиль остаётся неизменным. При равных selected x допустимы разные хвосты;
изменение любого выбранного x начинает новый temporal block.

## Файлы результата

| Файл | Назначение |
| --- | --- |
| `features.jsonl` | Одна строка на зону каждого допустимого текущего профиля, включая прогрев |
| `schema.json` | Версия, смысл колонок, типы и JSON Schema строки |
| `config.json` | Настройки, proposal, версии Python/NumPy, SHA256 входов и реализации |
| `report.json` | Полный учёт ingestion/feature exclusions, строки, окна, блоки, причины и SHA256 после чтения |
| `issues.jsonl` | Файл, запись, физическая строка, severity/code/description каждой проблемы |

Generated artifacts не добавляются в Git. CLI не пишет внутри каталога
исходного CSV и не заменяет symlink. Повторный запуск обновляет только пять
именованных производных файлов узнаваемого JOTB-7 run; прочие файлы сохраняет.
Для разных экспериментов используйте разные output dirs. Exit 0 — нет
отклонений/исключений/файловых ошибок, не гарантия качества; 1 — такие проблемы
или изменение SHA; 2 — неверные параметры/небезопасный output directory.

Повторяемый `--input` читает файлы в заданном порядке, без сортировки/соединения
по времени. Другой файл/канал и naive/aware домен начинают новый блок.
Неизвестный timezone остаётся неизвестным. CSV corruption с неизвестными
границами остатка явно даёт `accounting_complete=false`.

## Чтение таблицы

Без новых зависимостей:

```python
import json
from pathlib import Path

with Path("artifacts/jotb7/features.jsonl").open(encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        # JSON arrays остаются list, null становится None.
        print(row["block_id"], row["zone_id"], row["timestamp"],
              row["median"], row["maximum_minus_median"], row["quality_flags"])
```

Для небольших таблиц с notebook dependencies:

```python
import pandas as pd
table = pd.read_json("artifacts/jotb7/features.jsonl", lines=True)
```

После pandas null числовых колонок может стать NaN: это не ноль и не нормальное
состояние. Для больших потоков предпочтительнее построчный JSON reader.

## Семантика и качество

Порядок: исходные значения → spatial median/min/max/std и excursions →
temporal статистики **медиан зон**. Узкий пик/провал может не менять median,
но остаётся в maximum/minimum и excursions. Это характеристики формы, не
обнаруженные события или физические причины.

- `valid_count` — текущие незамаскированные точки; `valid_fraction` делится
  на nominal width. Неполная крайняя зона сохраняется с `INCOMPLETE_ZONE`.
- Raw -300 или sentinel_mask=True исключаются только из производных чисел.
  `sentinel_fraction` — их доля среди фактических точек, не вероятность отказа.
- Нулевая/недостаточная полнота даёт null статистики и quality flag; исходные
  значения не меняются. Производная median -300 из обычных чисел валидна.
- `lag_medians`: ровно H значений current-first, null при unavailable/padding.
  `history_sources` тоже current-first, но только реальные профили (без padding).
- Прогрев: `history_ready=false`, spatial признаки есть; delta/rate/rolling
  mean/std пока null. Полное окно: history_ready=true даже при missing медианах.
  Всегда проверяйте `temporal_valid_count` и сами null, не только ready.
  Неполная доступность H-медиан отражается PARTIAL_TEMPORAL_VALUES; нулевая —
  дополнительно NO_VALID_TEMPORAL_VALUES. Это признаки качества, не события.
- rolling_mean/std равновесовые по доступным профилям, std ddof=0; не среднее
  за фиксированные пять минут. `rate` использует реальные секунды.
- `block_id` локален одному запуску; ключ зонального ряда включает dataset/run,
  source/dts/channel, block_id, selected_grid_sha256 и zone_id.
- Координаты экспортируются как `x_start`/`x_end`: это первая/последняя
  фактическая точка зоны, а не найденные границы события или помещения.
- Отклонение ingestion/feature stage разрывает историю. Duplicate/backward
  запись явно исключается, следующая начинает новый прогрев.

Для Ильи: используйте координаты, spatial extrema и masked/missing flags при
визуализации; исходный raw профиль можно получить по source/row/line. Не
соединяйте разные blocks и grids. Фильтры, FFT/STFT и DSP здесь не реализованы.

Для Максима: происхождение всех H профилей доступно. Перед fit отдельно
согласуйте split, обработку прогрева/NaN/полноты и критерий оценки. Split по
строкам JSONL случайно делать нельзя: соседние зоны и окна зависимы. Не обучайте
transform по всему датасету и не объявляйте train автоматически «нормальным».

## Маленький синтетический пример

Готовые образцы `data/sample/jotb7_profiles.csv`, `jotb7_config.json`,
`jotb7_features.jsonl`, `jotb7_schema.json`. Это искусственные числа и координаты,
не температурные единицы или размеченные аварии. 4 профиля, 2 зоны, H=3.

```bash
python scripts/build_features.py --input data/sample/jotb7_profiles.csv \
  --output-dir artifacts/jotb7-fixture --index-stop 6 --zone-width 3 \
  --history-size 3 --max-gap-seconds 40 --min-valid-fraction 0.5
```

Пример включает узкий пик и провал при неизменной median, частичную и полную
маску; на третьем профиле уже доступна полная история. Два файла table/schema
позволяют разрабатывать reader без обработки real CSV и без запуска детектора.
Разработка fixture не равна оценке качества на реальных событиях.

## Самостоятельная проверка

```bash
pytest
ruff check .
git diff --check
```

Проверьте report: loaded+rejected=ingestion_records;
aggregated_profiles+feature_excluded=loaded; warmup_rows+full_history_rows=
feature_rows. SHA до/после должен совпадать. Report full_windows считает моменты
с H-историей, не умноженные на зоны независимые испытания.

Три фрагмента, которые нужно понимать: `aggregate_profile` (маска и полнота),
`iter_feature_results` (координаты/границы/происхождение),
`compute_temporal_values` (current-first lags и реальный Δt).
