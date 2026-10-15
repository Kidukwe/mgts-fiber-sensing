# Детектор температурных событий, baseline и интеграция

Исследовательский детектор кандидатов `temperature-events-v1-proposal` потребляет
`FeatureResult` из JOTB-7 (`jotb7-zones-v1-proposal`). Единицы сигнала и координат
не подтверждены. Названия heating/cooling описывают знак изменения сигнала;
выход не подтверждает температуру, аварию или её причину. Эта реализация не
утверждает предложенную ранее постановку регрессии JOTB-5 и не подменяет её оценку.

## Быстрый запуск

```bash
python -m pip install -e '.[dev]'
python scripts/run_baseline.py --input data/sample/temperature_events.csv \
  --output-dir artifacts/temperature-events-demo --index-stop 6 --zone-width 3
```

`demo/demo.py` принимает те же аргументы и вызывает тот же pipeline.
Образец полностью искусственный: 14 профилей, 2 зоны, шаг 120 секунд. После
5 стабильных профилей идут 3 профиля нагрева зоны 0 и охлаждения зоны 1,
восстановление, затем по 2 профиля локального пика/провала и восстановления.
Ожидаются 4 кандидата с `close_reason=recovered`. Это проверка поведения,
а не оценка качества распознавания на реальных данных.

```bash
python scripts/run_baseline.py --input MGTS_data/2.csv \
  --output-dir artifacts/temperature-events-real \
  --index-stop 2120 --zone-width 40 --max-gap-seconds 121 \
  --baseline-window 30 --min-history 5 --threshold 3 \
  --mad-multiplier 6 --min-consecutive 2 --release-consecutive 2
```

Все параметры доступны через `--help`. Порог 3 задан в исходных единицах сигнала;
это пример для разработки, не согласованный физический порог. Настройки, версии
реализации и SHA256 входов сохраняются в каждом запуске. Подбор на всей записи
и сообщение точности на той же записи недопустимы: для настройки нужны отдельные
временные блоки, а для precision/recall — независимая разметка.

## Baseline и причинность

На каждой зоне отдельно поддерживаются три baseline: для `median`,
`maximum_minus_median`, `median_minus_minimum`. Из последних **принятых прошлых**
наблюдений считаются median и `1.4826 * MAD` (median absolute deviation).
Окно ограничено 30 наблюдениями, минимальная история — 5. Текущая точка никогда
не участвует в своём прогнозе и пороге. JOTB-7 rolling_mean/std включают текущую
точку и для baseline детектора не используются. Его прогрев независим от
`history_ready` upstream: пригодные spatial-признаки доступны и раньше.

Порог `T = max(threshold, mad_multiplier * scale)` обеспечивает определённый
результат при постоянной истории, когда MAD=0. Подозрение замораживает baseline
соответствующего признака, включая период подтверждения. Активное событие
сохраняет заморозку до восстановления; другие признаки и зоны обновляются
независимо. Так устойчивый ступенчатый сдвиг не исчезает из-за адаптации baseline.
Стабильный высокий уровень уже при запуске станет baseline и не будет найден:
абсолютный температурный порог требует подтверждённых единиц и отдельного правила.

Дополнительный baseline `persistence` — предыдущее пригодное значение медианы
зоны; обновляется даже во время события. Ошибки обоих baseline рассчитываются
до обновления и только на одних и тех же доступных строках.

## Правила событий

| kind | residual |
| --- | --- |
| heating | median − baseline(median) |
| cooling | baseline(median) − median |
| local_peak | maximum_minus_median − baseline(maximum_minus_median) |
| local_dip | median_minus_minimum − baseline(median_minus_minimum) |

`score = residual / T`. Score — отношение к порогу, не вероятность или
калиброванная уверенность. Локальный пик/провал виден при неизменной median;
постоянная пространственная форма учитывается в своём baseline.

Событие подтверждается после `min_consecutive` последовательных `score >= 1`.
`start` — первая из этих точек; `detected_at` — момент подтверждения, когда
кандидат впервые доступен. Единичный выброс не становится событием при default=2.
После подтверждения событие продолжается при `score >= release_ratio` (default 0.5).
`release_consecutive` последовательных точек ниже этой границы завершают его.
`end` — последнее наблюдение выше границы удержания, не время закрытия.
`duration_seconds = end - start`; количество `observations` включает подтверждение
и удержание, но не точки ниже release. Краткие провалы могут лежать внутри интервала.

Зоны не объединяются между собой: события одной причины могут дать несколько
кандидатов в соседних зонах. `x_start/x_end` — координаты зоны, не уточнённые
границы очага. Разные kind в одной зоне могут пересекаться.

## Качество и границы

- Идентичность блока включает source/device/channel/block_id/selected_grid_sha256.
  Смена любого компонента закрывает кандидатов и очищает baseline.
- Rejected/excluded/file_error upstream закрывает кандидатов и сбрасывает состояние.
  Длинные gaps, смена часового домена/сетки и duplicate/backward timestamps
  обрабатываются JOTB-7 и передаются детектору как границы/исключения.
- Пустая/недостаточно полная зона, null/nonfinite spatial-признак,
  `INSUFFICIENT_VALID_POINTS`, `NONFINITE_DERIVED_STATISTIC`, `INCOMPLETE_ZONE`
  закрывают кандидатов этой зоны и начинают её прогрев заново. Остальные зоны
  продолжаются. `--min-valid-fraction` применяется к обоим этапам, default=1.0.
- `SUSPECT_SENTINEL` от другой зоны не запрещает пригодную текущую зону;
  upstream уже маскирует исходный -300. Качество сохраняется в scores/events.
- Неизвестный timezone остаётся naive. Время не переводится в придуманный UTC.
- `end_of_input`, `block_boundary`, `upstream_exclusion`, `invalid_observation`,
  `nonfinite_score` закрывают наблюдение, не доказывают восстановление.
  Только `recovered` означает наблюдавшееся выполнение release-правила.

## API

```python
from mgts_sensing.events import DetectorConfig, EventDetector
from mgts_sensing.io import iter_profiles
from mgts_sensing.zones import ZoneConfig, iter_feature_results

detector = EventDetector(DetectorConfig())
for result in iter_feature_results(iter_profiles(["MGTS_data/2.csv"]), ZoneConfig()):
    batch = detector.process(result)
    # batch.scores: tuple[dict, ...]; batch.events: завершённые кандидаты.
    # Сохраняйте обе коллекции последовательно.
remaining_events = detector.finish()  # обязательно в конце потока
```

Один экземпляр — один упорядоченный запуск. Детектор потребляет исходный поток
FeatureResult целиком, включая rejected outcomes. Не передавайте отфильтрованные
строки без границ и не смешивайте block_id из разных запусков. Память ограничена
числом зон, baseline_window и min_consecutive. Автоматической сортировки нет.

## Выход и оценка

- `scores.jsonl`: одна строка на kind каждой пригодной зоны. Общие поля:
  schema_version, source/device/channel/block/grid/zone и границы зоны, timestamp,
  kind, status, quality_flags. Для `scored` дополнительно row_number, value,
  baseline, scale, threshold, residual, score, persistence (только median kinds),
  candidate (score>=1), active (подтверждённый кандидат ещё открыт).
  `baseline_warmup` не содержит численных прогнозов. Непригодная зона даёт одну
  строку `invalid_observation`, kind=null; вычислительное переполнение даёт
  `nonfinite_score`. Это не строки нормального состояния и не нулевые scores.
- `events.jsonl`: завершённые кандидаты со schema_version, run-local event_id,
  status=candidate, происхождением и границами зоны, kind, start/detected_at/end,
  duration_seconds, observations, peak_score/peak_timestamp/peak_residual,
  threshold_at_peak/baseline_at_peak, start_row_number/end_row_number,
  quality_flags, close_reason. Pending-подозрения короче min_consecutive исключены.
  Открытые кандидаты видны сразу в scores.active, а в events попадают при закрытии
  или `finish`. event_id требует namespace запуска из config/input hashes.
- `issues.jsonl`: все проблемы ingestion/features с исходными строками и кодами.
- `config.json`: параметры обоих этапов, версии схем, Python/NumPy, seed=42
  (случайность не используется), SHA256 исходников до/после и модулей pipeline.
- `report.json`: учёт outcomes/зон/событий, виды и причины закрытия, проблемы,
  неизменность raw и полнота чтения, causal MAE/RMSE обоих baseline.

MAE сначала усредняет пригодные зоны одного профиля, затем профили; RMSE —
корень из среднего профильных MSE. Прогрев/непригодные наблюдения исключаются,
число профилей и зональных наблюдений записывается. Это описательные ошибки
последовательного прогноза сигнала, **не** метрики качества детектора, не
эксперимент с отложенным test и не доказательство превосходства модели.
Precision/recall остаются null, пока нет внешних labels. Baseline не обучает
трансформации на будущем и не объявляет начало записи нормальным состоянием.

CLI только читает raw и не пишет внутрь каталогов исходников. Повторный запуск
в узнаваемый output dir заменяет только пять именованных файлов; чужие файлы
сохраняются, symlink-выходы отклоняются. Для отдельных экспериментов нужны
отдельные каталоги. Файлы сначала готовятся во временном каталоге; публикация
каждого файла атомарна, но всей группы при аварийном завершении — нет.
Exit 0 — данные обработаны без исключений; 1 — rejection/exclusion/file error
или изменившийся SHA; 2 — неверные параметры или небезопасный output dir.
Наличие кандидатов само по себе не меняет exit code.
