# JOTB-7: proposal API и схема признаков

Статус: **proposal**, schema version `jotb7-zones-v1-proposal`.
Это исследовательская таблица признаков, не события, target или ML-выборка.
Единицы `x`, `y` и соответствие зон объектам неизвестны (`null` в schema).
Timestamp означает записанное время измерения; online-доступность не доказана.

Уточнения draft до передачи, 2026-10-14: экспорт использует x_start/x_end
согласно запросу пользователя (внутренние ZoneSummary.x_min/x_max имеют те же
значения). Добавлены явные flags PARTIAL_TEMPORAL_VALUES/NO_VALID_TEMPORAL_VALUES
и диагностика unexpected NaN/Inf. Численные формулы и прежний JOTB-3 API не менялись.

## Вход и настройки

`zones.py` принимает существующие `ProfileResult` от `iter_profiles`.
Исходные `Profile`, CSV и массивы не меняются. Все dataclass результата frozen;
производные ndarray read-only. Вход обрабатывается в переданном порядке;
автоматической сортировки, интерполяции и общей будущей сетки нет.

```python
ZoneConfig(index_start=0, index_stop=2120, zone_width=40,
           history_size=5, max_gap_seconds=121, min_valid_fraction=1.0)
aggregate_profile(profile: Profile, config: ZoneConfig) -> AggregatedProfile
iter_feature_results(results: Iterable[ProfileResult], config: ZoneConfig)
    -> Iterator[FeatureResult]
```

Индексы целые (bool запрещён): `0 <= index_start < index_stop`, width >= 1,
H >= 2; max_gap конечный > 0, `0 < min_valid_fraction <= 1`.
`index_stop` exclusive. Все параметры сохраняются, не подбираются автоматически.
Недостаточная длина для **всего** выбранного slice исключает профиль явно;
никаких предположений о отсутствующих координатах.

`AggregatedProfile`: `profile` (исходный Profile), `selected_x` (точная копия
выбранных координат), `zones` (tuple из `ZoneSummary`).
`ZoneSummary` имеет поля `zone_id`, `index_start`, `index_stop`, `x_min`, `x_max`,
`nominal_count`, `point_count`, `valid_count`, `sentinel_count`, `valid_fraction`,
`sentinel_fraction`, `median`, `minimum`, `maximum`, `maximum_minus_median`,
`median_minus_minimum`, `std`, `quality_flags` (tuple[str, ...]).
Невалидный Profile или недоступный slice вызывает `ValueError` в одиночном API;
потоковый API перехватывает его и выдаёт contextual Issue, а не теряет запись.

`FeatureResult`: `input_result: ProfileResult`, `status`, `rows: tuple[dict, ...]`,
`issues: tuple[Issue, ...]`, `block_id: int | None`.
Status: `aggregated` (есть зональные строки), `rejected` (ingestion отказал),
`excluded` (ingestion загрузил, признаки недопустимы), `file_error` (row_number=0).
Ровно один FeatureResult на каждый входной outcome; issues включают исходные
ingestion issues и новые проблемы. Report не считает row_number=0 как профиль.

## Пространственная агрегация, затем время

Зоны — последовательные группы width точек внутри slice; `zone_id` начинается
с нуля. Последнюю неполную зону **сохраняем**, flag `INCOMPLETE_ZONE`.
Индексы зоны абсолютные относительно исходного профиля, stop exclusive.
Экспортные x_start/x_end — первая/последняя фактическая точка, не границы помещения.

Недоступная точка: `sentinel_mask=True` **или** raw `y == -300`. Она исключается
только из производных статистик. `nominal_count=zone_width`,
`point_count` — фактически присутствующие точки, `valid_count` — незамаскированные,
`sentinel_count` — число подозрительных точек.
`valid_fraction=valid_count/nominal_count`; полнота последней зоны намеренно
меньше единицы. `sentinel_fraction=sentinel_count/point_count`.
Если valid_count=0 или valid_fraction < min_valid_fraction, все пространственные
числовые статистики null и flag `INSUFFICIENT_VALID_POINTS`.
Иначе median, minimum, maximum, std (`ddof=0`), maximum_minus_median и
median_minus_minimum считаются по всем доступным точкам.
Flags описывают качество, не неисправность/аномалию объекта.

Временные признаки вычисляются **из зональных медиан**, не из уже вычисленных
по координатам временных признаков. Валидная производная median == -300
(например, из raw [-301, -299]) остаётся валидной: служебный marker проверяется
только до пространственной агрегации.

Численное ядро `compute_temporal_values(values, valid_mask, timestamps)` принимает
конечную матрицу `(H, zones)` oldest-first, явную bool mask одинаковой формы
(True = доступно) и H datetime oldest-first; оно не интерпретирует -300.
Времена должны строго возрастать в одном naive/aware домене; известные offsets
сравниваются как UTC instants. Δt вычисляется по двум последним timestamps.
Возвращает `TemporalValues` с полями lag_values, delta, rate, rolling_mean,
rolling_std, valid_count, delta_seconds; пропуски внутри вычислений NaN.
Старый `compute_time_features` сохраняет прежний API и прежнюю raw-mask
семантику, лишь делегирует численную часть ядру.

Для `iter_windows` допустим внутренний **производный** Profile: x=zone ordinal,
y=median (0 как finite placeholder при отсутствии), sentinel_mask=missing;
metadata сохраняется, points_size=zone_count. Он не экспортируется как raw.
Точная совместимость selected_x проверяется отдельно **до** этого представления.
Полное H-окно проверяется существующим iter_windows; численное ядро получает
явную missing mask. Старый raw-feature API к этому Profile не применяется.

## История и строки прогрева

История локальна непрерывному блоку одного source_file/dts_id/channel_id и
naive/aware домена. Известные offsets сравниваются как UTC instants, naive
остаётся naive. Смена этих ключей, selected_x или gap > max_gap_seconds
начинает новый блок. Gap **равный** границе допускается.
Изменение хвоста вне slice не разрывает историю при точном равенстве selected_x.

Rejected/file error/feature-excluded outcome разрывает историю. Duplicate key
или время <= предыдущего внутри блока исключает текущую запись явно
(`DUPLICATE_KEY` / `NONINCREASING_TIMESTAMP`) и разрывает историю; последующая
допустимая запись начинает прогрев. Дубликаты не удаляются незаметно.
Поток одновременно хранит только текущий блок (не продолжает прежний после
перехода к другому); block_id — возрастающий ID, первый 0, не sensor ID.

На каждом aggregated профиле выводятся все зоны, включая неполные/пустые по
маске. На прогреве `history_ready=false`, flag `HISTORY_WARMUP`, lag_medians
длины H с доступными текущими/прошлыми значениями и null справа.
`rolling_mean`, `rolling_std`, `delta`, `rate`, `delta_seconds` на прогреве null.
После H профилей history_ready=true **даже если** некоторые медианы отсутствуют;
это полнота по числу профилей, не гарантия качества зоны. temporal_valid_count
считает доступные медианы; mean/std используют их равновесово, std ddof=0;
при нуле доступных mean/std null, при одном std=0. delta/rate доступны только
если две последние медианы доступны; rate делится на фактическое Δt.
Полная история с отсутствующими медианами получает PARTIAL_TEMPORAL_VALUES;
при отсутствии всех значений дополнительно NO_VALID_TEMPORAL_VALUES.
Непредвиденный NaN/Inf при доступных входах даёт NONFINITE_DERIVED_STATISTIC,
null и contextual Issue; обычный пропуск не объявляется переполнением.

## Точные ключи строки features.jsonl

Без вложенных feature namespaces; null вместо NaN/Inf, JSON arrays вместо
ndarray. Все строки имеют одинаковые ключи:

| Ключи | Смысл / JSON тип |
|---|---|
| schema_version | `jotb7-zones-v1-proposal` |
| source_file, row_number, line_number | string, int, int; происхождение текущего профиля |
| dts_id, channel_id | string, string |
| timestamp, timestamp_raw | ISO string без выдуманного offset; исходный string |
| timezone_raw, utc_offset_seconds, timezone_known | string/null, number/null, bool |
| block_id, zone_id | int, int |
| selected_grid_sha256 | string SHA256 selected_x float64 bytes; различает внутренние сетки |
| index_start, index_stop, x_start, x_end | int, int, number, number |
| nominal_count, point_count, valid_count, sentinel_count | int |
| valid_fraction, sentinel_fraction | number |
| median, minimum, maximum, maximum_minus_median, median_minus_minimum, std | number/null |
| history_size, history_count, history_ready | int H, int <= H, bool |
| history_start, history_end | ISO strings доступного oldest/current времени |
| history_sources | array current-first: `{source_file,row_number,line_number,timestamp,timestamp_raw}` |
| lag_medians | array длины H current-first, number/null |
| temporal_valid_count | int доступных медиан (включая прогрев) |
| delta_seconds, delta, rate, rolling_mean, rolling_std | number/null |
| quality_flags | unique array strings: raw ingestion + zone + boundary/warmup flags |

`history_sources` содержит только фактическую доступную историю, без padding;
история происхождения одинакова у всех зон данного текущего профиля.
Никакие строки не являются labels или подтверждёнными событиями.

## CLI / отчёт

`python scripts/build_features.py --input MGTS_data/2.csv --output-dir artifacts/jotb7`.
Повторяемый --input сохраняет порядок файлов. Параметры соответствуют ZoneConfig.
Создаются только производные features.jsonl/schema.json/config.json/report.json/
issues.jsonl вне raw. Schema содержит ключи, типы, proposal, неизвестные единицы,
маску, причины null и порядок пространственной/временной агрегации.

Report: ingestion_records=loaded+rejected; file events отдельно; loaded=aggregated+
feature_excluded; zone_count=feature_rows; warmup_rows и full_history_rows отдельно;
full_windows — число профилей с history_ready, не число зональных строк;
temporal_blocks, issue counts по code/severity, hashes до/после и raw_unchanged.
CSV_READ_ERROR помечает accounting_complete=false: unread remainder невозможно
молчаливо объявить учтённым. Config сохраняет параметры, версии и входы.
Issues: source_file/row_number/line_number/severity/code/description на каждую
проблему, включая ingestion. Не строить события и не применять пороги температуры.
