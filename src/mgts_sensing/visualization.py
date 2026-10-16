"""Static diagnostic figures and a standalone local HTML report."""

from __future__ import annotations

import html
from itertools import pairwise
from pathlib import Path

import numpy as np


def plot_solution(records, meta, values, prediction, split, alerts, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_zone = max((max(r.get("zones", {0: {}}), default=0) for r in records), default=0)
    # Bound visual memory; aggregate neighbouring profiles only for display.
    step = max(1, len(records) // 700)
    matrix = np.full((len(records[::step]), max_zone + 1), np.nan)
    for t, record in enumerate(records[::step]):
        for zone_id, zone in record.get("zones", {}).items():
            matrix[t, zone_id] = zone["median"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), layout="constrained")
    finite = matrix[np.isfinite(matrix)]
    lo, hi = np.quantile(finite, [.02, .98]) if finite.size else (0, 1)
    image = axes[0].imshow(matrix.T, origin="lower", aspect="auto", cmap="inferno",
                           vmin=lo, vmax=max(lo + .2, hi),
                           extent=(0, len(records), 0, max_zone + 1), interpolation="nearest")
    for boundary in (.6, .8):
        axes[0].axvline(len(records) * boundary, color="cyan", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Profile index (gaps retain record order)", ylabel="Fiber zone",
                title="Temperature map / train 60% / validation 20% / test 20%")
    fig.colorbar(image, ax=axes[0], label="Temperature, °C (2–98% display scale)")
    indices = np.flatnonzero(split == 2)
    # Show the zone with most residual variability, explicitly a post-hoc diagnostic.
    zone_ids = np.array([m["zone_id"] for m in meta])
    unique = np.unique(zone_ids[indices])
    zone = max(unique, key=lambda z: np.mean(np.abs(values[indices[zone_ids[indices] == z], 0] -
                                                      prediction[indices[zone_ids[indices] == z], 0])))
    selected = indices[zone_ids[indices] == zone]
    positions = np.array([meta[i]["profile_index"] for i in selected], dtype=float)
    actual, predicted = values[selected, 0].copy(), prediction[selected, 0].copy()
    # Insert plot breaks at session discontinuities; never draw a line across a gap.
    breaks = np.array([False] + [meta[b]["block"] != meta[a]["block"] or
                                meta[b]["profile_index"] != meta[a]["profile_index"] + 1
                                for a, b in pairwise(selected)])
    actual[breaks] = np.nan
    predicted[breaks] = np.nan
    axes[1].plot(positions, actual, label="Observed median", color="#e25432", linewidth=1.6)
    axes[1].plot(positions, predicted, label="Selected one-step forecast", color="#167d9b",
                 linewidth=1.2)
    axes[1].set(xlabel="Profile index", ylabel="Temperature, °C",
                title=f"Test forecast: zone {zone} (largest test residual; diagnostic selection)")
    axes[1].legend(loc="best")
    fig.savefig(Path(output) / "diagnostics.png", dpi=150)
    plt.close(fig)


def write_solution_report(summary, output):
    esc = html.escape
    files = summary["files"]
    rows, figures, ablation = [], [], []
    for f in files:
        name = esc(f["source_file"])
        metric = f["test_metrics_selected"]
        persistence = f["test_metrics"]["persistence"]["mae_c"]
        rows.append(f'<tr><td>{name}</td><td>{f["profiles"]}</td>'
                    f'<td>{esc(f["selected_model"])}</td><td>{metric["mae_c"]:.4f}</td>'
                    f'<td>{metric["rmse_c"]:.4f}</td><td>{persistence:.4f}</td>'
                    f'<td>{f["candidate_events"]}</td></tr>')
        ab = f["feature_ablation"]
        a, b = ab["temporal"], ab["spatial_dsp"]
        ablation.append(f'<tr><td>{name}</td>'
                        f'<td>{f["validation_metrics"][a]["mae_c"]:.4f}</td>'
                        f'<td>{f["validation_metrics"][b]["mae_c"]:.4f}</td>'
                        f'<td>{f["test_metrics"][a]["mae_c"]:.4f}</td>'
                        f'<td>{f["test_metrics"][b]["mae_c"]:.4f}</td></tr>')
        figures.append(f'<details><summary>{name} — карта температуры и прогноз</summary>'
                       f'<img src="{esc(Path(name).stem)}/diagnostics.png" '
                       f'alt="Карта температуры и прогноз для {name}">'
                       f'<p><a href="{Path(name).stem}/model.json">Модель и параметры</a> · '
                       f'<a href="{Path(name).stem}/metrics.json">Все метрики</a> · '
                       f'<a href="{Path(name).stem}/events.json">События JSON</a></p></details>')
    events = []
    kind_names = {"heating": "Нагрев", "cooling": "Охлаждение", "local_peak": "Локальный пик",
                  "local_dip": "Локальный провал"}
    for e in summary["events"][:1000]:
        proof = e["peak_evidence"]
        evidence = (f'строка {proof["row_number"]}; T={proof["actual_temperature_c"]:.2f} °C; '
                    f'скорость {proof["change_rate_c_per_minute"]:.2f} °C/мин; '
                    f'канал {proof["evidence_channel"]}; '
                    f'доля нагрева/охлаждения {proof["fraction_hot"]:.0%}/{proof["fraction_cold"]:.0%}')
        events.append('<tr data-source="' + esc(e["source_file"]) + '"><td>' +
                      esc(e["source_file"]) + '</td><td>' + kind_names[e["kind"]] + '</td><td>' +
                      esc(e["start"]) + '</td><td>' + f'{e["distance_start_m"]:.3f}–'
                      f'{e["distance_stop_m"]:.3f}</td><td>{e["peak_amplitude_c"]:.2f}</td>'
                      f'<td>{e["duration_seconds"]:.0f}</td><td>{e["priority_score"]:.1f}</td>'
                      f'<td>{esc(evidence)}</td></tr>')
    content = '''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>МГТС — мониторинг температуры</title>
<style>
body{margin:0;background:#f3f5f4;color:#172a2a;font:16px/1.6 system-ui,sans-serif}
main{max-width:1200px;margin:auto;padding:48px 24px}h1{font-size:clamp(30px,5vw,54px);
line-height:1.12;max-width:900px;margin:16px 0 24px}h2{font-size:26px;margin-top:40px}
.eyebrow{color:#24716b;letter-spacing:.12em;font-weight:700;font-size:13px}
.cards{display:flex;gap:16px;flex-wrap:wrap;margin:32px 0}.card{flex:1;min-width:170px;
background:white;padding:20px;border-top:3px solid #27847a}.card strong{display:block;
font-size:30px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;
font-size:14px}th,td{padding:12px 14px;text-align:left;border-bottom:1px solid #dce4e1}
th{color:#44635f;white-space:nowrap}td{font-variant-numeric:tabular-nums}
.note{border-left:4px solid #d78d2b;padding:12px 20px;background:#fff9ef}
details{background:white;padding:18px 22px;margin:12px 0}summary{cursor:pointer;font-weight:600}
img{width:100%;height:auto;margin-top:16px}a{color:#126d67}select{font:inherit;padding:8px;
border:1px solid #8ca9a3;background:white;margin:12px 0}footer{margin-top:50px;color:#526c67}
</style><main><div class="eyebrow">МГТС / ТЕЛЕКОМ БУДУЩЕГО / ТЕМПЕРАТУРНАЯ СЕНСОРИКА</div>
<h1>Инженерные коммуникации под наблюдением</h1>
<p>Поиск изменений температуры вдоль волокна: пространственная обработка сигнала,
обученная регрессия и совместная оценка амплитуды, длительности и протяжённости.</p>
<div class="cards">CARDS</div>
<p class="note">События — кандидаты для проверки инженером. Разметки аварий нет:
precision и recall не определены. MAE ниже характеризует прогноз температуры.
Протечку и её причину нельзя подтвердить только температурным сигналом.</p>
<p class="note">Записи уже исследовались при разработке первой версии. Результаты v2 на тех же
периодах — повторная исследовательская оценка, а не новый независимый тест.
Для подтверждения переносимости нужен следующий период или отдельный объект.</p>
<h2>Проверка на отложенном времени</h2><p>Для каждого файла: первые 60% профилей — обучение,
следующие 20% — выбор модели и калибровка, последние 20% — тест. Текущий профиль
не входит в признаки собственного прогноза. Сравнение ведётся на одних наблюдениях;
каждый момент времени имеет одинаковый вес. Целевая величина — следующая медианная
температура зоны. Модель выбирается по validation MAE, включая простые baseline.</p>
<div class="scroll"><table><thead><tr><th>Файл</th><th>Профилей</th><th>Выбрана модель</th>
<th>Test MAE °C</th><th>Test RMSE °C</th><th>Persistence MAE °C</th><th>Событий на тесте</th>
</tr></thead><tbody>METRICS</tbody></table></div>
<h2>Вклад пространственной обработки</h2><p>Сравниваем одну семью ridge на одинаковых
целевых значениях и временных блоках: временные признаки и расширение пространственной
формой/DSP. Alpha каждого варианта выбирается только на validation. Улучшение
пространственного варианта не предполагается заранее. Ширина участка — настраиваемая
опора агрегации; требования использовать 40 точек нет.</p>
<div class="scroll"><table><thead><tr><th>Файл</th><th>Validation temporal MAE</th>
<th>Validation DSP MAE</th><th>Test temporal MAE</th><th>Test DSP MAE</th></tr></thead>
<tbody>ABLATION</tbody></table></div>
<h2>Сигнал и прогноз</h2><p>Пунктир отделяет выборки. Исходная ось x сохранена в JSON;
метры в событиях рассчитаны по индексам с шагом 0,0025 м. Численный шаг CSV сам по себе
не доказывает единицу исходной оси. Профили разной длины не дополняются вымышленными значениями.</p>
FIGURES
<h2>Приоритет проверки</h2><p>Кандидат должен повториться в двух последовательных профилях.
Используются отклонения от прогноза и от устойчивого прошлого уровня. Ошибка прогноза
подтверждается направлением фактического отклонения от фона вне погрешности, чтобы возврат
к нормальному уровню не превращался в противоположную тревогу.
Фон замораживается при подозрении. Порог прогноза — 99,5-й перцентиль validation-остатка
(по умолчанию), фон — 6 MAD. Нижняя граница — 0,2 °C для разности температур
и 0,4 °C для изменения max−median / median−min, где участвуют четыре измеренных величины.
Это настройки алгоритма, а не согласованные границы аварии.
Приоритет = ln(1 + амплитуда/порог) × ln(2 + длительность/шаг времени) × √число точек очага.
Это рейтинг, не вероятность. Протяжённость относится к волокну, а не площади протечки.
Объединение соседних зон выполняется после обнаружения; времена start и detected_at разделены.</p>
<label>Источник <select id="source"><option value="">Все файлы</option>OPTIONS</select></label>
<div class="scroll"><table id="events"><thead><tr><th>Файл</th><th>Тип</th><th>Начало с UTC offset</th>
<th>Расстояние, м</th><th>Амплитуда °C</th><th>Длительность, с</th><th>Приоритет</th><th>Доказательства на пике</th></tr></thead>
<tbody>EVENTS</tbody></table></div><p>Показано до 1000 приоритетных кандидатов на тесте.
Полный каталог: <a href="solution.json">solution.json</a>.</p>
<h2>Эксплуатация и границы решения</h2><p>Температура — °C, точность ±0,1 °C,
заявленный пространственный шаг — 0,25 см, время без offset — фиксированный UTC+3.
Значения −300 исключены как конец волокна. Часовой пояс явных timestamp сохранён.
Окна разрываются при смене канала, сетки участка, отклонённой записи или разрыве
с интервалом вне 0,5–1,5 типичного шага (типичный шаг оценивается только по train).
Полный профиль считается доступным при обработке записи; задержка его получения неизвестна.</p>
<p>Тип объекта влияет на нормальный режим и интерпретацию: режим работы HVAC,
температура теплоносителя, расписание и контакт волокна требуют настройки для здания.
Для привязки к помещению нужна схема трассы. Для подтверждения причин нужны внешние
датчики или осмотр. Абсолютный перегрев, существовавший до начала записи, требует
паспортных пределов оборудования. Обучение и наблюдения здесь не объявляются нормой автоматически.</p>
<footer>Воспроизводимый локальный эксперимент · Python / NumPy · SHA256 входов и кода в JSON ·
без внешних скриптов и сетевых зависимостей отчёта</footer></main>
<script>document.querySelector('#source').addEventListener('change', e => {
 document.querySelectorAll('#events tbody tr').forEach(row => {
 row.hidden = !!e.target.value && row.dataset.source !== e.target.value;
 });
});</script></html>'''
    cards = [(str(len(files)), "исходных файлов"),
             (f'{sum(f["profiles"] for f in files):,}', "профилей обработано"),
             (str(len(summary["events"])), "кандидатов на тесте"),
             ("±0,1 °C", "заявленная точность")]
    replacements = {
        "CARDS": "".join(f'<div class="card"><strong>{value}</strong>{label}</div>'
                         for value, label in cards),
        "METRICS": "".join(rows), "ABLATION": "".join(ablation), "FIGURES": "".join(figures), "EVENTS": "".join(events),
        "OPTIONS": "".join(f'<option>{esc(f["source_file"])}</option>' for f in files),
    }
    for key, value in replacements.items():
        content = content.replace(key, value)
    (Path(output) / "report.html").write_text(content, encoding="utf-8")
    markdown = ["# Итог эксперимента МГТС", "",
                "Результат — прогноз температуры и кандидаты аномалий на отложенном времени.",
                "", "| Файл | Модель | Test MAE °C | Persistence MAE °C | Кандидатов |",
                "| --- | --- | ---: | ---: | ---: |"]
    for f in files:
        markdown.append(f'| {f["source_file"]} | {f["selected_model"]} | '
                        f'{f["test_metrics_selected"]["mae_c"]:.6f} | '
                        f'{f["test_metrics"]["persistence"]["mae_c"]:.6f} | '
                        f'{f["candidate_events"]} |')
    markdown += ["", "## Пространственные признаки против временных", "",
                 "| Файл | Temporal test MAE | Spatial DSP test MAE |",
                 "| --- | ---: | ---: |"]
    for f in files:
        ab = f["feature_ablation"]
        markdown.append(f'| {f["source_file"]} | '
                        f'{f["test_metrics"][ab["temporal"]]["mae_c"]:.6f} | '
                        f'{f["test_metrics"][ab["spatial_dsp"]]["mae_c"]:.6f} |')
    markdown += ["", "Разделение 60/20/20 по времени каждого файла. Масштабирование обучено на train.",
                 "Параметры и champion выбраны на validation. Test используется только для отчёта.",
                 "Записи уже исследовались в v1; результат v2 не является новым независимым тестом.",
                 "Каждый timestamp имеет одинаковый вес. Реальных меток аварий нет:",
                 "precision/recall не вычисляются. Задержка получения профиля неизвестна.",
                 "События рассчитаны только на тесте. Исходные CSV проверены SHA256 до и после."]
    (Path(output) / "results.md").write_text("\n".join(markdown) + "\n")
