"""Leakage, physical semantics, model portability and joint anomaly behavior."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from mgts_sensing.dsp import spatial_features
from mgts_sensing.forecast import RidgeForecaster, predict_artifact
from mgts_sensing.io import iter_profiles
from mgts_sensing.solution import build_alerts, extract, make_samples, metric, run_file


def write_csv(path, count=60):
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dts_id', 'channel_id', 'timestamp', 'points'])
        for i in range(count):
            y = [20 + i * .02] * 8 + [-300] * 4
            writer.writerow(['dts', '1', f'2026-10-01T00:{i:02d}:00',
                             json.dumps(list(enumerate(y)))])


def test_confirmed_timezone_is_explicit_and_raw_preserved(tmp_path):
    path = tmp_path / '5.csv'
    write_csv(path)
    original = path.read_bytes()
    p = next(iter_profiles([path], naive_utc_offset_hours=3)).profile
    assert p.timestamp.isoformat() == '2026-10-01T00:00:00+03:00'
    assert p.timestamp_raw == '2026-10-01T00:00:00'
    assert p.timezone_raw is None
    assert p.utc_offset_seconds == 10800
    assert not next(iter_profiles([path])).profile.timezone_known
    assert path.read_bytes() == original


def test_spatial_dsp_preserves_narrow_peak():
    features = spatial_features(np.array([20, 20, 20, 40, 20, 20, 20]))
    assert features['median'] == 20
    assert features['hot'] == 20
    assert features['roughness'] > 0
    assert 0 <= features['high_frequency_fraction'] <= 1
    with pytest.raises(ValueError):
        spatial_features(np.array([np.nan]))


def test_history_causality_split_and_tail_handling(tmp_path):
    path = tmp_path / '5.csv'
    write_csv(path)
    records, _ = extract(path, 4)
    assert all(len(r['zones']) == 2 for r in records)
    assert all(z['median'] > 0 for r in records for z in r['zones'].values())
    x, y, values, persistence, _, meta, split, cadence = make_samples(records)
    assert x.shape[1] == 24
    assert cadence == 60
    assert max(m['profile_index'] for m, s in zip(meta, split) if s == 0) == 35
    assert min(m['profile_index'] for m, s in zip(meta, split) if s == 1) == 36
    assert np.allclose(y, values - persistence)
    assert all(m['forecast_origin'] < m['timestamp'] for m in meta)
    prefix = make_samples(records[:40])
    assert np.array_equal(prefix[0], x[:len(prefix[0])])
    # Future edits do not alter any existing feature or training target.
    records[-1]['zones'][0]['median'] = 10000
    changed = make_samples(records)
    assert np.array_equal(changed[0], x)
    assert np.array_equal(changed[1][split == 0], y[split == 0])


def test_gap_rejection_and_grid_change_reset_history(tmp_path):
    path = tmp_path / '5.csv'
    write_csv(path)
    records, _ = extract(path, 4)
    records[10] = {'invalid': True, 'row': 11}
    records[25]['zones'][0]['grid'] = 'changed'
    *_, meta, _, _ = make_samples(records)
    assert not any(10 <= m['profile_index'] <= 15 for m in meta)
    assert not any(25 <= m['profile_index'] <= 30 and m['zone_id'] == 0 for m in meta)
    assert any(m['profile_index'] == 25 and m['zone_id'] == 1 for m in meta)


def test_ridge_round_trip_and_train_only_scaling():
    x = np.arange(100).reshape(50, 2)
    y = np.c_[x[:, 0] * .2, x[:, 0] * .4, x[:, 0] * .1]
    model = RidgeForecaster.fit(x[:30], y[:30], alpha=.1)
    assert np.array_equal(model.mean, x[:30].mean(axis=0))
    loaded = RidgeForecaster.from_dict(json.loads(json.dumps(model.to_dict())))
    assert np.array_equal(model.predict(x[30:]), loaded.predict(x[30:]))
    assert np.mean(np.abs(model.predict(x[30:]) - y[30:])) < 1


def test_sustained_step_detected_even_when_predictor_adapts(tmp_path):
    path = tmp_path / '5.csv'
    write_csv(path)
    records, _ = extract(path, 4)
    _, _, values, persistence, _, meta, split, cadence = make_samples(records)
    test = split == 2
    values[test, 0] += 10
    # Perfectly adapted one-step predictor still leaves frozen-level evidence.
    alerts = build_alerts(meta, values, values.copy(), np.array([1., 1., 1.]),
                          split, cadence, '5.csv')
    assert alerts
    event = next(e for e in alerts if e['kind'] == 'heating')
    assert event['observations'] == 12
    assert event['duration_seconds'] == 660
    assert event['peak_extent_m'] == pytest.approx(8 * .0025)
    assert event['detected_at'] > event['start']
    assert event['priority_score'] > 0
    # One isolated innovation cannot be a confirmed event.
    unchanged = persistence.copy()
    unchanged[np.flatnonzero(test)[0], 0] += 100
    one = build_alerts(meta, unchanged, persistence, np.ones(3), split, cadence, '5.csv')
    assert not one


def test_profile_weighting():
    m = metric(np.array([2., 4., 9.]), np.zeros(3), np.array([1, 1, 2]))
    assert m['mae_c'] == 6


def test_end_to_end_portable_artifacts(tmp_path):
    path = tmp_path / '5.csv'
    write_csv(path)
    before = path.read_bytes()
    result, _ = run_file(path, tmp_path / 'out', width=4)
    model = json.loads((tmp_path / 'out/model.json').read_text())
    stored = np.load(tmp_path / 'out/predictions.npz')
    ridge = RidgeForecaster.from_dict(model['ridge'])
    assert ridge.predict(stored['features'][:, model['feature_indices']]).shape == stored['actual'].shape
    assert np.allclose(predict_artifact(model, stored["features"], stored["persistence"],
                                        stored["rolling"], stored["zone_id"]), stored["predicted"])
    assert result['event_precision'] is None
    assert result['raw_unchanged']
    assert model['semantics']['sample_spacing_m'] == .0025
    assert Path(tmp_path / 'out/diagnostics.png').stat().st_size > 1000
    assert path.read_bytes() == before


def test_forecast_overshoot_at_recovery_is_not_opposite_alarm(tmp_path):
    path = tmp_path / '5.csv'
    write_csv(path)
    records, _ = extract(path, 4)
    _, _, values, _, _, meta, split, cadence = make_samples(records)
    test_indices = np.flatnonzero(split == 2)
    forecast = values.copy()
    # A model trained on ordinary operation may overshoot after an excursion.
    # The observed level is normal, so this cannot justify physical cooling.
    forecast[test_indices, 0] += 10
    alerts = build_alerts(meta, values, forecast, np.ones(3), split, cadence, '5.csv')
    assert not alerts


def test_point_change_support_and_train_only_thresholds(tmp_path):
    from mgts_sensing.dsp import change_features

    f = change_features(np.array([0., 1., 1., 0., -1., -1., -1.]), .2, 7)
    assert f['fraction_hot'] == pytest.approx(2 / 7)
    assert f['longest_cold_fraction'] == pytest.approx(3 / 7)
    path = tmp_path / '5.csv'
    write_csv(path)
    records, _ = extract(path, 4)
    threshold = records[-1]['zones'][0]['change_threshold_c']
    # An extreme future value cannot change the training threshold.
    data = list(csv.reader(path.open()))
    points = json.loads(data[-1][-1])
    points[0][1] += 1000
    data[-1][-1] = json.dumps(points)
    with path.open('w', newline='') as handle:
        csv.writer(handle).writerows(data)
    changed, _ = extract(path, 4)
    assert changed[-1]['zones'][0]['change_threshold_c'] == threshold
    assert changed[-1]['zones'][0]['fraction_hot'] > 0


def test_different_support_sizes_are_valid_and_not_hardcoded(tmp_path):
    path = tmp_path / '5.csv'
    write_csv(path)
    for width in (2, 3, 4, 7, 16):
        records, _ = extract(path, width)
        samples = make_samples(records)
        assert samples[0].shape[1] == 24
        assert all(z['index_stop'] <= 8 for r in records for z in r['zones'].values())


def test_saved_model_inference_does_not_fit(tmp_path, monkeypatch):
    from mgts_sensing.inference import main as infer

    path = tmp_path / '5.csv'
    write_csv(path)
    trained = tmp_path / 'trained'
    run_file(path, trained, width=4)

    def forbidden(*args, **kwargs):
        raise AssertionError('Inference must never train a model')

    monkeypatch.setattr(RidgeForecaster, 'fit', forbidden)
    output = tmp_path.parent / (tmp_path.name + '-inference')
    assert infer(['--input', str(path), '--model', str(trained / 'model.json'),
                  '--output-dir', str(output)]) == 0
    report = json.loads((output / 'report.json').read_text())
    assert report['model_fitted'] is False
    assert report['thresholds_fitted'] is False
    assert report['forecast_rows'] > 0


def test_gap_is_not_a_longer_forecast_target(tmp_path):
    path = tmp_path / '5.csv'
    write_csv(path)
    records, _ = extract(path, 4)
    # A missing measurement creates a 120s transition on a 60s stream.
    del records[30]
    *_, meta, _, _ = make_samples(records)
    assert not any(30 <= m['profile_index'] <= 34 for m in meta)
    assert all((__import__('datetime').datetime.fromisoformat(m['timestamp']) -
                __import__('datetime').datetime.fromisoformat(m['forecast_origin'])).total_seconds()
               == 60 for m in meta)
