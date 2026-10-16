"""Controlled end-to-end anomaly benchmark; never report as real accident accuracy."""

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from mgts_sensing.solution import dump, run_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/benchmark'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise-c', type=float, default=.05)
    parser.add_argument('--amplitude-c', type=float, default=5.)
    args = parser.parse_args()
    if args.noise_c < 0 or args.amplitude_c < 0:
        parser.error('Noise and amplitude must be nonnegative')
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        parser.error('Use a new empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=3)))
    # All four events lie within the final 20%, which is never fitted.
    truth = [('heating', 245, 251, 0), ('cooling', 255, 261, 1),
             ('local_peak', 265, 271, 2), ('local_dip', 275, 281, 3)]
    source = output / 'input'
    source.mkdir()
    path = source / 'synthetic.csv'
    rng = np.random.default_rng(args.seed)
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dts_id', 'channel_id', 'timestamp', 'points'])
        for i in range(300):
            y = 20 + rng.uniform(-args.noise_c, args.noise_c, 32)
            for kind, a, b, zone in truth:
                if a <= i <= b:
                    if kind in ('heating', 'cooling'):
                        y[zone * 8:(zone + 1) * 8] += args.amplitude_c if kind == 'heating' else -args.amplitude_c
                    else:
                        y[zone * 8 + 3] += args.amplitude_c if kind == 'local_peak' else -args.amplitude_c
            writer.writerow(['synthetic', '1', (start + timedelta(minutes=i)).isoformat(),
                             json.dumps(list(enumerate(y.tolist())))])
    result, alerts = run_file(path, output / 'result', width=8)
    matched = set()
    true_positive = 0
    delays = []
    for kind, a, b, zone in truth:
        for j, e in enumerate(alerts):
            begin = (datetime.fromisoformat(e['start']) - start).total_seconds() / 60
            end = (datetime.fromisoformat(e['end']) - start).total_seconds() / 60
            if (j not in matched and e['kind'] == kind and begin <= b and end >= a
                    and e['index_start'] <= zone * 8 < e['index_stop']):
                matched.add(j)
                true_positive += 1
                delays.append((datetime.fromisoformat(e['detected_at']) -
                               (start + timedelta(minutes=a))).total_seconds())
                break
    metrics = {'dataset': 'synthetic_only_not_real_accident_accuracy', 'seed': args.seed, 'noise_c': args.noise_c, 'amplitude_c': args.amplitude_c, 'evaluation_role': 'development_scenarios_not_independent_test',
               'injected_events': len(truth), 'detected_candidates': len(alerts),
               'matched_events': true_positive,
               'precision': true_positive / len(alerts) if alerts else 0,
               'recall': true_positive / len(truth),
               'detection_delay_seconds': delays, 'forecast_test_metrics': result['test_metrics_selected'],
               'matching': 'one-to-one same kind, overlapping time and injected zone',
               'truth': [{'kind': k, 'start_profile': a, 'end_profile': b, 'zone': z}
                         for k, a, b, z in truth]}
    dump(output / 'benchmark.json', metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
