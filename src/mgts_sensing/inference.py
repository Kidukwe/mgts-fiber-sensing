"""Replay new profiles with a saved model, without fitting or calibrating anything."""

import argparse
import json
from pathlib import Path

import numpy as np

from .forecast import predict_artifact
from .solution import SEMANTICS, build_alerts, dump, extract, make_samples, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    source = args.input.resolve()
    if output == source.parent or source.parent in output.parents:
        parser.error('Output must be outside the raw input directory')
    if output.exists() and any(output.iterdir()):
        parser.error('Choose an empty output directory')
    artifact = json.loads(args.model.read_text())
    if artifact.get('schema_version') != 'building-temperature-v2':
        parser.error('A building-temperature-v2 model is required')
    output.mkdir(parents=True, exist_ok=True)
    before = sha256(source)
    records, issues = extract(source, artifact['zone_width_points'],
                              issue_path=output / 'issues.jsonl',
                              change_thresholds=artifact['point_change_thresholds_c'],
                              cadence_override=artifact['cadence_seconds'])
    streams = {tuple(record['stream']) for record in records if not record.get('invalid')}
    if not streams <= {tuple(s) for s in artifact['allowed_streams']}:
        parser.error('Input sensor/channel is not covered by the saved model')
    x, _, current, persistence, rolling, meta, _, cadence = make_samples(
        records, cadence_override=artifact['cadence_seconds'])
    prediction = predict_artifact(artifact, x, persistence, rolling,
                                  [m['zone_id'] for m in meta])
    events = build_alerts(meta, current, prediction, np.array(artifact['thresholds_c']),
                          np.full(len(meta), 2), cadence, source.name)
    dump(output / 'events.json', events)
    with (output / 'forecasts.jsonl').open('w', encoding='utf-8') as handle:
        for row, actual, predicted in zip(meta, current, prediction):
            item = {**row, 'predicted_median_c': float(predicted[0]),
                    'observed_median_c': float(actual[0]),
                    'forecast_residual_c': float(actual[0] - predicted[0])}
            handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
    unchanged = before == sha256(source)
    report = {'schema_version': 'building-temperature-v2-inference', 'semantics': SEMANTICS,
              'input_sha256': before, 'model_sha256': sha256(args.model), 'raw_unchanged': unchanged,
              'model_fitted': False, 'thresholds_fitted': False,
              'issues': issues, 'profiles': len(records), 'forecast_rows': len(meta),
              'candidate_events': len(events),
              'mode': 'batch_replay_of_saved_model; historical replay is not independent validation'}
    dump(output / 'report.json', report)
    print(f'{len(meta)} forecast rows, {len(events)} candidate events: {output}')
    return int(not unchanged or any(record.get('invalid') for record in records))
