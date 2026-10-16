"""Validate a delivered experiment without retraining; raw data are optional."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .forecast import predict_artifact


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_output(output_dir, data_dir=None):
    from .solution import metric

    output = Path(output_dir)
    manifest = json.loads((output / 'solution.json').read_text())
    if (manifest.get('status') != 'complete' or
            manifest.get('schema_version') != 'building-temperature-v2'):
        raise ValueError('A complete building-temperature-v2 experiment is required')
    for name, digest in manifest['implementation_sha256'].items():
        if _hash(Path(__file__).with_name(name)) != digest:
            raise ValueError(f'Code changed since run: {name}; regenerate the experiment')
    for item in manifest['files']:
        name = item['source_file']
        if data_dir is not None and _hash(Path(data_dir) / name) != item['sha256']:
            raise ValueError(f'Raw input changed: {name}')
        directory = output / Path(name).stem
        artifact = json.loads((directory / 'model.json').read_text())
        if artifact['source_sha256'] != item['sha256']:
            raise ValueError(f'Model source fingerprint mismatch: {name}')
        with np.load(directory / 'predictions.npz', allow_pickle=False) as data:
            actual = data['actual']
            predicted = predict_artifact(artifact, data['features'], data['persistence'],
                                          data['rolling'], data['zone_id'])
            if (predicted.shape != actual.shape or not np.isfinite(actual).all()
                    or not np.isfinite(predicted).all()
                    or not np.allclose(predicted, data['predicted'])):
                raise ValueError(f'Saved model cannot reproduce test predictions: {name}')
            result = metric(actual[:, 0], predicted[:, 0], data['profile_index'])
            expected = item['test_metrics_selected']
            if (not np.isclose(result['mae_c'], expected['mae_c']) or
                    not np.isclose(result['rmse_c'], expected['rmse_c']) or
                    result['profiles'] != expected['profiles'] or
                    result['zone_observations'] != expected['zone_observations']):
                raise ValueError(f'Test metrics mismatch: {name}')
            if not all(datetime.fromisoformat(a) < datetime.fromisoformat(b)
                       for a, b in zip(data['forecast_origin'], data['timestamp'])):
                raise ValueError(f'Noncausal forecast timestamps: {name}')
        events = json.loads((directory / 'events.json').read_text())
        if len(events) != item['candidate_events']:
            raise ValueError(f'Candidate event count mismatch: {name}')
    if sum(f['candidate_events'] for f in manifest['files']) != len(manifest['events']):
        raise ValueError('Combined candidate event count mismatch')
    return manifest
