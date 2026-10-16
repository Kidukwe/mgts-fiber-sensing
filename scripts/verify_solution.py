"""Verify delivered artifacts against raw files, saved models and test predictions."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from mgts_sensing.forecast import predict_artifact
from mgts_sensing.solution import metric, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/final'))
    parser.add_argument('--data-dir', type=Path, default=Path('MGTS_data'))
    args = parser.parse_args()
    manifest = json.loads((args.output_dir / 'solution.json').read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('Experiment is not complete')
    import mgts_sensing
    package = Path(mgts_sensing.__file__).parent
    for name, digest in manifest['implementation_sha256'].items():
        assert sha256(package / name) == digest, f'Code changed since run: {name}'
    for item in manifest['files']:
        name = item['source_file']
        assert sha256(args.data_dir / name) == item['sha256'], f'Input changed: {name}'
        directory = args.output_dir / Path(name).stem
        model = json.loads((directory / 'model.json').read_text())
        with np.load(directory / 'predictions.npz', allow_pickle=False) as data:
            actual = data['actual']
            prediction = predict_artifact(model, data['features'], data['persistence'], data['rolling'], data['zone_id'])
            assert np.isfinite(prediction).all()
            assert np.allclose(prediction, data['predicted']), f'Cannot reproduce predictions: {name}'
            metrics = metric(actual[:, 0], prediction[:, 0], data['profile_index'])
            assert np.isclose(metrics['mae_c'], item['test_metrics_selected']['mae_c'])
            assert metrics['zone_observations'] == item['test_metrics_selected']['zone_observations']
            assert all(datetime.fromisoformat(a) < datetime.fromisoformat(b)
                       for a, b in zip(data['forecast_origin'], data['timestamp']))
        print(f'{name}: raw SHA, model inference, causal timestamps and test metrics verified')
    print('All artifacts verified.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
