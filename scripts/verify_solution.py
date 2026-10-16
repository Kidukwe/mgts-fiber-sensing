"""Verify code, exported models, test predictions, metrics and optional raw data."""

import argparse
from pathlib import Path

from mgts_sensing.artifacts import verify_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/final'))
    parser.add_argument('--data-dir', type=Path, default=Path('MGTS_data'))
    parser.add_argument('--skip-raw', action='store_true',
                        help='Validate delivered models without the original CSV files')
    args = parser.parse_args()
    manifest = verify_output(args.output_dir, None if args.skip_raw else args.data_dir)
    for item in manifest['files']:
        print(f"{item['source_file']}: model, causal timestamps, test metrics and events verified")
    print('All artifacts verified.' + (' Raw checks skipped.' if args.skip_raw else ' Raw SHA verified.'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
