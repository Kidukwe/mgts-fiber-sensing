"""Package source and finished experiment, excluding raw data and environments."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    root = Path(__file__).resolve().parents[1]
    final = root / 'artifacts/final'
    from mgts_sensing.artifacts import verify_output
    verify_output(final)
    target = root / 'artifacts/submission.zip'
    paths = [root / name for name in ('README.md', 'pyproject.toml', 'LICENSE', 'CONTRIBUTING.md', '.gitignore',
                                           'notebooks/03_building_monitoring.ipynb')]
    for name in ('src', 'scripts', 'configs', 'docs', 'reports', 'tests', 'data/sample', 'demo',
                 '.github', 'artifacts/final'):
        paths.extend((root / name).rglob('*'))
    # Only intentional deliverables; no cache, environment, raw or hidden user files.
    allowed = {'.py', '.md', '.toml', '.yaml', '.yml', '.json', '.jsonl', '.csv', '.html', '.png', '.npz', '.ipynb'}
    with ZipFile(target, 'w', compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(paths)):
            if (path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts
                    and (path.suffix in allowed or path.name in ('LICENSE', '.gitignore'))):
                archive.write(path, path.relative_to(root))
    print(f'{target} ({target.stat().st_size / 1024 / 1024:.1f} MiB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
