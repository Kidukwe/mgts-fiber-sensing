"""Audit the building-temperature dataset with its confirmed fixed UTC+3 offset."""
import sys

from validate_data import main

if __name__ == '__main__':
    if '--naive-utc-offset-hours' not in sys.argv:
        sys.argv.extend(['--naive-utc-offset-hours', '3'])
    raise SystemExit(main())
