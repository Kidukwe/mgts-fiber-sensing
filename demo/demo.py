"""Run the same reproducible event pipeline used by the baseline CLI.

See docs/temperature_events.md for the synthetic sample command.
"""

from mgts_sensing.event_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
