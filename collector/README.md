# Collector

This is phase 1. It downloads read-only Square data and adds weather, holidays,
events, and promotions that may affect sales.

```bash
.venv/bin/python -m collector.run --check
.venv/bin/python -m collector.run --since 2024-01-01
.venv/bin/python -m collector.run
```

The first command checks Square permissions. The second does a history import,
and later runs only request newer changes.

Raw responses are saved as JSONL under `data/raw/`. They stay unchanged so a
bad database or model run can be rebuilt without downloading everything again.
Money stays in cents, and timestamps stay in UTC.

Sample local events and promotions are in `collector/reference/`. Replace them
with information for the restaurant's real area.
