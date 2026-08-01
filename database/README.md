# Database

This is phase 2. It loads the raw files into one SQLite database and safely
updates records Square has changed.

```bash
.venv/bin/python -m database load
.venv/bin/python -m database stats
.venv/bin/python -m database check
.venv/bin/python -m database sales
```

The database lives at `data/forecast.db`. Square IDs prevent duplicates, money
is stored as integer cents, and UTC timestamps are converted into each
restaurant's local business day. The check command catches missing or broken
data before training starts.
