# Pipelines

This is phase 3. It turns the SQLite data into a table PyTorch can learn from.

```bash
.venv/bin/python -m pipelines.validate_data
.venv/bin/python -m pipelines.build_daily_summary
.venv/bin/python -m pipelines.build_features
```

The phases are simple:

1. Validate dates, totals, and required history.
2. Build one Pandas row for each restaurant day.
3. Add weather, calendar, lag, and rolling-average features.
4. Split dates into train, validation, and test sets without mixing future data
   into the past.

The output is `data/features/dataset.npz` plus a manifest describing the rows,
features, dates, and scaler. Training stops if this data contract is invalid.
