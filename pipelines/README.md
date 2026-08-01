# pipelines

**Phase 3.** Checks the data, makes one row per day, and builds the numbers
the model will learn from.

The database knows about individual orders. The model needs whole days, with the
weather and the calendar attached, and with history it is allowed to look at.

## Three steps, in order

### 1. `validate_data.py` — is this data safe to learn from?

Runs 14 checks and **exits with an error code if any serious one fails**. This
matters more than it sounds. If the collector quietly missed three days, a model
will still train and still give confident answers, and nothing will warn you.

Serious (stops everything): no orders, missing dates, negative money, totals that
do not add up, gaps in the calendar, orders dated in the future.
Just a warning: not much history yet, missing weather, unusually big days.

### 2. `build_daily_summary.py` — one clean row per day

Adds up orders into a `daily_summary` table. Days the restaurant was **closed**
still get a row, with zeros. That matters: a missing row would look like a gap in
time, but a zero is the truth.

Also works out lunch vs dinner counts, the busiest hour, and which customers were
new versus returning.

### 3. `build_features.py` — the numbers the model gets

Adds the things a network needs but SQL is awkward at:

- **lags** — sales 1, 2, 3, 7, 14 and 28 days ago
- **rolling averages** — last 7 days, last 28 days
- **circle encoding** — Sunday and Monday are next to each other, not 6 apart
- **holiday distance** — the week before Thanksgiving is not a normal week

Then it does two things that must happen in this order:

1. **splits by date** — oldest days to train, then validation, then the newest
   days for the test. Never shuffled: shuffling would let the model see the
   future while it learns.
2. **fits the scaler on the training days only.** If you scale using all the
   data, the test period's average leaks into training, and every score you
   report afterwards is flattering and wrong.

There is also a guard: `SAME_DAY_OUTCOMES` lists everything you only know after
the day is over (`order_count`, `tip_cents`, `labor_hours`, ...). If any of those
reach the model, the build stops with a `LeakageError`.

## What comes in

`data/forecast.db` from the database folder.

## What it creates

```
daily_summary table   in the database
features table        in the database (readable, for checking)
data/features/dataset.npz     the arrays the model trains on
data/features/manifest.json   feature names, scaler numbers, split dates
```

## Example

```bash
python -m pipelines.validate_data        # stop if the data is bad
python -m pipelines.build_daily_summary  # one row per day
python -m pipelines.build_features       # the model's numbers
```

## What runs next

`training/` — it reads `dataset.npz` and `manifest.json`.
