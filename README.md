# restaurant-forecast-ai

Forecast a restaurant's next day — revenue, covers, menu demand, staffing — from
its own Square history plus the outside world (weather, holidays, local events).

PyTorch is maybe 10% of this. The other 90% is getting good data, which is what
the pipeline below is for.

Production deployment is packaged under [`deploy/`](deploy/README.md). It uses
an Oracle Cloud Always Free ARM VM, Docker, a persistent data volume, and Caddy
for automatic HTTPS. Do not deploy this app to an ephemeral free web service:
connected Square tokens, SQLite warehouses, and trained models must survive
restarts.

```
Square API ──┐
Weather      ├──> collector/ ──> database/ ──> pipelines/ ──> training/
Calendar     │                                                    │
Events/promos┘                                                    ▼
                              dashboard/  <──  api/  <──  models/
```

| # | Folder | Job | Status |
| --- | --- | --- | --- |
| 0 | `server.mjs` | Square OAuth connect screen | done |
| 1 | `collector/` | get the data from Square, weather, calendar | done |
| 2 | `database/` | store it safely, no duplicates | done |
| 3 | `pipelines/` | check it, one row per day, build features | done |
| 4 | `training/` | teach the PyTorch model | done |
| 5 | `models/` | trained artifacts only | done |
| 6 | `api/` | serve the model over HTTP | done |
| 7 | `dashboard/` | show it to a human | done |

Each folder has its own README explaining what comes in, what goes out, and what
runs next.

## Run the whole thing

```bash
python -m api.serve                              # connect Square, then use the dashboard
.venv/bin/python -m collector.run --since 2024-01-01
.venv/bin/python -m database load
.venv/bin/python -m pipelines.validate_data      # stops here if the data is bad
.venv/bin/python -m pipelines.build_daily_summary
.venv/bin/python -m pipelines.build_features
.venv/bin/python -m training.train
.venv/bin/python -m training.evaluate
.venv/bin/python -m training.predict
.venv/bin/uvicorn api.main:app                   # http://localhost:8000
```

---

## Phase 0: connect Square

```bash
python -m api.serve
# http://localhost:8080 -> "Connect Square account"
```

The Python app serves the Square connect flow and the forecast dashboard from
the same local address. Each connected Square merchant gets its own folder under
`workspaces/`, with the OAuth token stored in `square.json` mode 600.

**Two things must be true before the button works:**

1. The Square Developer Dashboard for this application lists
   `http://localhost:8080/api/square/callback` as an OAuth Redirect URL on the
   **sandbox** tab, or Square returns `redirect_uri mismatch`.
2. Your `.env` has sandbox credentials from the **sandbox** tab:
   `SQUARE_APPLICATION_ID`, `SQUARE_APPLICATION_SECRET`, `SQUARE_ENVIRONMENT=sandbox`,
   and the exact same `SQUARE_REDIRECT_URL`.

Scopes requested (all read-only) — `MERCHANT_PROFILE_READ`, `ORDERS_READ`,
`PAYMENTS_READ`, `ITEMS_READ`, `CUSTOMERS_READ`, `INVENTORY_READ`,
`EMPLOYEES_READ`, `TIMECARDS_READ`. They must stay in sync with
`REQUIRED_SCOPES` in [collector/config.py](collector/config.py); a scope missing
there is a 403 at collection time.

> If you connected before the customer/inventory/labor scopes were added, click
> **Disconnect** and reconnect — an existing token keeps its original scopes.

---

## Phase 1: collect everything

```bash
.venv/bin/python -m collector.run --check          # which pulls does this token allow?
.venv/bin/python -m collector.run --since 2024-01-01   # first full backfill
.venv/bin/python -m collector.run                  # incremental, from stored watermarks
.venv/bin/python -m collector.run --only weather,calendar
.venv/bin/python -m collector.run --dry-run        # count rows, write nothing
```

### What it pulls

**From Square** — `collector/square_api.py`

| Entity | Notable columns |
| --- | --- |
| `locations` | coordinates, timezone, currency, state |
| `orders` | revenue / discount / tax / tip / service charge / net sales, source, fulfillment type, customer id, payment types |
| `order_items` | one row per line item: item, variation, quantity, price, modifiers |
| `payments` | amount, tip, processing fee, source type, card brand, entry method |
| `refunds` | amount, reason, status, destination |
| `customers` | email, phone, birthday, creation source, groups |
| `catalog` | items, categories, variations with prices, modifier lists |
| `inventory` | stock count per variation per location |
| `team_members` | status, assigned locations |
| `shifts` | start/end, hours, job title, hourly rate, derived labor cost |

**From outside** — because a model that only sees sales can't explain them

| Source | Module | Notes |
| --- | --- | --- |
| Weather | `weather_api.py` | Open-Meteo, no API key. Temp, feels-like, rain, snow, humidity, wind, plus `is_rainy`/`is_snowy`/`is_stormy`. Stitches the archive endpoint (history, ~6-day lag) to the forecast endpoint, which reaches 15 days ahead — requests past that are clipped, not rejected. |
| Calendar | `calendar_api.py` | Federal/state holidays, holiday eve and day-after, observances (Valentine's, NYE), day/month/quarter parts, weekend, school breaks, payday windows. No network — generates future dates too. |
| Events | `events.py` | Games, concerts, festivals from `collector/reference/events.csv` |
| Promotions | `events.py` | Ad spend, coupons, specials from `collector/reference/promotions.csv` |

Events and promotions are CSVs on purpose: ticketing APIs need keys and still
miss the block party two doors down, and no API knows you ran a $5-off Instagram
promo last Tuesday. A file the owner keeps current beats a feed that's 60% right.
Swap in a paid feed later behind `fetch_events` and nothing downstream changes.

**Replace the sample rows in `collector/reference/*.csv` with real ones** — they
ship as illustrative Arlington-area examples, not data.

### Design decisions worth knowing

- **Raw is append-only.** Each run writes `data/raw/<entity>/<entity>-<utc-stamp>.jsonl`
  and never rewrites a file. Dedupe and upsert belong to the database step, which
  means a bad transform is always replayable from what Square actually said.
- **Money stays in cents** (`*_cents`, int). Converting to dollars is a
  feature-engineering decision, not a storage one.
- **Timestamps stay UTC RFC-3339.** Local business dates get derived once, in
  features, using the location's timezone — a 1am Saturday order belongs to
  Friday's business day, and that rule lives in one place.
- **Incremental by watermark.** `data/collector-state.json` stores the newest
  `updated_at` seen per entity; the next run re-asks from 6 hours before it, since
  Square can back-date an update and duplicate rows are free.
- **Orders filter on `updated_at`, not `created_at`**, so an order that was
  comped or refunded days later gets re-collected.
- **One failing source never stops the run.** A missing scope costs you that
  entity, not the other thirteen.
- **External sources run past today** (`--forecast-days`, default 14). Predicting
  tomorrow needs tomorrow's weather and calendar as model inputs. Calendar needs
  no network so it reaches arbitrarily far ahead; weather stops at 15 days.
- **Customer visit counts and lifetime spend are deliberately not collected.**
  Square doesn't expose them; deriving them from our own order history is more
  accurate and free.

### Tests

```bash
.venv/bin/python -m pytest tests/ -q     # 105 passing
```

Covers order/line-item normalization, cursor pagination, holiday and weekend
logic, multi-day event expansion, and weather flags — everything that doesn't
need a live token.

---

## Configuration

`.env` (gitignored):

| Variable | Purpose |
| --- | --- |
| `SQUARE_APPLICATION_ID` | OAuth client id (public) |
| `SQUARE_APPLICATION_SECRET` | OAuth client secret — server-side only |
| `SQUARE_ENVIRONMENT` | `sandbox` or `production` |
| `SQUARE_REDIRECT_URL` | must match the Redirect URL in the Square Developer Dashboard |
| `SQUARE_WEBHOOK_SIGNATURE_KEY` | not used yet; kept for a future webhook ingest |
| `PORT` | defaults to `8080` |
| `SITE_LATITUDE` / `SITE_LONGITUDE` | optional; otherwise taken from the Square location, geocoding its postal code if needed |
| `SITE_TIMEZONE` / `SITE_COUNTRY` / `SITE_STATE` | optional; `SITE_STATE` picks up state holidays |
| `SQUARE_ACCESS_TOKEN` | optional; bypasses `.square-tokens.json` for server deployments |

Switching to production means new credentials — production and sandbox apps do
not share an application secret.

---

## Phase 2: the database

```bash
.venv/bin/python -m database load       # load everything new from data/raw/
.venv/bin/python -m database stats      # row counts, date span, file size
.venv/bin/python -m database check      # gaps and orphans that would poison training
.venv/bin/python -m database features   # preview the training table
.venv/bin/python -m database sales      # recent daily totals
.venv/bin/python -m database items      # best sellers
```

One SQLite file at `data/forecast.db`. 17 tables, 8 views, WAL mode.

### How it's stored

**Types.** Money is `INTEGER` cents, never a float — `0.1 + 0.2` has no place in
a ledger. Timestamps are Square's RFC 3339 UTC text and dates are `YYYY-MM-DD`,
because ISO-8601 sorts correctly as text, so a date-range scan uses the index
with no conversion. Booleans are 0/1. Every table is `STRICT`, so SQLite rejects
a wrong-typed value at write time instead of silently keeping `"4521"` as text
in a money column.

**Keys.** Primary keys are the upstream Square ids (`order_id`, `payment_id`,
`(order_id, line_item_uid)`, …). Loading is `INSERT ... ON CONFLICT DO UPDATE`,
so re-running the loader is a no-op and an order Square re-sent because it was
refunded overwrites the stale copy in place. `ingested_files` tracks which raw
files are already in, so a re-run does no redundant work at all.

**Two transformations happen on write**, once per row instead of once per query:

- `business_date` — a UTC timestamp becomes the local trading day, with a 4am
  cutoff (`--cutoff-hour`), so a 1:30am Saturday order counts as Friday's
  business. Timezones are per-location, so a second location in another zone
  stays correct without touching a query.
- Flattening — line-item modifiers and catalog variations become their own
  tables rather than JSON blobs, because they get grouped by. "How often is
  Butter Chicken ordered extra spicy" is then an index scan.

**Views, not materialized tables** — `daily_sales`, `daily_item_sales`,
`daily_labor`, `daily_payments`, `daily_refunds`, `daily_events`,
`daily_promotions`, and `daily_forecast_features`. They run in single-digit
milliseconds against the indexes, so a stale copy would be the bigger problem.

### The training table

`daily_forecast_features` is one row per business day per location, 68 columns,
45 of them model features. Its spine is `calendar_days`, not `orders` — a day
the restaurant was closed must appear as a zero, and future dates must appear so
tomorrow can be predicted. `is_observed` separates "closed, sold nothing" from
"hasn't happened yet".

Lags and rolling means are computed in SQL window functions, every frame ending
at `1 PRECEDING`, so **no history feature can leak the day's own answer** into
training. That's asserted in the tests, not just intended.

```python
from database.queries import feature_frame, FEATURE_COLUMNS
import torch

df = feature_frame(drop_warmup=True)          # pandas DataFrame
X = torch.tensor(df[FEATURE_COLUMNS].values, dtype=torch.float32)
y = torch.tensor(df["target_sales_cents"].values, dtype=torch.float32).reshape(-1, 1)
```

`drop_warmup` matters: the 14-day lag means the first 14 days of any history
have nothing to look back at. Those NULLs are honest, but they become silent
NaNs in a tensor.

Use `prediction_rows()` for the future dates — same columns, target unknown,
weather present as a forecast. That is the whole reason the collector reaches
past today.

### Tests

```bash
.venv/bin/python -m pytest tests/ -q     # 105 passing
```

`tests/test_database.py` runs the full load against synthetic Square-shaped raw
files — the Square API can't be reached without a token, so the fixtures use the
exact record shape `square_api.py` emits. They assert the 4am cutoff, upsert
idempotency, canceled orders staying out of revenue, and lag correctness.

---

## Phase 3: the pipelines

```bash
.venv/bin/python -m pipelines.validate_data        # gate: exits 1 if unsafe to train
.venv/bin/python -m pipelines.build_daily_summary  # one clean row per day
.venv/bin/python -m pipelines.build_features       # lags, encodings, split, scaling
```

Run in that order. The database stores what happened; these decide what is fit
to learn from, roll it up by day, and produce the arrays PyTorch trains on.

### `validate_data.py` — the gate

`database check` reports numbers; this decides whether they're acceptable and
**exits non-zero when they aren't**. That difference is the whole point: a model
trained on a week where the collector silently missed three days still produces
a confident number, and nothing downstream ever tells you it's wrong.

14 checks at three severities. Blocking (`error`): no orders, missing
`business_date`, negative money, `net ≠ revenue − tax − tip`, holes in the
calendar spine, orders dated in the future. Advisory (`warning`): short history,
long runs of zero-sales days, missing weather, past days still sitting on
*forecast* weather rather than reanalysis, 5-sigma outliers. `--strict` promotes
warnings to errors.

### `build_daily_summary.py` — one row per trading day

Materializes a `daily_summary` table that goes past the `daily_sales` view in
four ways:

- every calendar date appears, **including days the restaurant was shut**, so a
  closure reads as a zero rather than a hole in the series
- dayparts — lunch / dinner / late — from each order's local hour, plus peak hour
- new vs returning customers, from each customer's first ever order
- operating ratios (sales per labor hour, labor cost %, refund rate) in basis
  points so they stay integers, with zero denominators answered deliberately
  rather than left as a NULL that becomes NaN in a tensor

A table rather than a view because `build_features` reads it repeatedly and the
first-visit logic is the expensive part. Rebuilds are idempotent.

### `build_features.py` — the model-ready arrays

This is the step that **cannot** be a SQL view, for one reason: the scaler must
be fit on the training split alone. Standardising over the whole dataset leaks
the test period's mean into training and quietly flatters every metric you then
report. So: split first, statistics from train only, same numbers saved for
inference to reuse.

Engineering SQL is clumsy at:

| Feature | Why |
| --- | --- |
| `dow_sin/cos`, `month_sin/cos`, `doy_sin/cos` | Sunday(7) vs Monday(1) is a distance of 6 to a network; on a circle it's 1, which is what it actually is |
| `days_to_holiday`, `days_since_holiday` | the week *before* Thanksgiving doesn't trade normally, and a binary `is_holiday` can't say so |
| `same_dow_mean_4` | the same weekday over past weeks — the strongest signal in most restaurant series, which a 7-day mean blurs away |
| `momentum_7_28` | is the last week running above or below the last month |
| `temp_vs_normal_f` | 85°F in April isn't 85°F in August |
| `rain_x_weekend`, `promo_x_weekend` | crosses worth stating outright |

**The leakage guard.** `SAME_DAY_OUTCOMES` lists every column only knowable once
the day is over — `order_count`, `tip_cents`, `labor_hours`, `peak_hour`, and 30
more. `assert_no_leakage()` raises `LeakageError` if any reaches the model.
Verified empirically too: `sales_lag_1` matches the previous day's actual on
every row, and no feature correlates above 0.73 with the target.

**Splits are chronological, never random** — a random split trains on the
future. `--val-days 0 --test-days 0` (the default) sizes holdouts to the history
available: 15% each, floored at 7 days, capped at 28. A fixed 28/28 is right for
two years and absurd for three months.

Rows whose history is incomplete are **dropped, not imputed** — a filled-in lag
is a fabricated fact about the past.

Outputs:

```
data/features/dataset.npz     X/y per split + dates + feature names
data/features/manifest.json   feature names, scaler stats, split boundaries
features table in the database (unscaled JSON payload, for inspection)
```

```python
from pipelines.build_features import load_dataset
import torch

d = load_dataset()
X = torch.tensor(d["X_train"])   # (n, 68)
y = torch.tensor(d["y_train"])   # (n, 1)
```

The npz is what trains; the `features` table is what you read when a prediction
looks wrong and you want to see exactly what the model was shown.

### Tests

```bash
.venv/bin/python -m pytest tests/ -q     # 105 passing
```

`tests/test_pipelines.py` builds ~70 days of synthetic trading through the real
loader, so it exercises the actual SQL and the actual pandas: closed days,
daypart boundaries, new-vs-returning customers, scaler isolation, chronological
splits, and the leakage guard.

---

## Phase 4: training

```bash
.venv/bin/python -m training.train      # learn
.venv/bin/python -m training.evaluate   # model vs baseline, in dollars
.venv/bin/python -m training.predict    # tomorrow
```

A small feed-forward network — `68 features -> 64 -> relu -> dropout -> 32 ->
relu -> 1`, about 6,500 parameters. No LSTM: with a year or two of daily rows
there isn't the data for one, the lag features already supply the history, and a
small model is far easier to debug and explain.

Trains in batches with a DataLoader, checks validation every epoch under
`torch.no_grad()`, stops early after 40 epochs without improvement, and
**restores the best weights before the test split is ever touched**.

Six rules are enforced in `dataset.py`, and break the run rather than degrade
quietly: the target is never an input, same-day outcomes are blocked, the splits
must not overlap in time, the pipeline's scaler is reused and never re-fit, the
test split is only read after training finishes, and seeds are set.

**The baseline is "sales 7 days ago."** It sounds trivial and it is genuinely
hard to beat, because the same weekday last week already carries most of the
signal. Scores are MAE / RMSE / MAPE in **dollars**, since "0.28 MSE in scaled
units" tells a restaurant owner nothing.

Everything needed to repeat a prediction goes into `data/models/model.pt`:
weights, model config, feature list, both scalers, the row counts, and the split
dates.

See [training/README.md](training/README.md).

---

## Phase 5: models

Storage only, no code. `training/` writes `model.pt` here and `api/` reads it.

The checkpoint is a bundle, not just weights: it carries the model config, the
feature list and its exact order, both scalers, and the validation residuals used
for the prediction interval. Load only the weights and you get a different
answer — that is why all of it travels together.

These files are gitignored. They are build outputs; rebuild with
`python -m training.train`. See [models/README.md](models/README.md).

---

## Phase 6: api

```bash
.venv/bin/uvicorn api.main:app --reload
# http://localhost:8000/       dashboard
# http://localhost:8000/docs   generated API docs
```

| Route | What you get |
| --- | --- |
| `GET /health` | is a model loaded, which version, when it was trained |
| `GET /predict` | tomorrow's sales, its range, confidence, what drove it |
| `GET /history?days=30` | actual vs predicted on days never trained on |
| `GET /metrics` | test scores and the baseline comparison |

**The model loads once, at startup.** Reading a checkpoint takes far longer than
the forward pass, so loading per request would make every call slow for nothing.
If loading fails the server still boots and `/health` reports why.

**No authentication yet** — fine on a laptop, not fine on the internet.

See [api/README.md](api/README.md).

---

## Phase 7: dashboard

One page, three files, no framework and no build step. Shows tomorrow's
prediction with its range, the features that drove it, actual vs predicted for
the last 30 days, and whether the model beat its baselines.

The chart is hand-drawn SVG in about 25 lines — a chart library would be 300kB
for two lines on one screen.

See [dashboard/README.md](dashboard/README.md).

---

## Known limitations

- **The Square half has never run against a live token.** Normalization,
  pagination and money handling are unit-tested against realistic payloads, but
  no real order has passed through. Everything downstream has been exercised with
  synthetic data.
- **The numbers in this README come from synthetic sales.** Weather and calendar
  are real; the orders are generated. Treat the metrics as proof the wiring
  works, not as forecast quality.
- **One location.** The schema and features are keyed by location, but nothing
  has been tested with two.
- **Events and promotions are hand-typed CSVs.** If nobody keeps them current,
  those features are dead weight.
- **The order estimate is arithmetic, not a model** — predicted sales divided by
  the recent average ticket. It inherits every error the sales prediction has.
- **The confidence interval assumes the future looks like validation.** A new
  competitor or a menu change breaks that assumption silently.
- **No retraining schedule.** The model goes stale as the restaurant changes;
  something needs to re-run this pipeline on a cadence.
