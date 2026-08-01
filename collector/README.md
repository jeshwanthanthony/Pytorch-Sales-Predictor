# collector

**Phase 1.** Gets the information from Square, the weather service, and a
couple of files we keep by hand.

This folder only downloads and saves. It does not clean anything, add anything
up, or make any decisions. If the number is wrong here, it was wrong at Square.

## What comes in

| Source | Needs a key? | What we get |
| --- | --- | --- |
| Square API | yes, the OAuth token from `server.mjs` | orders, line items, payments, refunds, customers, catalog, inventory, staff, shifts |
| Open-Meteo | no | daily weather, history and forecast |
| `holidays` package | no | US and state holidays |
| `reference/*.csv` | no | local events and your promotions, typed in by hand |

## What it creates

Text files, one JSON object per line:

```
data/raw/orders/orders-20260731T170901Z.jsonl
data/raw/weather/weather-20260731T170901Z.jsonl
data/raw/calendar/calendar-20260731T170901Z.jsonl
...
```

Every run writes a **new** file and never edits an old one. That way, if we mess
up a later step, we can always start again from exactly what Square told us.

It also writes `data/collector-state.json`, which remembers the newest record we
have seen, so the next run only asks for what is new.

## Example

```bash
# what is this Square token actually allowed to read?
python -m collector.run --check

# first big download
python -m collector.run --since 2024-01-01

# later runs only get what is new
python -m collector.run
```

## What runs next

`database/` — it reads `data/raw/` and loads it into SQLite.

## Files here

| File | Job |
| --- | --- |
| `run.py` | the command you actually run, calls everything else |
| `config.py` | where the token and the restaurant's location come from |
| `http.py` | retries and page-by-page downloading |
| `storage.py` | writes the .jsonl files and remembers where we got to |
| `square_api.py` | all ten Square things we pull |
| `weather_api.py` | Open-Meteo, past and future |
| `calendar_api.py` | holidays, weekends, school breaks, paydays |
| `events.py` | reads the two CSV files |
| `reference/*.csv` | **edit these** — the sample rows are made up |
