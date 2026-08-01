# database

**Phase 2.** Stores the raw information safely and stops duplicates.

The collector's files are just text. This folder puts them into a real database
so we can ask questions like "how much did we sell last Friday" without reading
every file again.

## The duplicate problem, and how we solve it

The collector re-downloads orders that changed (refunds, comps). So the same
order arrives more than once. If we just added every row we would count that
order twice and every total would be wrong.

The fix: **the primary key is Square's own order id.** When the same id shows up
again we update that row instead of adding a new one. Re-running the loader as
many times as you like changes nothing.

## What comes in

`data/raw/*.jsonl` from the collector.

## What it creates

`data/forecast.db` — one SQLite file. 17 tables and 8 views.

Two things get worked out while loading, so no query has to redo them:

- **business_date** — a 1:30am Saturday order belongs to Friday's business.
  We convert UTC to local time and use a 4am cutoff.
- **flattening** — modifiers and catalog variations get their own tables, so
  "how often do people order extra spicy" is a fast lookup.

Money is stored as whole cents (integers), never decimals. Money and floats do
not mix.

## Example

```bash
python -m database load     # load everything new
python -m database stats    # how many rows do we have?
python -m database sales    # recent daily totals
```

```sql
-- one row per day, ready to look at
SELECT business_date, order_count, net_sales_cents FROM daily_sales
ORDER BY business_date DESC LIMIT 7;
```

## What runs next

`pipelines/` — it reads these tables and builds the model features.

## Files here

| File | Job |
| --- | --- |
| `schema.sql` | every table, index, and view |
| `db.py` | connecting, settings, and the update-or-insert helper |
| `load.py` | reads the .jsonl files into the tables |
| `queries.py` | the queries other folders call |
| `__main__.py` | the `python -m database ...` commands |
