# dashboard

**Phase 7, the last one.** One page that shows the forecast to a human.

Three files, no framework, no build step, no npm. Open the page and it calls the
API. That is the whole thing.

## What it shows

- **tomorrow's predicted sales** with its range, e.g. `$1,263`, `80% range $1,074 to $1,468`
- **confidence**, plus how much the prediction wobbles when dropout is left on
- **estimated orders** — derived from the recent average ticket, not predicted directly
- **what drove this prediction** — the features that moved the answer most
- **actual vs predicted**, last 30 days, only on days the model never trained on
- **model scores** on the test split and whether it beat both baselines

## Run it

The API serves this page, so there is nothing separate to start:

```bash
uvicorn api.main:app --reload
# http://localhost:8000/
```

## What comes in / what goes out

**In:** JSON from `/health`, `/predict`, `/history`, `/metrics`.
**Out:** pixels. It stores nothing and computes nothing.

## Files here

| File | Job |
| --- | --- |
| `index.html` | the layout, no logic |
| `app.js` | fetches the four endpoints and fills the page in |
| `styles.css` | looks, including a dark and light theme |

## Why the chart is hand-drawn

`app.js` builds the line chart as raw SVG in about 25 lines. Pulling in a chart
library would be 300kB and one more thing to keep updated, for two lines on one
screen. If the dashboard grows, that trade changes.

## Reading the chart

Solid grey is what actually happened. Dashed blue is what the model predicted for
that day. The days shown are validation and test days only, so the model was
never trained on them — if the two lines track each other here, that is real.

## Who calls this next

Nobody. This is the end of the pipeline.
