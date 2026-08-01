# api

**Phase 6.** Serves the trained model over HTTP.

No machine learning happens here. Everything was learned in `training/`; this
folder loads the result and answers questions about it.

## The one idea worth remembering

**The model is loaded once, when the server starts.** Reading a checkpoint off
disk takes far longer than the forward pass, so loading it inside a request
handler would make every single call slow for no reason. FastAPI's `lifespan`
hook does the loading, and every request reuses the object in memory.

If the model fails to load the server still starts. `/health` then tells you
what went wrong instead of the whole thing refusing to boot.

## Run it

```bash
uvicorn api.main:app --reload
```

- http://localhost:8000/ — the dashboard
- http://localhost:8000/docs — generated API docs you can click through

## Endpoints

| Route | What you get |
| --- | --- |
| `GET /health` | is a model loaded, which version, when it was trained |
| `GET /predict` | tomorrow's sales, the range, confidence, and what drove it |
| `GET /history?days=30` | actual vs predicted on days the model never trained on |
| `GET /metrics` | test scores and the baseline comparison |

### Example

```bash
curl localhost:8000/predict
```

```json
{
  "business_date": "2026-07-02",
  "predicted_sales": 1262.79,
  "interval_low": 1073.61,
  "interval_high": 1467.83,
  "interval_label": "80% range",
  "confidence": 0.688,
  "model_uncertainty": 94.34,
  "estimated_orders": 54,
  "important_features": [
    { "name": "sales_roll_mean_7", "value": 1136.57, "contribution": 0.071, "direction": "down" }
  ],
  "model_version": "1.0.0"
}
```

## What comes in / what goes out

**In:** `models/model.pt`, `models/metrics.json`, and the feature file at
`data/features/`.
**Out:** JSON. Nothing is written to disk.

## Files here

| File | Job |
| --- | --- |
| `main.py` | the routes, and loading the model at startup |
| `service.py` | holds the model in memory and builds the answers |
| `schemas.py` | the response shapes, which also generate the docs |

## Not done yet

**There is no authentication.** Anyone who can reach the port can call it. That
is fine on your laptop and not fine on the internet. Before deploying you would
want an API key or a token check, rate limiting, and CORS rules.

## Who calls this next

`dashboard/` — it is plain HTML and JavaScript that calls these endpoints.
