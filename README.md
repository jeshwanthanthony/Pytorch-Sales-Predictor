# Pytorch-Sales-Predictor


This is a small sales forecasting app for restaurants that use Square. A user
connects their Square account, the app cleans their sales history, and a
PyTorch model predicts the next day's sales.

The dashboard shows each step like a terminal so it is easy to see where the
data is going or where something failed.

## How it works

1. **Connect:** Square OAuth gives the app read-only access to the restaurant.
2. **Collect:** Orders, payments, refunds, weather, and calendar data are saved.
3. **Clean:** SQLite removes duplicates and organizes everything by business day.
4. **Prepare:** Pandas builds daily totals, lag values, rolling averages, and
   other model features.
5. **Train:** PyTorch uses train, validation, and test dates. Early stopping
   keeps the best epoch instead of the last one.
6. **Predict:** The app returns tomorrow's sales, an 80% range, expected orders,
   confidence, and the main feature values behind the forecast.

The neural network is intentionally small:

```text
68 inputs -> 64 -> 32 -> 1 sales prediction
```

The final test compares it with two simple guesses: last week's sales and the
last seven-day average. The app reports the real scores even when a simple
baseline wins.

## Run locally

```bash
cd /Users/jeshwanthanthony/restaurant-forecast-ai
.venv/bin/python -m api.serve
```

Open `http://localhost:8080` and click **Connect Square**.

Your `.env` needs:

```env
SQUARE_APPLICATION_ID=...
SQUARE_APPLICATION_SECRET=...
SQUARE_ENVIRONMENT=sandbox
SQUARE_REDIRECT_URL=http://localhost:8080/api/square/callback
PORT=8080
```

For real Square sellers, use production credentials and a public HTTPS callback.
The redirect URL must exactly match the one in the Square Developer Dashboard.

## Run phases by hand

```bash
.venv/bin/python -m collector.run --since 2024-01-01
.venv/bin/python -m database load
.venv/bin/python -m pipelines.validate_data
.venv/bin/python -m pipelines.build_daily_summary
.venv/bin/python -m pipelines.build_features
.venv/bin/python -m training.train
.venv/bin/python -m training.evaluate
.venv/bin/python -m training.predict
```

Generated Square data, databases, tokens, and model files stay out of Git.
Deployment notes are in [`deploy/README.md`](deploy/README.md).

## Results

The full test suite currently has 174 passing tests. Forecast quality depends on
how much real sales history the connected restaurant has, so the model metrics
are shown in the app instead of promising a fake accuracy number.
