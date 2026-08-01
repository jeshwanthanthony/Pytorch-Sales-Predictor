# API

This folder runs the FastAPI backend. It handles Square OAuth, starts the data
pipeline, loads the latest model, and sends progress logs to the dashboard.

```bash
.venv/bin/python -m api.serve
```

Open `http://localhost:8080`. Useful routes include:

- `/api/square/connect` starts Square OAuth.
- `/api/square/callback` finishes the connection.
- `/api/setup/start` starts collection, cleaning, training, and prediction.
- `/api/setup/status` returns the current terminal log and result.
- `/predict` returns the latest forecast.
- `/history` and `/metrics` return saved model results.

Each browser only sees its own connected Square merchant. Tokens and account
data are stored server-side and are never sent to the frontend.
