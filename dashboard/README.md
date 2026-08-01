# Dashboard

This is the browser screen served by the API. It connects a Square account,
starts the pipeline, and prints each stage like a terminal:

```text
Square data -> cleaned SQLite -> Pandas table -> PyTorch epochs -> forecast
```

```bash
.venv/bin/python -m api.serve
```

Open `http://localhost:8080`. The page uses plain HTML, CSS, and JavaScript so
there is no separate frontend build. It polls the API for live logs and shows
the latest prediction when training finishes.
