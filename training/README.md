# Training

This is phase 4. A small PyTorch network learns from the finished feature table.

```text
68 features -> 64 neurons -> ReLU -> dropout -> 32 neurons -> 1 result
```

```bash
.venv/bin/python -m training.train
.venv/bin/python -m training.evaluate
.venv/bin/python -m training.predict
```

Training checks validation dates after every epoch and stops when the model is
no longer improving. It restores the best weights, then uses the untouched test
dates for the final score.

Evaluation reports MAE, RMSE, and MAPE in normal dollar values. It also compares
the model with last week's sales and the seven-day average. Prediction adds an
80% range, confidence, expected orders, and recent lag values.

Outputs are saved in `models/` as the model, training history, metrics, and
latest prediction.
