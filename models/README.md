# Models

This folder holds generated training results:

- `model.pt` has the PyTorch weights, feature list, and scalers.
- `training_history.json` has loss and validation scores by epoch.
- `metrics.json` compares the model with simple baselines.
- `predictions.json` has the latest forecast and range.

These files are created locally and ignored by Git because every connected
restaurant gets a different model. Run `.venv/bin/python -m training.train` to
rebuild them.
