# models

**Storage only.** No code lives here. `training/` writes these files and `api/`
reads them.

## What ends up here

| File | Written by | What it is |
| --- | --- | --- |
| `model.pt` | `training/train.py` | the trained network plus everything needed to reuse it |
| `training_history.json` | `training/train.py` | train and validation loss for every epoch |
| `metrics.json` | `training/evaluate.py` | model vs baselines on train, val and test |
| `predictions.json` | `training/predict.py` | the latest forecast |

## What is inside model.pt

It is not just the weights. A prediction made next month has to match one made
today, so the checkpoint carries everything that shaped it:

```python
{
  "state_dict":      # the learned weights
  "model_config":    # layer sizes and dropout, so the network can be rebuilt
  "model_version":   # so the api can say what it is serving
  "feature_names":   # the exact column order the model expects
  "feature_scaler":  # copied from the pipeline, so we scale new rows identically
  "target_scaler":   # to turn the model output back into cents
  "residuals":       # validation errors, used for the prediction interval
  "training":        # hyperparameters, best epoch, seed, how long it took
  "data":            # row counts and split dates
  "saved_at":        # when
}
```

If any of that were missing you could load the weights and still get a different
answer. That is the whole reason this file is a bundle and not a `.pth` of
tensors.

## Why these files are gitignored

They are build outputs, not source. Anyone can rebuild them:

```bash
python -m pipelines.build_features
python -m training.train
```

A real deployment would push `model.pt` to object storage or a model registry
and record the version. Committing a binary that changes every run makes for a
miserable git history.

## Who reads this folder

`api/main.py` loads `model.pt` once at startup and keeps it in memory.
