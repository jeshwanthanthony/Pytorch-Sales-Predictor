# training

**Phase 4.** Takes the finished feature table and teaches the PyTorch model.

This folder does not touch Square, the database, or any feature logic. It reads
two files the pipeline made and learns from them.

## The model

Small on purpose:

```
68 features -> 64 neurons -> relu -> dropout -> 32 neurons -> relu -> 1 number
```

About 6,500 parameters. No LSTM. With a year or two of daily rows there is not
enough data for one, the lag features already hand the model the history, and a
small model is much easier to debug and explain.

## How the training works

1. read `dataset.npz` and `manifest.json`
2. check the data is safe (see below)
3. scale the target using the **training days only**
4. train in batches with a DataLoader
5. after every epoch, check the validation days with `torch.no_grad()`
6. if validation stops improving for 40 epochs, stop early
7. put the **best** weights back, not the last ones
8. only then, look at the test days
9. save everything into `models/model.pt`

## Rules that stop us fooling ourselves

- the target is never one of the inputs
- same-day facts (`order_count`, `tip_cents`, ...) are blocked
- train, validation and test must not overlap in time
- the scaler comes from the pipeline and is **never** re-fit on val or test
- the test split is only touched after training is completely finished
- seeds are set, so two runs give the same answer

If any of these break, `dataset.py` raises `DataContractError` and nothing trains.

## The baselines

`evaluate.py` compares the model against two guesses that cost nothing:

| Baseline | What it guesses |
| --- | --- |
| last week | whatever we sold on the same weekday 7 days ago |
| rolling 7 | the average of the last 7 days |

They sound too simple to matter. In a restaurant the same weekday last week is
already a strong guess, and plenty of neural networks lose to it. The verdict
only says "worth using" if the model beats **both**.

Scores are MAE, RMSE and MAPE **in dollars**, because "0.28 MSE in scaled units"
tells a restaurant owner nothing.

## The prediction interval

A single number is a bad forecast. `predict.py` adds a range:

- **interval** — the 10th and 90th percentile of the errors the model made on
  validation days. Honest, because it comes from days it never trained on.
- **confidence** — how narrow that range is next to the prediction itself.
- **model uncertainty** — run the model 100 times with dropout left *on* and see
  how much the answer moves. That tells you if the model is confused about this
  particular day. It is seeded, so the same question gives the same answer.

## What comes in

```
data/features/dataset.npz
data/features/manifest.json
```

## What it creates

```
models/model.pt               weights + config + both scalers + residuals + feature list
models/training_history.json  loss and val MAE/RMSE/MAPE per epoch
models/metrics.json           model vs both baselines
models/predictions.json       the forecast with its interval
```

## Example

```bash
python -m training.train      # learn
python -m training.evaluate   # how good is it, really?
python -m training.predict    # what about tomorrow?
```

```
forecast

  2026-07-02     $  1,262.79
                 80% range  $1,074 to $1,468   confidence 69%
                 roughly 54 orders
                 recent: lag 1 $946, lag 7 $1,076, roll mean 7 $1,137
```

## What runs next

`api/` loads `models/model.pt` once at startup and serves it.

## Files here

| File | Job |
| --- | --- |
| `config.py` | every path, constant and default in one place |
| `dataset.py` | loads the feature file, checks it, makes tensors and DataLoaders |
| `model.py` | the network itself |
| `train.py` | the training loop, early stopping, saving |
| `evaluate.py` | test scores and the baseline comparison |
| `predict.py` | loads the saved model and predicts the future rows |
