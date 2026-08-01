"""Train on the training split, watch validation, stop before it overtrains.

    python -m training.train
    python -m training.train --epochs 500 --patience 40

Writes everything needed to repeat a prediction into models/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import (
    INTERVAL_HIGH_QUANTILE,
    INTERVAL_LOW_QUANTILE,
    MODEL_VERSION,
    ArtifactPaths,
    TrainConfig,
)
from .dataset import Splits, TargetScaler, load_splits, make_loaders, set_seed
from .evaluate import predict_array, score
from .model import ModelConfig, build_model, count_parameters

log = logging.getLogger("train")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """One pass over a loader. Passing an optimizer makes it a training pass."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total, seen = 0.0, 0
    # no_grad for validation, we are only measuring there
    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for features, target in loader:
            prediction = model(features)
            # compare the prediction with the real answer
            loss = loss_fn(prediction, target)

            if is_training:
                # clear the old gradients before the next update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total += loss.item() * len(features)
            seen += len(features)

    return total / max(seen, 1)


def dollar_metrics(model: nn.Module, splits: Splits, target_scaler: TargetScaler, split: str) -> dict:
    """MAE, RMSE and MAPE in dollars, so the log means something to a human."""
    predicted = predict_array(model, splits.X[split], target_scaler)
    return score(splits.y[split], predicted).to_dict()


class EarlyStopping:
    """Keeps the best weights and says when to give up."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_state: dict | None = None
        self.waited = 0

    def update(self, epoch: int, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss - 1e-6:
            self.best_loss = val_loss
            self.best_epoch = epoch
            # save the best model only, not whatever the last epoch left behind
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            self.waited = 0
        else:
            self.waited += 1

    @property
    def should_stop(self) -> bool:
        return self.waited >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is None:
            raise RuntimeError("training never improved once — check the feature file")
        model.load_state_dict(self.best_state)


def validation_residuals(
    model: nn.Module, splits: Splits, target_scaler: TargetScaler
) -> dict[str, float]:
    """How wrong the model usually is, measured on validation.

    We reuse this spread as the prediction interval later. It is honest: it comes
    from days the model never trained on, not from an assumption about the shape
    of the errors.
    """
    predicted = predict_array(model, splits.X["val"], target_scaler).ravel()
    actual = splits.y["val"].ravel()
    residuals = actual - predicted
    return {
        "low": float(np.quantile(residuals, INTERVAL_LOW_QUANTILE)),
        "high": float(np.quantile(residuals, INTERVAL_HIGH_QUANTILE)),
        "std": float(residuals.std()),
        "n": int(len(residuals)),
        "low_quantile": INTERVAL_LOW_QUANTILE,
        "high_quantile": INTERVAL_HIGH_QUANTILE,
    }


def build_bundle(
    model: nn.Module,
    splits: Splits,
    target_scaler: TargetScaler,
    model_config: ModelConfig,
    train_config: TrainConfig,
    stopper: EarlyStopping,
    epochs_run: int,
    seconds: float,
) -> dict:
    """Everything needed to make the same prediction again tomorrow."""
    return {
        "state_dict": model.state_dict(),
        "model_config": model_config.to_dict(),
        "model_version": MODEL_VERSION,
        "feature_names": splits.feature_names,
        "target": splits.manifest["target"],
        # the feature scaler from the pipeline, copied so predict needs one file
        "feature_scaler": splits.manifest["scaler"],
        "target_scaler": target_scaler.to_dict(),
        "residuals": validation_residuals(model, splits, target_scaler),
        "training": {
            **train_config.to_dict(),
            "epochs_run": epochs_run,
            "best_epoch": stopper.best_epoch,
            "best_val_loss": stopper.best_loss,
            "seconds": round(seconds, 1),
        },
        "data": {
            "row_counts": splits.counts(),
            "date_spans": splits.manifest.get("date_spans", {}),
            "features_built_at": splits.manifest.get("created_at"),
            "database": splits.manifest.get("database"),
        },
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def train(
    dataset_file: Path | None = None,
    manifest_file: Path | None = None,
    output_dir: Path | None = None,
    config: TrainConfig | None = None,
    on_epoch: Callable[[dict], None] | None = None,
) -> dict:
    config = config or TrainConfig()
    paths = (ArtifactPaths(Path(output_dir)) if output_dir else ArtifactPaths()).ensure()

    set_seed(config.seed)
    splits = load_splits(**_file_kwargs(dataset_file, manifest_file))

    # scale the target on train only, same rule the pipeline used for features
    target_scaler = TargetScaler.fit(splits.y["train"])
    loaders = make_loaders(splits, target_scaler, config.batch_size, config.seed)

    model_config = ModelConfig(
        input_size=splits.n_features,
        hidden_1=config.hidden_1,
        hidden_2=config.hidden_2,
        dropout=config.dropout,
    )
    model = build_model(model_config)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    counts = splits.counts()
    log.info(
        "train %d | val %d | test %d rows, %d features, %d parameters",
        counts["train"], counts["val"], counts["test"], splits.n_features, count_parameters(model),
    )

    stopper = EarlyStopping(config.patience)
    history: list[dict] = []
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(model, loaders["train"], loss_fn, optimizer)
        val_loss = run_epoch(model, loaders["val"], loss_fn)
        stopper.update(epoch, val_loss, model)

        metrics = dollar_metrics(model, splits, target_scaler, "val")
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val": metrics}
        )
        if on_epoch:
            on_epoch(history[-1])

        if epoch == 1 or epoch % config.log_every == 0:
            _log_epoch(epoch, train_loss, val_loss, metrics, stopper)

        if stopper.should_stop:
            log.info("early stop at epoch %d, no improvement for %d epochs", epoch, config.patience)
            break

    # put the best weights back before anything is measured or saved
    stopper.restore(model)
    seconds = time.perf_counter() - started
    log.info(
        "restored best model from epoch %d (val loss %.4f), %.1fs total",
        stopper.best_epoch, stopper.best_loss, seconds,
    )

    final = dollar_metrics(model, splits, target_scaler, "val")
    log.info(
        "best model on validation: MAE $%.2f  RMSE $%.2f  MAPE %s",
        final["mae_dollars"], final["rmse_dollars"],
        "n/a" if final["mape_percent"] is None else f"{final['mape_percent']:.1f}%",
    )

    bundle = build_bundle(
        model, splits, target_scaler, model_config, config, stopper, len(history), seconds
    )
    torch.save(bundle, paths.checkpoint)
    paths.history.write_text(json.dumps(history, indent=2))
    log.info("saved %s", paths.checkpoint)

    return {
        "checkpoint": str(paths.checkpoint),
        "best_epoch": stopper.best_epoch,
        "best_val_loss": stopper.best_loss,
        "epochs_run": len(history),
        "seconds": round(seconds, 1),
        "row_counts": counts,
        "val_metrics": final,
    }


def _log_epoch(
    epoch: int, train_loss: float, val_loss: float, metrics: dict, stopper: EarlyStopping
) -> None:
    percent = "n/a" if metrics["mape_percent"] is None else f"{metrics['mape_percent']:.1f}%"
    log.info(
        "epoch %3d  train %.4f  val %.4f  |  val MAE $%.0f  RMSE $%.0f  MAPE %s  (best @ %d)",
        epoch, train_loss, val_loss, metrics["mae_dollars"], metrics["rmse_dollars"],
        percent, stopper.best_epoch,
    )


def _file_kwargs(dataset_file: Path | None, manifest_file: Path | None) -> dict:
    kwargs = {}
    if dataset_file:
        kwargs["dataset_file"] = dataset_file
    if manifest_file:
        kwargs["manifest_file"] = manifest_file
    return kwargs


def main(argv: list[str] | None = None) -> int:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train the sales forecast model.")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )

    config = TrainConfig(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout,
        seed=args.seed,
    )
    result = train(args.dataset, args.manifest, args.output_dir, config)

    print(f"\nbest epoch {result['best_epoch']} of {result['epochs_run']} run "
          f"in {result['seconds']}s")
    print(f"checkpoint: {result['checkpoint']}")
    print("\nnext: python -m training.evaluate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
