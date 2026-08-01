"""Predict the future rows the pipeline prepared, with an honest error bar.

A single number is a bad forecast. "$1,260, probably between $1,090 and $1,430"
is something a manager can actually staff against. Two different uncertainties
go into that:

  interval    how wrong the model usually was on validation days
  uncertainty how much the prediction moves if you re-run it with dropout on

The first is the one that matters. The second tells you whether the model is
confused about this particular day.

    python -m training.predict
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .config import (
    CENTS_PER_DOLLAR,
    CONTEXT_FEATURES,
    MC_DROPOUT_SAMPLES,
    MC_DROPOUT_SEED,
    ArtifactPaths,
)
from .dataset import Splits, TargetScaler, future_tensor, load_splits, set_seed
from .evaluate import load_checkpoint, rebuild_model, unscale_feature

log = logging.getLogger("predict")


@dataclass
class Prediction:
    """One day's forecast, in dollars, ready to hand to an API."""

    business_date: str
    predicted_sales: float
    interval_low: float
    interval_high: float
    confidence: float
    model_uncertainty: float
    estimated_orders: int | None
    context: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def monte_carlo_uncertainty(
    model: torch.nn.Module,
    features: torch.Tensor,
    target_scaler: TargetScaler,
    seed: int = MC_DROPOUT_SEED,
) -> np.ndarray:
    """Run the model many times with dropout left on, and see how much it wobbles.

    Normally dropout is switched off for prediction. Leaving it on gives a
    slightly different network each pass, so the spread of the answers is a
    rough measure of how unsure the model is about this row.

    Seeded, because asking the same question twice must give the same answer.
    """
    torch.manual_seed(seed)
    model.train()  # dropout on
    samples = []
    with torch.no_grad():
        for _ in range(MC_DROPOUT_SAMPLES):
            samples.append(target_scaler.inverse(model(features)).ravel())
    model.eval()
    return np.stack(samples).std(axis=0)


def confidence_from_interval(predicted: np.ndarray, low: float, high: float) -> np.ndarray:
    """Turn the interval width into a 0-1 score.

    A band that is narrow next to the prediction means confident. A band as wide
    as the prediction itself means we barely know anything, so score near zero.
    """
    width = abs(high - low)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(predicted > 0, width / predicted, 1.0)
    return np.clip(1.0 - relative, 0.0, 1.0)


def context_values(splits: Splits, scaler: dict) -> dict[str, np.ndarray]:
    """Recent sales numbers, unscaled, so a human can sanity check the forecast."""
    values = {}
    for column in CONTEXT_FEATURES:
        if column in splits.feature_names:
            values[column] = unscale_feature(
                splits.X["future"], splits.feature_names, scaler, column
            )
    return values


def estimate_orders(predicted_cents: np.ndarray, splits: Splits, scaler: dict) -> np.ndarray | None:
    """Rough order count = predicted sales / recent average ticket.

    This is arithmetic, not a second model. It is useful for staffing but it
    inherits every error the sales prediction has, so treat it as a hint.
    """
    if "avg_ticket_lag_7" not in splits.feature_names:
        return None
    ticket = unscale_feature(splits.X["future"], splits.feature_names, scaler, "avg_ticket_lag_7")
    with np.errstate(divide="ignore", invalid="ignore"):
        # Scaler roundoff can turn a true zero ticket into a tiny positive
        # number. Requiring at least $1 prevents a meaningless huge quotient.
        orders = np.where((ticket >= 100) & (predicted_cents > 0), predicted_cents / ticket, np.nan)
    return orders


def build_predictions(
    dates: list[str],
    predicted_cents: np.ndarray,
    residuals: dict,
    uncertainty: np.ndarray,
    context: dict[str, np.ndarray],
    orders: np.ndarray | None,
) -> list[Prediction]:
    low_offset, high_offset = residuals["low"], residuals["high"]
    interval_low = predicted_cents + low_offset
    interval_high = predicted_cents + high_offset
    confidence = confidence_from_interval(predicted_cents, low_offset, high_offset)

    rows = []
    for index, date in enumerate(dates):
        estimated = None
        if orders is not None and np.isfinite(orders[index]):
            estimated = int(round(float(orders[index])))

        rows.append(
            Prediction(
                business_date=str(date),
                predicted_sales=_dollars(predicted_cents[index]),
                interval_low=_dollars(max(interval_low[index], 0)),
                interval_high=_dollars(interval_high[index]),
                confidence=round(float(confidence[index]), 3),
                model_uncertainty=_dollars(uncertainty[index]),
                estimated_orders=estimated,
                context={
                    name: _dollars(values[index]) for name, values in context.items()
                },
            )
        )
    return rows


def _dollars(cents: float) -> float:
    return round(float(cents) / CENTS_PER_DOLLAR, 2)


def predict(
    checkpoint_path: Path | None = None,
    dataset_file: Path | None = None,
    manifest_file: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    paths = (ArtifactPaths(Path(output_dir)) if output_dir else ArtifactPaths()).ensure()
    bundle = load_checkpoint(checkpoint_path or paths.checkpoint)

    set_seed(bundle["training"]["seed"])
    splits = load_splits(**_file_kwargs(dataset_file, manifest_file))

    # the columns must match exactly or the model is reading the wrong numbers
    if splits.feature_names != bundle["feature_names"]:
        raise RuntimeError("feature list does not match the checkpoint. Re-train before predicting.")

    model, target_scaler = rebuild_model(bundle)
    features, dates = future_tensor(splits)

    # no_grad, we are only asking the model a question
    with torch.no_grad():
        predicted_cents = target_scaler.inverse(model(features)).ravel()

    scaler = bundle["feature_scaler"]
    rows = build_predictions(
        dates=dates,
        predicted_cents=predicted_cents,
        residuals=bundle["residuals"],
        uncertainty=monte_carlo_uncertainty(model, features, target_scaler),
        context=context_values(splits, scaler),
        orders=estimate_orders(predicted_cents, splits, scaler),
    )

    payload = {
        "predicted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": bundle.get("model_version"),
        "model_trained_at": bundle.get("saved_at"),
        "interval": {
            "low_quantile": bundle["residuals"]["low_quantile"],
            "high_quantile": bundle["residuals"]["high_quantile"],
            "from_validation_days": bundle["residuals"]["n"],
        },
        "predictions": [row.to_dict() for row in rows],
    }

    paths.predictions.write_text(json.dumps(payload, indent=2))
    return payload


def _file_kwargs(dataset_file: Path | None, manifest_file: Path | None) -> dict:
    kwargs = {}
    if dataset_file:
        kwargs["dataset_file"] = dataset_file
    if manifest_file:
        kwargs["manifest_file"] = manifest_file
    return kwargs


def render(payload: dict) -> str:
    interval = payload["interval"]
    span = int((interval["high_quantile"] - interval["low_quantile"]) * 100)
    lines = ["", "forecast", ""]
    for row in payload["predictions"]:
        lines.append(f"  {row['business_date']}     ${row['predicted_sales']:>10,.2f}")
        lines.append(
            f"                 {span}% range  ${row['interval_low']:,.0f} to "
            f"${row['interval_high']:,.0f}   confidence {row['confidence']:.0%}"
        )
        if row["estimated_orders"]:
            lines.append(f"                 roughly {row['estimated_orders']} orders")
        context = row["context"]
        if context:
            readable = ", ".join(
                f"{name.replace('sales_', '').replace('_', ' ')} ${value:,.0f}"
                for name, value in context.items()
            )
            lines.append(f"                 recent: {readable}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict upcoming days.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    payload = predict(args.checkpoint, args.dataset, args.manifest, args.output_dir)
    print(render(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
