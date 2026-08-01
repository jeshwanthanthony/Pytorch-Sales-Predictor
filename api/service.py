"""Loads the trained model once and answers questions about it.

The important idea: **loading happens at startup, not per request.** Reading a
checkpoint off disk takes far longer than the forward pass, so doing it inside a
request handler would make every call slow for no reason.

All the machine learning lives in training/. This file only arranges the answers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from training.config import ArtifactPaths, CENTS_PER_DOLLAR
from training.dataset import Splits, TargetScaler, load_splits
from training.evaluate import load_checkpoint, predict_array, rebuild_model, unscale_feature
from training.predict import (
    build_predictions,
    context_values,
    estimate_orders,
    monte_carlo_uncertainty,
)

log = logging.getLogger("api")

# how many features to show next to a prediction
TOP_FEATURES = 6
# splits the model never trained on, so actual vs predicted is honest there
HONEST_SPLITS = ("val", "test")


class ModelNotLoaded(RuntimeError):
    """Raised when the API is asked for a prediction before a model exists."""


@dataclass
class ModelService:
    """Everything the API needs, held in memory."""

    bundle: dict
    model: torch.nn.Module
    target_scaler: TargetScaler
    splits: Splits
    metrics: dict | None

    @classmethod
    def for_workspace(cls, workspace) -> ModelService:
        """Load one restaurant's own model. Nothing is shared between accounts."""
        return cls.load(
            workspace.models_dir, workspace.dataset_file, workspace.manifest_file
        )

    @classmethod
    def load(
        cls,
        model_dir: Path | None = None,
        dataset_file: Path | None = None,
        manifest_file: Path | None = None,
    ) -> ModelService:
        paths = ArtifactPaths(Path(model_dir)) if model_dir else ArtifactPaths()
        bundle = load_checkpoint(paths.checkpoint)

        kwargs = {}
        if dataset_file:
            kwargs["dataset_file"] = dataset_file
        if manifest_file:
            kwargs["manifest_file"] = manifest_file
        splits = load_splits(**kwargs)

        if splits.feature_names != bundle["feature_names"]:
            raise ModelNotLoaded(
                "the feature file and the checkpoint disagree on columns. Re-train the model."
            )

        model, target_scaler = rebuild_model(bundle)
        metrics = json.loads(paths.metrics.read_text()) if paths.metrics.exists() else None

        log.info(
            "loaded model %s trained %s, %d features",
            bundle.get("model_version"), bundle.get("saved_at"), len(bundle["feature_names"]),
        )
        return cls(bundle, model, target_scaler, splits, metrics)

    # -- prediction ---------------------------------------------------------

    def predict_next(self) -> dict:
        """Tomorrow's merchant forecast, summed across locations that trade."""
        features = torch.tensor(self.splits.X["future"], dtype=torch.float32)
        dates = self.splits.dates["future"]
        if not len(dates):
            raise ModelNotLoaded("no future rows in the feature file. Re-run build_features.")

        with torch.no_grad():
            predicted_cents = self.target_scaler.inverse(self.model(features)).ravel()

        scaler = self.bundle["feature_scaler"]
        rows = build_predictions(
            dates=dates,
            predicted_cents=predicted_cents,
            residuals=self.bundle["residuals"],
            uncertainty=monte_carlo_uncertainty(self.model, features, self.target_scaler),
            context=context_values(self.splits, scaler),
            orders=estimate_orders(predicted_cents, self.splits, scaler),
        )

        tomorrow = min(str(value) for value in dates)
        recent = context_values(self.splits, scaler).get("sales_roll_mean_28")
        indexes = [
            index for index, value in enumerate(dates)
            if str(value) == tomorrow and (recent is None or recent[index] > 0)
        ]
        if not indexes:
            indexes = [index for index, value in enumerate(dates) if str(value) == tomorrow]

        selected = [rows[index] for index in indexes]
        predicted = sum(row.predicted_sales for row in selected)
        low = sum(row.interval_low for row in selected)
        high = sum(row.interval_high for row in selected)
        width = max(high - low, 0)
        uncertainty = float(np.sqrt(sum(row.model_uncertainty**2 for row in selected)))
        orders = [
            row.estimated_orders
            for row in selected
            if row.estimated_orders is not None and row.estimated_orders >= 0
        ]
        context = {
            name: round(sum(values[index] for index in indexes) / CENTS_PER_DOLLAR, 2)
            for name, values in context_values(self.splits, scaler).items()
        }

        return {
            "business_date": tomorrow,
            "predicted_sales": round(predicted, 2),
            "interval_low": round(low, 2),
            "interval_high": round(high, 2),
            "confidence": round(float(np.clip(1 - width / predicted, 0, 1)), 3)
            if predicted > 0 else 0.0,
            "model_uncertainty": round(uncertainty, 2),
            "estimated_orders": sum(orders) if orders else None,
            "context": context,
            "important_features": self.explain(features[indexes], indexes),
            "interval_label": self.interval_label(),
            "model_version": self.bundle.get("model_version", "unknown"),
            "model_trained_at": self.bundle.get("saved_at"),
        }

    def interval_label(self) -> str:
        residuals = self.bundle["residuals"]
        span = (residuals["high_quantile"] - residuals["low_quantile"]) * 100
        return f"{span:.0f}% range"

    # -- explanation --------------------------------------------------------

    def explain(self, row: torch.Tensor, indexes: list[int] | None = None) -> list[dict]:
        """Which features moved this prediction the most.

        Gradient times input: ask the network how much the answer would change if
        each feature nudged, then weight that by how big the feature actually is.
        It is a rough guide, not a causal claim, but it tells you whether the
        model leaned on last week's sales or on the fact that it is a Saturday.
        """
        row = row.clone().requires_grad_(True)
        self.model.eval()
        output = self.model(row).sum()
        output.backward()

        saliency = (row.grad * row).detach().numpy().sum(axis=0)
        magnitude = np.abs(saliency)
        total = magnitude.sum()
        if total <= 0:
            return []

        scaler = self.bundle["feature_scaler"]
        indexes = indexes or [0]
        order = np.argsort(magnitude)[::-1][:TOP_FEATURES]

        results = []
        for index in order:
            name = self.splits.feature_names[index]
            unscaled = unscale_feature(
                self.splits.X["future"], self.splits.feature_names, scaler, name
            )
            value = float(np.mean(unscaled[indexes]))
            # sales columns are stored in cents, show dollars instead
            if name.startswith("sales_") or name.startswith("avg_ticket"):
                value = value / CENTS_PER_DOLLAR
            results.append(
                {
                    "name": name,
                    "value": round(value, 2),
                    "contribution": round(float(magnitude[index] / total), 4),
                    "direction": "up" if saliency[index] >= 0 else "down",
                }
            )
        return results

    # -- history ------------------------------------------------------------

    def history(self, days: int = 30) -> list[dict]:
        """Actual vs predicted on days the model never trained on."""
        grouped: dict[tuple[str, str], dict] = {}
        for split in HONEST_SPLITS:
            predicted = predict_array(self.model, self.splits.X[split], self.target_scaler).ravel()
            actual = self.splits.y[split].ravel()
            for index, date in enumerate(self.splits.dates[split]):
                key = (split, str(date))
                point = grouped.setdefault(
                    key,
                    {"business_date": str(date), "actual": 0.0, "predicted": 0.0, "split": split},
                )
                point["actual"] += float(actual[index]) / CENTS_PER_DOLLAR
                point["predicted"] += float(predicted[index]) / CENTS_PER_DOLLAR

        points = list(grouped.values())
        for point in points:
            point["actual"] = round(point["actual"], 2)
            point["predicted"] = round(point["predicted"], 2)
        points.sort(key=lambda point: point["business_date"])
        return points[-days:]

    # -- reporting ----------------------------------------------------------

    def test_metrics(self) -> dict:
        if not self.metrics:
            raise ModelNotLoaded("no metrics.json. Run `python -m training.evaluate`.")

        block = self.metrics["splits"]["test"]
        verdict = self.metrics["verdict"]
        baselines = [
            {
                "name": name.replace("baseline_", "").replace("_", " "),
                "mae": entry["baseline_mae_dollars"],
                "improvement": entry["improvement_dollars"],
                "model_wins": entry["model_wins"],
            }
            for name, entry in verdict["versus"].items()
        ]
        return {
            "split": "test",
            "mae": block["model"]["mae_dollars"],
            "rmse": block["model"]["rmse_dollars"],
            "mape": block["model"]["mape_percent"],
            "mean_actual": block["model"]["mean_actual_dollars"],
            "n_days": block["model"]["n_days"],
            "baselines": baselines,
            "beats_all_baselines": verdict["beats_all_baselines"],
        }

    def health(self) -> dict:
        counts = self.bundle.get("data", {}).get("row_counts", {})
        return {
            "status": "ok",
            "model_loaded": True,
            "model_version": self.bundle.get("model_version"),
            "model_trained_at": self.bundle.get("saved_at"),
            "features": len(self.bundle["feature_names"]),
            "trained_on_days": counts.get("train"),
        }
