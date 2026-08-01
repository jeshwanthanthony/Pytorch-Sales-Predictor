"""Score the model and compare it against baselines that cost nothing to run.

Two baselines, both of them harder to beat than they look:

  last week   whatever we sold on the same weekday 7 days ago
  rolling 7   the average of the last 7 days

If the network cannot beat both of these, it is not earning its keep, and this
file is where you find that out.

    python -m training.evaluate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .config import (
    BASELINE_LAG_7,
    BASELINE_ROLLING_7,
    CENTS_PER_DOLLAR,
    SPLIT_NAMES,
    ArtifactPaths,
)
from .dataset import Splits, TargetScaler, load_splits, set_seed
from .model import ModelConfig, build_model

log = logging.getLogger("evaluate")


@dataclass
class Scores:
    """One set of results, always in dollars."""

    mae_dollars: float
    rmse_dollars: float
    mape_percent: float | None
    mean_actual_dollars: float
    n_days: int

    def to_dict(self) -> dict:
        return asdict(self)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    """Percent error, skipping closed days.

    A zero-sales day would divide by zero, and a percentage of nothing means
    nothing. Returns None if every day was zero.
    """
    open_days = actual != 0
    if not open_days.any():
        return None
    errors = np.abs((actual[open_days] - predicted[open_days]) / actual[open_days])
    return float(np.mean(errors) * 100)


def score(actual_cents: np.ndarray, predicted_cents: np.ndarray) -> Scores:
    # convert to dollars first, nobody can read cents or scaled units
    actual = actual_cents.ravel() / CENTS_PER_DOLLAR
    predicted = predicted_cents.ravel() / CENTS_PER_DOLLAR
    percent = mape(actual, predicted)
    return Scores(
        mae_dollars=round(mae(actual, predicted), 2),
        rmse_dollars=round(rmse(actual, predicted), 2),
        mape_percent=None if percent is None else round(percent, 2),
        mean_actual_dollars=round(float(actual.mean()), 2),
        n_days=int(len(actual)),
    )


def unscale_feature(
    features: np.ndarray, feature_names: list[str], scaler: dict, column: str
) -> np.ndarray:
    """Read one feature back in its original units.

    The pipeline standardised X, so lag_7 in there is not dollars any more.
    Undo it with the same numbers that scaled it.
    """
    if column not in feature_names:
        raise KeyError(f"{column} is not in the feature list")
    values = features[:, feature_names.index(column)].astype(float)
    if column in scaler:
        values = values * scaler[column]["std"] + scaler[column]["mean"]
    return values


def baseline_predictions(splits: Splits, scaler: dict, split: str) -> dict[str, np.ndarray]:
    """The free guesses we have to beat."""
    return {
        "baseline_last_week": unscale_feature(
            splits.X[split], splits.feature_names, scaler, BASELINE_LAG_7
        ).reshape(-1, 1),
        "baseline_rolling_7": unscale_feature(
            splits.X[split], splits.feature_names, scaler, BASELINE_ROLLING_7
        ).reshape(-1, 1),
    }


def predict_array(
    model: torch.nn.Module, features: np.ndarray, target_scaler: TargetScaler
) -> np.ndarray:
    """Run the model over a whole split and return cents."""
    model.eval()
    # no_grad because we are only measuring, not learning
    with torch.no_grad():
        scaled = model(torch.tensor(features, dtype=torch.float32))
    return target_scaler.inverse(scaled)


def load_checkpoint(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m training.train` first.")
    return torch.load(path, weights_only=False)


def rebuild_model(bundle: dict) -> tuple[torch.nn.Module, TargetScaler]:
    """Put a saved model back together exactly as it was."""
    model = build_model(ModelConfig.from_dict(bundle["model_config"]))
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, TargetScaler.from_dict(bundle["target_scaler"])


def evaluate(
    checkpoint_path: Path | None = None,
    dataset_file: Path | None = None,
    manifest_file: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    paths = ArtifactPaths(Path(output_dir)) if output_dir else ArtifactPaths()
    bundle = load_checkpoint(checkpoint_path or paths.checkpoint)

    set_seed(bundle["training"]["seed"])
    splits = load_splits(**_file_kwargs(dataset_file, manifest_file))

    if splits.feature_names != bundle["feature_names"]:
        raise RuntimeError(
            "the feature list changed since training. Re-run build_features and re-train."
        )

    model, target_scaler = rebuild_model(bundle)
    scaler = bundle["feature_scaler"]

    results = {}
    for split in SPLIT_NAMES:
        actual = splits.y[split]
        predictions = {"model": predict_array(model, splits.X[split], target_scaler)}
        predictions.update(baseline_predictions(splits, scaler, split))

        results[split] = {
            name: score(actual, values).to_dict() for name, values in predictions.items()
        }
        results[split]["dates"] = [min(splits.dates[split]), max(splits.dates[split])]

    payload = {
        "checkpoint": str(checkpoint_path or paths.checkpoint),
        "model_trained_at": bundle.get("saved_at"),
        "splits": results,
        "verdict": build_verdict(results["test"]),
    }

    paths.ensure()
    paths.metrics.write_text(json.dumps(payload, indent=2))
    return payload


def build_verdict(test_block: dict) -> dict:
    """Did the model actually win, on the split it never saw?"""
    model_mae = test_block["model"]["mae_dollars"]
    beaten = {}
    for name in ("baseline_last_week", "baseline_rolling_7"):
        baseline_mae = test_block[name]["mae_dollars"]
        improvement = baseline_mae - model_mae
        beaten[name] = {
            "baseline_mae_dollars": baseline_mae,
            "improvement_dollars": round(improvement, 2),
            "improvement_percent": round(improvement / baseline_mae * 100, 1) if baseline_mae else 0.0,
            "model_wins": improvement > 0,
        }

    return {
        "model_mae_dollars": model_mae,
        "versus": beaten,
        # only a real win if it beats every baseline, not just the weakest one
        "beats_all_baselines": all(entry["model_wins"] for entry in beaten.values()),
    }


def _file_kwargs(dataset_file: Path | None, manifest_file: Path | None) -> dict:
    kwargs = {}
    if dataset_file:
        kwargs["dataset_file"] = dataset_file
    if manifest_file:
        kwargs["manifest_file"] = manifest_file
    return kwargs


ROW_LABELS = {
    "model": "pytorch model",
    "baseline_last_week": "baseline: last week",
    "baseline_rolling_7": "baseline: 7 day avg",
}


def render(payload: dict) -> str:
    lines = []
    for split, block in payload["splits"].items():
        span = f"{block['dates'][0]} .. {block['dates'][1]}"
        lines.append(f"\n{split.upper()}  ({block['model']['n_days']} days, {span})")
        lines.append(f"  {'':<22}{'MAE':>12}{'RMSE':>12}{'MAPE':>9}")
        for key, label in ROW_LABELS.items():
            metrics = block[key]
            percent = "n/a" if metrics["mape_percent"] is None else f"{metrics['mape_percent']:.1f}%"
            lines.append(
                f"  {label:<22}{'$' + format(metrics['mae_dollars'], ',.2f'):>12}"
                f"{'$' + format(metrics['rmse_dollars'], ',.2f'):>12}{percent:>9}"
            )
        lines.append(f"  actual average: ${block['model']['mean_actual_dollars']:,.2f}/day")

    verdict = payload["verdict"]
    lines.append("")
    for name, entry in verdict["versus"].items():
        label = ROW_LABELS[name]
        if entry["model_wins"]:
            lines.append(
                f"  beats {label:<22} by ${entry['improvement_dollars']:,.2f} MAE "
                f"({entry['improvement_percent']}%)"
            )
        else:
            lines.append(
                f"  LOSES to {label:<19} by ${abs(entry['improvement_dollars']):,.2f} MAE"
            )
    lines.append("")
    lines.append(
        "verdict: the model is worth using"
        if verdict["beats_all_baselines"]
        else "verdict: a baseline is as good or better, do not ship this yet"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the trained model.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    payload = evaluate(args.checkpoint, args.dataset, args.manifest, args.output_dir)
    print(render(payload))
    print("\nnext: python -m training.predict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
