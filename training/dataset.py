"""Load the feature arrays built by pipelines/build_features.py and turn them
into PyTorch tensors and DataLoaders.

Nothing here computes a feature or fits a scaler. If a number is wrong, it was
wrong before it got here.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pipelines.build_features import DATASET_FILE, MANIFEST_FILE, SAME_DAY_OUTCOMES, TARGET

from .config import SPLIT_NAMES


class DataContractError(RuntimeError):
    """Raised when the feature file breaks an assumption training depends on."""


@dataclass
class Splits:
    """Everything training needs, already split by the pipeline."""

    X: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    # plain strings, not numpy arrays — "2026-07-02" compares correctly as text
    dates: dict[str, list[str]]
    feature_names: list[str]
    manifest: dict

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def counts(self) -> dict[str, int]:
        return {name: len(array) for name, array in self.X.items()}


def set_seed(seed: int = 42) -> None:
    # same seed everywhere so two runs give the same numbers
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_splits(
    dataset_file: Path = DATASET_FILE,
    manifest_file: Path = MANIFEST_FILE,
) -> Splits:
    """Read dataset.npz + manifest.json and check they are safe to train on."""
    dataset_file, manifest_file = Path(dataset_file), Path(manifest_file)
    if not dataset_file.exists():
        raise FileNotFoundError(
            f"{dataset_file} not found. Run `python -m pipelines.build_features` first."
        )

    manifest = json.loads(manifest_file.read_text())
    feature_names = list(manifest["feature_names"])

    # load the clean features made by the pipeline
    with np.load(dataset_file, allow_pickle=True) as data:
        available = set(data.files)
        X = {name: data[f"X_{name}"] for name in SPLIT_NAMES if f"X_{name}" in available}
        y = {name: data[f"y_{name}"] for name in SPLIT_NAMES if f"y_{name}" in available}
        dates = {
            name: [str(value) for value in data[f"dates_{name}"]]
            for name in (*SPLIT_NAMES, "future")
            if f"dates_{name}" in available
        }
        if "X_future" in available:
            X["future"] = data["X_future"]

    splits = Splits(X=X, y=y, dates=dates, feature_names=feature_names, manifest=manifest)
    check_contract(splits)
    return splits


def check_contract(splits: Splits) -> None:
    """Fail loudly if the data would train a model that looks better than it is."""
    names = splits.feature_names

    # the answer must never be one of the inputs
    if TARGET in names:
        raise DataContractError(f"{TARGET} is in the feature list — that is the answer, not an input")

    leaked = sorted(set(names) & SAME_DAY_OUTCOMES)
    if leaked:
        raise DataContractError(
            f"same-day outcome columns found in features: {', '.join(leaked)}. "
            "These are only known after the day ends."
        )

    for name in SPLIT_NAMES:
        if name not in splits.X:
            raise DataContractError(f"missing the {name} split — rebuild features")
        if splits.X[name].shape[1] != len(names):
            raise DataContractError(
                f"{name}: {splits.X[name].shape[1]} columns but {len(names)} feature names"
            )
        if len(splits.X[name]) != len(splits.y[name]):
            raise DataContractError(f"{name}: X and y have different lengths")
        if not np.isfinite(splits.X[name]).all():
            raise DataContractError(f"{name}: features contain NaN or inf")

    # keep dates in order because this is time series data
    train_end = max(splits.dates["train"])
    val_start, val_end = min(splits.dates["val"]), max(splits.dates["val"])
    test_start = min(splits.dates["test"])
    if not (train_end < val_start and val_end < test_start):
        raise DataContractError(
            f"splits overlap in time (train ends {train_end}, val {val_start}..{val_end}, "
            f"test starts {test_start}). A model must never see the future during training."
        )

    # the scaler was fit on train only back in the pipeline, we just reuse it
    if not splits.manifest.get("scaler"):
        raise DataContractError("manifest has no scaler — cannot reproduce this at prediction time")


@dataclass
class TargetScaler:
    """Standardises y so the loss is a sensible size.

    Sales in cents are ~100,000, and squaring that gives a loss around 1e10 that
    swamps the optimiser. Fit on train only, same rule as the feature scaler.
    """

    mean: float
    std: float

    @classmethod
    def fit(cls, y_train: np.ndarray) -> TargetScaler:
        std = float(y_train.std())
        return cls(mean=float(y_train.mean()), std=std if std > 1e-9 else 1.0)

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.std

    def inverse(self, y: np.ndarray | torch.Tensor) -> np.ndarray:
        values = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
        return values * self.std + self.mean

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> TargetScaler:
        return cls(mean=float(values["mean"]), std=float(values["std"]))


def make_loaders(
    splits: Splits,
    target_scaler: TargetScaler,
    batch_size: int = 16,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """One DataLoader per split. Only train gets shuffled."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    loaders = {}
    for name in SPLIT_NAMES:
        # x is the information the model gets, y is the real sales answer
        features = torch.tensor(splits.X[name], dtype=torch.float32)
        target = torch.tensor(target_scaler.transform(splits.y[name]), dtype=torch.float32)

        loaders[name] = DataLoader(
            TensorDataset(features, target),
            batch_size=batch_size,
            # shuffling train is fine, the time order is already baked into the lags
            shuffle=(name == "train"),
            generator=generator if name == "train" else None,
            drop_last=False,
        )
    return loaders


def future_tensor(splits: Splits) -> tuple[torch.Tensor, list[str]]:
    """The rows build_features prepared for days that have not happened yet."""
    if "future" not in splits.X or len(splits.X["future"]) == 0:
        raise DataContractError(
            "no future rows in the dataset — re-run build_features with --horizon-days"
        )
    return (
        torch.tensor(splits.X["future"], dtype=torch.float32),
        splits.dates.get("future", []),
    )
