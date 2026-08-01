"""Paths, constants, and default settings for training.

Everything tunable lives here so the other files hold logic only. If you want to
change how the model trains, you should not have to open train.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# trained artifacts live here, nothing else does
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_FILE = MODEL_DIR / "model.pt"
HISTORY_FILE = MODEL_DIR / "training_history.json"
METRICS_FILE = MODEL_DIR / "metrics.json"
PREDICTIONS_FILE = MODEL_DIR / "predictions.json"

# a version string so the api can say which model it is serving
MODEL_VERSION = "1.0.0"

SPLIT_NAMES = ("train", "val", "test")

# the two baselines we have to beat to be worth anything
BASELINE_LAG_7 = "sales_lag_7"
BASELINE_ROLLING_7 = "sales_roll_mean_7"

# features shown next to a prediction so a human can sanity check it
CONTEXT_FEATURES = ("sales_lag_1", "sales_lag_7", "sales_roll_mean_7", "sales_roll_mean_28")

# how wide the prediction interval is, from validation residuals
INTERVAL_LOW_QUANTILE = 0.10
INTERVAL_HIGH_QUANTILE = 0.90

# how many forward passes with dropout on, to estimate model uncertainty
MC_DROPOUT_SAMPLES = 100
# fixed so the same question always gets the same answer
MC_DROPOUT_SEED = 12345

CENTS_PER_DOLLAR = 100


@dataclass(frozen=True)
class TrainConfig:
    """Everything the training loop needs to know."""

    epochs: int = 400
    patience: int = 40
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    dropout: float = 0.3
    hidden_1: int = 64
    hidden_2: int = 32
    seed: int = 42
    # print a metrics line every N epochs
    log_every: int = 25

    def to_dict(self) -> dict:
        return {
            "epochs": self.epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "hidden_1": self.hidden_1,
            "hidden_2": self.hidden_2,
            "seed": self.seed,
        }


@dataclass
class ArtifactPaths:
    """Where one training run writes its files."""

    directory: Path = MODEL_DIR

    @property
    def checkpoint(self) -> Path:
        return self.directory / CHECKPOINT_FILE.name

    @property
    def history(self) -> Path:
        return self.directory / HISTORY_FILE.name

    @property
    def metrics(self) -> Path:
        return self.directory / METRICS_FILE.name

    @property
    def predictions(self) -> Path:
        return self.directory / PREDICTIONS_FILE.name

    def ensure(self) -> ArtifactPaths:
        self.directory.mkdir(parents=True, exist_ok=True)
        return self
