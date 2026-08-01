"""The model. Deliberately small.

Two hidden layers over ~68 features. With only a few hundred days of history
anything bigger memorises the training set, and a small model is far easier to
debug and explain. No LSTM yet — the lag features already hand the network the
history it would otherwise have to learn.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    """Saved next to the weights so the same model can be rebuilt later."""

    input_size: int
    hidden_1: int = 64
    hidden_2: int = 32
    dropout: float = 0.2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> ModelConfig:
        known = {field: values[field] for field in cls.__annotations__ if field in values}
        return cls(**known)


class SalesForecastNet(nn.Module):
    """features -> 64 -> relu -> dropout -> 32 -> relu -> 1 predicted sales value"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.layers = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_1),
            nn.ReLU(),
            # dropout only after the first layer, the second one is already small
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_1, config.hidden_2),
            nn.ReLU(),
            nn.Linear(config.hidden_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def build_model(config: ModelConfig) -> SalesForecastNet:
    return SalesForecastNet(config)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
