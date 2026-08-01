"""Response shapes for the API.

Declaring these explicitly means FastAPI documents and validates every field, and
the dashboard knows exactly what it is getting.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FeatureContribution(BaseModel):
    """One feature and how much it moved this particular prediction."""

    name: str
    value: float = Field(description="the feature in its original units, not scaled")
    contribution: float = Field(description="relative influence on this prediction, 0-1")
    direction: str = Field(description="up or down")


class PredictionResponse(BaseModel):
    business_date: str
    predicted_sales: float = Field(description="dollars")
    interval_low: float
    interval_high: float
    interval_label: str = Field(description="e.g. '80% range'")
    confidence: float = Field(ge=0, le=1)
    model_uncertainty: float = Field(description="dollars of spread across dropout samples")
    estimated_orders: int | None = None
    context: dict[str, float] = Field(default_factory=dict)
    important_features: list[FeatureContribution] = Field(default_factory=list)
    model_version: str
    model_trained_at: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    model_trained_at: str | None = None
    features: int | None = None
    trained_on_days: int | None = None
    detail: str | None = None


class HistoryPoint(BaseModel):
    business_date: str
    actual: float
    predicted: float
    split: str = Field(description="val or test, the days the model did not train on")


class HistoryResponse(BaseModel):
    days: int
    points: list[HistoryPoint]


class BaselineComparison(BaseModel):
    name: str
    mae: float
    improvement: float
    model_wins: bool


class MetricsResponse(BaseModel):
    split: str
    mae: float
    rmse: float
    mape: float | None
    mean_actual: float
    n_days: int
    baselines: list[BaselineComparison]
    beats_all_baselines: bool
