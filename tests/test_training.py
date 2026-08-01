"""Tests for the training package.

These build a small feature file directly instead of running the whole pipeline,
so they stay fast. The pipeline itself is covered by test_pipelines.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from training.dataset import (
    DataContractError,
    TargetScaler,
    load_splits,
    make_loaders,
    set_seed,
)
from training.config import TrainConfig
from training.evaluate import evaluate, mape, score, unscale_feature
from training.model import ModelConfig, build_model, count_parameters
from training.predict import confidence_from_interval, predict
from training.train import train

FEATURES = [
    "day_of_week", "is_weekend", "sales_lag_1", "sales_lag_7",
    "sales_roll_mean_7", "avg_ticket_lag_7", "temp_max_f",
]


def make_feature_file(
    tmp_path: Path, n_train: int = 120, n_val: int = 20, n_test: int = 20
) -> tuple[Path, Path]:
    """A learnable toy series: weekend days sell more, plus a little noise."""
    rng = np.random.default_rng(0)
    total = n_train + n_val + n_test + 1

    day_of_week = np.array([(i % 7) + 1 for i in range(total)], dtype=float)
    is_weekend = (day_of_week >= 6).astype(float)
    temp = rng.normal(70, 8, total)
    sales = 80_000 + 40_000 * is_weekend + rng.normal(0, 4_000, total)

    lag_1 = np.roll(sales, 1)
    lag_7 = np.roll(sales, 7)
    lag_1[0] = sales[0]
    lag_7[:7] = sales[:7]
    roll_7 = np.convolve(np.roll(sales, 1), np.ones(7) / 7, mode="same")
    avg_ticket = np.full(total, 2_500.0)

    X = np.column_stack(
        [day_of_week, is_weekend, lag_1, lag_7, roll_7, avg_ticket, temp]
    ).astype(np.float32)
    y = sales.astype(np.float32).reshape(-1, 1)
    dates = np.array([f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(total)])
    # keep them strictly increasing so the split check is meaningful
    dates = np.array(
        [(np.datetime64("2026-01-01") + np.timedelta64(i, "D")).astype(str) for i in range(total)]
    )

    slices = {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, n_train + n_val + n_test),
        "future": slice(total - 1, total),
    }

    # scale features the way the pipeline would, using train rows only
    scaler = {}
    for index, name in enumerate(FEATURES):
        if name.startswith(("is_", "has_")):
            continue
        column = X[slices["train"], index]
        scaler[name] = {"mean": float(column.mean()), "std": float(column.std()) or 1.0}

    X_scaled = X.copy()
    for index, name in enumerate(FEATURES):
        if name in scaler:
            X_scaled[:, index] = (X[:, index] - scaler[name]["mean"]) / scaler[name]["std"]

    arrays = {"feature_names": np.array(FEATURES)}
    for split, window in slices.items():
        arrays[f"X_{split}"] = X_scaled[window]
        arrays[f"dates_{split}"] = dates[window]
        if split != "future":
            arrays[f"y_{split}"] = y[window]

    dataset_file = tmp_path / "dataset.npz"
    manifest_file = tmp_path / "manifest.json"
    np.savez_compressed(dataset_file, **arrays)
    manifest_file.write_text(
        json.dumps(
            {
                "target": "target_sales_cents",
                "feature_names": FEATURES,
                "feature_count": len(FEATURES),
                "scaler": scaler,
                "created_at": "2026-07-31T00:00:00+00:00",
                "date_spans": {"train": [str(dates[0]), str(dates[n_train - 1])]},
                "database": "test",
            }
        )
    )
    return dataset_file, manifest_file


@pytest.fixture(scope="module")
def feature_files(tmp_path_factory) -> tuple[Path, Path]:
    return make_feature_file(tmp_path_factory.mktemp("features"))


@pytest.fixture(scope="module")
def trained(tmp_path_factory, feature_files) -> tuple[Path, dict]:
    dataset_file, manifest_file = feature_files
    out = tmp_path_factory.mktemp("models")
    result = train(
        dataset_file, manifest_file, out, TrainConfig(epochs=120, patience=25, seed=42)
    )
    return out / "model.pt", result


class TestDataContract:
    def test_loads_and_checks(self, feature_files):
        splits = load_splits(*feature_files)
        assert splits.n_features == len(FEATURES)
        assert splits.counts()["train"] == 120

    def test_target_in_features_is_rejected(self, tmp_path):
        dataset_file, manifest_file = make_feature_file(tmp_path)
        manifest = json.loads(manifest_file.read_text())
        manifest["feature_names"] = FEATURES[:-1] + ["target_sales_cents"]
        manifest_file.write_text(json.dumps(manifest))

        with pytest.raises(DataContractError, match="that is the answer"):
            load_splits(dataset_file, manifest_file)

    def test_same_day_outcome_is_rejected(self, tmp_path):
        dataset_file, manifest_file = make_feature_file(tmp_path)
        manifest = json.loads(manifest_file.read_text())
        manifest["feature_names"] = FEATURES[:-1] + ["order_count"]
        manifest_file.write_text(json.dumps(manifest))

        with pytest.raises(DataContractError, match="same-day outcome"):
            load_splits(dataset_file, manifest_file)

    def test_overlapping_splits_are_rejected(self, tmp_path):
        dataset_file, manifest_file = make_feature_file(tmp_path)
        with np.load(dataset_file, allow_pickle=True) as data:
            arrays = {key: data[key] for key in data.files}
        # push val back so it starts before train ends
        arrays["dates_val"] = arrays["dates_train"][:len(arrays["dates_val"])]
        np.savez_compressed(dataset_file, **arrays)

        with pytest.raises(DataContractError, match="overlap in time"):
            load_splits(dataset_file, manifest_file)

    def test_nan_features_are_rejected(self, tmp_path):
        dataset_file, manifest_file = make_feature_file(tmp_path)
        with np.load(dataset_file, allow_pickle=True) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["X_train"][0, 0] = np.nan
        np.savez_compressed(dataset_file, **arrays)

        with pytest.raises(DataContractError, match="NaN"):
            load_splits(dataset_file, manifest_file)

    def test_missing_scaler_is_rejected(self, tmp_path):
        dataset_file, manifest_file = make_feature_file(tmp_path)
        manifest = json.loads(manifest_file.read_text())
        manifest["scaler"] = {}
        manifest_file.write_text(json.dumps(manifest))

        with pytest.raises(DataContractError, match="no scaler"):
            load_splits(dataset_file, manifest_file)


class TestTargetScaler:
    def test_round_trip(self):
        y = np.array([[100.0], [200.0], [300.0]])
        scaler = TargetScaler.fit(y)
        assert np.allclose(scaler.inverse(scaler.transform(y)), y)

    def test_fit_uses_only_what_it_is_given(self, feature_files):
        splits = load_splits(*feature_files)
        scaler = TargetScaler.fit(splits.y["train"])
        assert scaler.mean == pytest.approx(float(splits.y["train"].mean()))
        # and it is not the mean over everything
        everything = np.concatenate([splits.y[s] for s in ("train", "val", "test")])
        assert scaler.mean != pytest.approx(float(everything.mean()), rel=1e-9)

    def test_constant_target_does_not_divide_by_zero(self):
        scaler = TargetScaler.fit(np.zeros((10, 1)))
        assert scaler.std == 1.0


class TestModel:
    def test_shape(self):
        model = build_model(ModelConfig(input_size=5))
        out = model(torch.zeros(3, 5))
        assert out.shape == (3, 1)

    def test_architecture_is_the_simple_one(self):
        model = build_model(ModelConfig(input_size=5, hidden_1=64, hidden_2=32))
        kinds = [type(layer).__name__ for layer in model.layers]
        assert kinds == ["Linear", "ReLU", "Dropout", "Linear", "ReLU", "Linear"]

    def test_config_round_trip(self):
        config = ModelConfig(input_size=9, hidden_1=16, hidden_2=8, dropout=0.5)
        assert ModelConfig.from_dict(config.to_dict()) == config

    def test_parameter_count_is_small(self):
        assert count_parameters(build_model(ModelConfig(input_size=68))) < 10_000

    def test_dropout_only_active_in_train_mode(self):
        model = build_model(ModelConfig(input_size=5, dropout=0.9))
        x = torch.ones(64, 5)
        model.eval()
        with torch.no_grad():
            assert torch.allclose(model(x), model(x))


class TestLoaders:
    def test_batches_and_shuffle(self, feature_files):
        splits = load_splits(*feature_files)
        scaler = TargetScaler.fit(splits.y["train"])
        loaders = make_loaders(splits, scaler, batch_size=16, seed=1)

        assert loaders["train"].batch_size == 16
        assert loaders["train"].sampler.__class__.__name__ == "RandomSampler"
        # val and test keep their order
        assert loaders["val"].sampler.__class__.__name__ == "SequentialSampler"

        features, target = next(iter(loaders["train"]))
        assert features.shape[1] == splits.n_features
        assert target.shape[1] == 1


class TestTrain:
    def test_checkpoint_has_everything_needed_to_reproduce(self, trained):
        path, _ = trained
        bundle = torch.load(path, weights_only=False)
        for key in (
            "state_dict", "model_config", "feature_names", "target",
            "feature_scaler", "target_scaler", "training", "data", "saved_at",
        ):
            assert key in bundle, key
        assert bundle["feature_names"] == FEATURES
        assert bundle["training"]["seed"] == 42

    def test_best_model_is_restored_not_the_last(self, trained):
        path, result = trained
        history = json.loads((path.parent / "training_history.json").read_text())
        best = min(row["val_loss"] for row in history)
        assert result["best_val_loss"] == pytest.approx(best)
        # early stopping means the best epoch is not the last one
        assert result["best_epoch"] <= result["epochs_run"]

    def test_early_stopping_fires(self, trained):
        _, result = trained
        assert result["epochs_run"] <= 120

    def test_same_seed_gives_same_weights(self, tmp_path, feature_files):
        dataset_file, manifest_file = feature_files
        config = TrainConfig(epochs=15, patience=15, seed=7)
        first = train(dataset_file, manifest_file, tmp_path / "a", config)
        second = train(dataset_file, manifest_file, tmp_path / "b", config)
        assert first["best_val_loss"] == pytest.approx(second["best_val_loss"])

        a = torch.load(tmp_path / "a" / "model.pt", weights_only=False)["state_dict"]
        b = torch.load(tmp_path / "b" / "model.pt", weights_only=False)["state_dict"]
        for key in a:
            assert torch.allclose(a[key], b[key])


class TestMetrics:
    def test_score_is_in_dollars(self):
        actual = np.array([[100_00.0], [200_00.0]])
        predicted = np.array([[110_00.0], [190_00.0]])
        result = score(actual, predicted)
        assert result.mae_dollars == 10.0
        assert result.n_days == 2

    def test_mape_skips_zero_days(self):
        actual = np.array([0.0, 100.0])
        predicted = np.array([50.0, 110.0])
        # only the 100 -> 110 day counts, so 10%
        assert mape(actual, predicted) == pytest.approx(10.0)

    def test_mape_is_none_when_everything_is_zero(self):
        assert mape(np.zeros(3), np.ones(3)) is None

    def test_unscale_feature_recovers_original_units(self, feature_files):
        splits = load_splits(*feature_files)
        values = unscale_feature(
            splits.X["test"], splits.feature_names, splits.manifest["scaler"], "sales_lag_7"
        )
        # back in cents, so a sensible sales number
        assert values.mean() > 50_000


class TestEvaluate:
    def test_reports_model_and_baseline(self, trained, feature_files):
        path, _ = trained
        payload = evaluate(path, *feature_files, output_dir=path.parent)
        assert set(payload["splits"]) == {"train", "val", "test"}
        for block in payload["splits"].values():
            assert block["model"]["mae_dollars"] > 0
            assert block["baseline_last_week"]["mae_dollars"] > 0
            assert block["baseline_rolling_7"]["mae_dollars"] > 0
        assert "beats_all_baselines" in payload["verdict"]

    def test_beats_baseline_on_a_learnable_series(self, trained, feature_files):
        path, _ = trained
        payload = evaluate(path, *feature_files, output_dir=path.parent)
        # the toy series is weekday-driven, which the model should learn
        assert payload["verdict"]["beats_all_baselines"]

    def test_metrics_file_written(self, trained, feature_files):
        path, _ = trained
        evaluate(path, *feature_files, output_dir=path.parent)
        assert (path.parent / "metrics.json").exists()

    def test_feature_list_mismatch_is_caught(self, trained, tmp_path):
        path, _ = trained
        dataset_file, manifest_file = make_feature_file(tmp_path)
        manifest = json.loads(manifest_file.read_text())
        manifest["feature_names"] = FEATURES[:-1] + ["something_else"]
        manifest_file.write_text(json.dumps(manifest))

        with pytest.raises(RuntimeError, match="feature list"):
            evaluate(path, dataset_file, manifest_file, output_dir=tmp_path)


class TestPredict:
    def test_predicts_the_future_row(self, trained, feature_files):
        path, _ = trained
        payload = predict(path, *feature_files, output_dir=path.parent)
        assert len(payload["predictions"]) == 1
        row = payload["predictions"][0]
        assert row["predicted_sales"] > 0
        assert row["business_date"]

    def test_is_deterministic(self, trained, feature_files):
        path, _ = trained
        first = predict(path, *feature_files, output_dir=path.parent)
        second = predict(path, *feature_files, output_dir=path.parent)
        assert (
            first["predictions"][0]["predicted_sales"]
            == second["predictions"][0]["predicted_sales"]
        )

    def test_carries_context_numbers(self, trained, feature_files):
        path, _ = trained
        row = predict(path, *feature_files, output_dir=path.parent)["predictions"][0]
        assert row["context"]["sales_lag_7"] > 0

    def test_interval_brackets_the_prediction(self, trained, feature_files):
        path, _ = trained
        row = predict(path, *feature_files, output_dir=path.parent)["predictions"][0]
        assert row["interval_low"] <= row["predicted_sales"] <= row["interval_high"]
        assert 0.0 <= row["confidence"] <= 1.0

    def test_uncertainty_is_reported(self, trained, feature_files):
        path, _ = trained
        row = predict(path, *feature_files, output_dir=path.parent)["predictions"][0]
        # dropout sampling always moves the answer a little
        assert row["model_uncertainty"] >= 0

    def test_order_estimate_is_derived_from_ticket(self, trained, feature_files):
        path, _ = trained
        row = predict(path, *feature_files, output_dir=path.parent)["predictions"][0]
        # avg ticket in the fixture is $25, so orders ~= sales / 25
        assert row["estimated_orders"] == pytest.approx(row["predicted_sales"] / 25, rel=0.02)


class TestConfidence:
    def test_narrow_band_is_confident(self):
        # a $100 band around a $10,000 day is a tight forecast
        assert confidence_from_interval(np.array([10_000.0]), -50, 50)[0] > 0.98

    def test_wide_band_is_not(self):
        # a band as wide as the prediction means we know almost nothing
        assert confidence_from_interval(np.array([1_000.0]), -500, 500)[0] == 0.0

    def test_never_goes_negative(self):
        assert confidence_from_interval(np.array([100.0]), -5_000, 5_000)[0] == 0.0


class TestSeeding:
    def test_set_seed_is_repeatable(self):
        set_seed(123)
        a = torch.randn(5)
        set_seed(123)
        assert torch.allclose(a, torch.randn(5))
