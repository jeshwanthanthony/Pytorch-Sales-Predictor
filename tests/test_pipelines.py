"""Tests for the cleaning, aggregation, and feature pipelines.

The fixture builds ~70 days of synthetic trading through the real loader, so
these exercise the actual SQL and the actual pandas — not mocks of them.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from collector.calendar_api import CalendarCollector
from collector.config import SiteConfig
from database.db import connect
from database.load import load_all
from pipelines import build_daily_summary as summary_mod
from pipelines.build_daily_summary import build as build_summary
from pipelines.build_features import (
    SAME_DAY_OUTCOMES,
    LeakageError,
    assert_no_leakage,
    auto_split_sizes,
    build as build_features,
    feature_columns,
)
from pipelines.validate_data import ERROR, validate

SITE = SiteConfig(38.8816, -77.0910, "America/New_York", "US", "VA")
START = date(2026, 4, 1)
DAYS = 70
CLOSED_DAY = START + timedelta(days=30)   # a deliberate closure


def write_raw(raw_dir: Path, entity: str, rows: list[dict], run: str = "20260701T000000Z") -> None:
    directory = raw_dir / entity
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{entity}-{run}.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )


def synth_orders() -> tuple[list[dict], list[dict]]:
    """Deterministic trading: busier weekends, one closed day, known dayparts."""
    orders, items = [], []
    for offset in range(DAYS):
        day = START + timedelta(days=offset)
        if day == CLOSED_DAY:
            continue

        count = 8 if day.isoweekday() >= 6 else 4
        for index in range(count):
            # Alternate lunch (12:00 local) and dinner (19:00 local).
            local_hour = 12 if index % 2 == 0 else 19
            stamp = datetime(
                day.year, day.month, day.day, local_hour, 0, tzinfo=timezone.utc
            ) + timedelta(hours=4)  # local -> UTC
            order_id = f"O{offset:03d}-{index:02d}"
            subtotal = 1_000
            tax, tip = 60, 150
            orders.append(
                {
                    "order_id": order_id,
                    "location_id": "LOC1",
                    "created_at": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "updated_at": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "state": "COMPLETED",
                    "currency": "USD",
                    "revenue_cents": subtotal + tax + tip,
                    "discount_cents": 0,
                    "tax_cents": tax,
                    "tip_cents": tip,
                    "service_charge_cents": 0,
                    "net_sales_cents": subtotal,
                    "source": "Square POS",
                    # A repeat customer every day, plus one first-timer per day.
                    "customer_id": "CREG" if index == 0 else (f"CNEW{offset:03d}" if index == 1 else None),
                    "payment_types": ["CARD"],
                    "tender_count": 1,
                    "line_item_count": 1,
                    "item_quantity": 1.0,
                    "has_returns": False,
                    "version": 1,
                }
            )
            items.append(
                {
                    "order_id": order_id,
                    "line_item_uid": "LI1",
                    "line_number": 0,
                    "location_id": "LOC1",
                    "created_at": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "catalog_object_id": "VAR_A",
                    "item_name": "Butter Chicken",
                    "quantity": 1.0,
                    "base_price_cents": subtotal,
                    "gross_sales_cents": subtotal,
                    "total_cents": subtotal,
                    "modifiers": [],
                }
            )
    return orders, items


def synth_weather(first: date, last: date) -> list[dict]:
    rows = []
    day = first
    while day <= last:
        rows.append(
            {
                "date": day.isoformat(),
                "source": "archive",
                "temp_max_f": 70.0 + (day.toordinal() % 15),
                "temp_min_f": 50.0,
                "temp_mean_f": 60.0,
                "feels_like_max_f": 72.0,
                "precipitation_in": 0.5 if day.toordinal() % 5 == 0 else 0.0,
                "snowfall_in": 0.0,
                "precipitation_hours": 2.0 if day.toordinal() % 5 == 0 else 0.0,
                "wind_max_mph": 8.0,
                "humidity_mean": 60.0,
                "is_rainy": day.toordinal() % 5 == 0,
                "is_snowy": False,
                "is_stormy": False,
            }
        )
        day += timedelta(days=1)
    return rows


@pytest.fixture(scope="module")
def manifest(pipeline_db, tmp_path_factory) -> dict:
    """Features built once for the module, into a throwaway output directory."""
    out = tmp_path_factory.mktemp("features_out")
    return build_features(
        pipeline_db, val_days=7, test_days=7, horizon_days=10, write_db=True, output_dir=out
    )


@pytest.fixture(scope="module")
def pipeline_db(tmp_path_factory) -> Path:
    """A warehouse with the daily summary already built."""
    tmp = tmp_path_factory.mktemp("pipelines")
    raw, db = tmp / "raw", tmp / "pipeline.db"
    last = START + timedelta(days=DAYS - 1)
    horizon = last + timedelta(days=20)

    orders, items = synth_orders()
    write_raw(raw, "locations", [{
        "location_id": "LOC1", "name": "Arlington", "status": "ACTIVE",
        "currency": "USD", "country": "US", "state": "VA", "timezone": "America/New_York",
    }])
    write_raw(raw, "calendar", list(CalendarCollector(SITE).fetch(START, horizon)))
    write_raw(raw, "weather", synth_weather(START, horizon))
    write_raw(raw, "orders", orders)
    write_raw(raw, "order_items", items)

    load_all(db, raw_dir=raw)
    build_summary(db)
    return db


class TestValidate:
    def test_clean_data_passes(self, pipeline_db):
        report = validate(pipeline_db)
        assert report.ok, [c.detail for c in report.errors]

    def test_broken_money_identity_blocks(self, pipeline_db, tmp_path):
        import shutil

        copy = tmp_path / "broken.db"
        shutil.copy(pipeline_db, copy)
        conn = connect(copy)
        conn.execute("UPDATE orders SET net_sales_cents = 999999 WHERE order_id LIKE 'O001%'")
        conn.commit()
        conn.close()

        report = validate(copy)
        assert not report.ok
        assert any(c.name == "totals reconcile" for c in report.errors)

    def test_missing_business_date_blocks(self, pipeline_db, tmp_path):
        import shutil

        copy = tmp_path / "nodate.db"
        shutil.copy(pipeline_db, copy)
        conn = connect(copy)
        conn.execute("UPDATE orders SET business_date = NULL WHERE order_id LIKE 'O002%'")
        conn.commit()
        conn.close()

        report = validate(copy)
        assert any(c.name == "business dates" and c.severity == ERROR for c in report.errors)

    def test_all_cancelled_orders_blocks(self, pipeline_db, tmp_path):
        """Orders can exist while no day counts as a sale. That must not pass."""
        import shutil

        copy = tmp_path / "cancelled.db"
        shutil.copy(pipeline_db, copy)
        conn = connect(copy)
        conn.execute("UPDATE orders SET state = 'CANCELED'")
        conn.commit()
        conn.close()

        report = validate(copy)
        assert not report.ok
        failed = [c for c in report.errors if c.name == "history length"]
        assert failed and "none of them count as sales" in failed[0].detail

    def test_strict_promotes_warnings(self, pipeline_db):
        lenient = validate(pipeline_db, strict=False)
        strict = validate(pipeline_db, strict=True)
        assert len(strict.errors) >= len(lenient.errors)

    def test_report_renders(self, pipeline_db):
        assert "orders present" in validate(pipeline_db).render()


class TestDailySummary:
    def test_one_row_per_calendar_day(self, pipeline_db):
        conn = connect(pipeline_db, read_only=True)
        count = conn.execute("SELECT COUNT(*) FROM daily_summary").fetchone()[0]
        conn.close()
        assert count == DAYS  # including the closed day

    def test_closed_day_is_a_zero_not_a_gap(self, pipeline_db):
        conn = connect(pipeline_db, read_only=True)
        row = conn.execute(
            "SELECT * FROM daily_summary WHERE business_date = ?", (CLOSED_DAY.isoformat(),)
        ).fetchone()
        conn.close()
        assert row is not None
        assert row["is_closed"] == 1
        assert row["order_count"] == 0
        assert row["net_sales_cents"] == 0

    def test_dayparts_split_by_local_hour(self, pipeline_db):
        conn = connect(pipeline_db, read_only=True)
        row = conn.execute(
            "SELECT lunch_orders, dinner_orders, late_orders, order_count "
            "FROM daily_summary WHERE is_closed = 0 AND order_count = 8 LIMIT 1"
        ).fetchone()
        conn.close()
        assert row["lunch_orders"] == 4
        assert row["dinner_orders"] == 4
        assert row["late_orders"] == 0

    def test_new_vs_returning_customers(self, pipeline_db):
        conn = connect(pipeline_db, read_only=True)
        rows = conn.execute(
            "SELECT business_date, new_customer_count, returning_customer_count, customer_count "
            "FROM daily_summary WHERE is_closed = 0 ORDER BY business_date"
        ).fetchall()
        conn.close()

        first, later = rows[0], rows[5]
        # Day one: both customers are new.
        assert first["new_customer_count"] == 2
        # Later: the daily regular is returning, the day's first-timer is new.
        assert later["new_customer_count"] == 1
        assert later["returning_customer_count"] == 1
        assert later["customer_count"] == 2

    def test_peak_hour_is_local(self, pipeline_db):
        conn = connect(pipeline_db, read_only=True)
        hours = {
            r["peak_hour"]
            for r in conn.execute("SELECT peak_hour FROM daily_summary WHERE is_closed = 0")
        }
        conn.close()
        assert hours <= {12, 19}

    def test_ratios_handle_zero_denominators(self, pipeline_db):
        conn = connect(pipeline_db, read_only=True)
        row = conn.execute(
            "SELECT sales_per_labor_hour_cents, labor_cost_ratio_bps, refund_rate_bps "
            "FROM daily_summary WHERE business_date = ?", (CLOSED_DAY.isoformat(),)
        ).fetchone()
        conn.close()
        # No labor, no sales — zeros, never NULL or a division error.
        assert row["sales_per_labor_hour_cents"] == 0
        assert row["labor_cost_ratio_bps"] == 0
        assert row["refund_rate_bps"] == 0

    def test_rebuild_is_idempotent(self, pipeline_db):
        conn = connect(pipeline_db, read_only=True)
        before = conn.execute("SELECT COUNT(*), SUM(net_sales_cents) FROM daily_summary").fetchone()
        conn.close()

        build_summary(pipeline_db)

        conn = connect(pipeline_db, read_only=True)
        after = conn.execute("SELECT COUNT(*), SUM(net_sales_cents) FROM daily_summary").fetchone()
        conn.close()
        assert tuple(before) == tuple(after)

    def test_summary_matches_daily_sales_view(self, pipeline_db):
        """The new table must agree with the view it supersedes."""
        conn = connect(pipeline_db, read_only=True)
        mismatches = conn.execute(
            "SELECT COUNT(*) FROM daily_summary s JOIN daily_sales v "
            "ON v.business_date = s.business_date AND v.location_id = s.location_id "
            "WHERE s.net_sales_cents <> v.net_sales_cents OR s.order_count <> v.order_count"
        ).fetchone()[0]
        conn.close()
        assert mismatches == 0


class TestFeatures:
    def test_leakage_guard_rejects_outcome_columns(self):
        with pytest.raises(LeakageError, match="net_sales_cents"):
            assert_no_leakage(["day_of_week", "net_sales_cents"])

    def test_no_outcome_column_survives_selection(self, pipeline_db):
        import pandas as pd

        from pipelines.build_features import (
            add_cyclical, add_history, add_holiday_distance, add_interactions,
            add_targets, load_source,
        )

        frame = load_source(pipeline_db, 10)
        for step in (add_targets, add_cyclical, add_holiday_distance, add_history, add_interactions):
            frame = step(frame)
        columns = feature_columns(frame)
        assert not set(columns) & SAME_DAY_OUTCOMES
        assert isinstance(frame, pd.DataFrame)

    def test_splits_are_chronological(self, manifest):
        spans = manifest["date_spans"]
        assert spans["train"][1] < spans["val"][0]
        assert spans["val"][1] < spans["test"][0]

    def test_split_sizes(self, manifest):
        assert manifest["row_counts"]["val"] == 7
        assert manifest["row_counts"]["test"] == 7
        assert manifest["row_counts"]["train"] > 0

    def test_scaler_comes_from_train_only(self, manifest, pipeline_db):
        import pandas as pd

        conn = connect(pipeline_db, read_only=True)
        rows = conn.execute("SELECT split, payload FROM features").fetchall()
        conn.close()

        frame = pd.DataFrame([{"split": r["split"], **json.loads(r["payload"])} for r in rows])
        column = "sales_lag_7"
        train_mean = frame[frame["split"] == "train"][column].mean()
        stored = manifest["scaler"][column]["mean"]
        assert stored == pytest.approx(train_mean, rel=1e-3)

        # And it must not equal the mean over everything.
        assert stored != pytest.approx(frame[column].mean(), rel=1e-6)

    def test_binary_columns_are_not_scaled(self, manifest):
        assert "is_weekend" in manifest["binary_features"]
        assert "is_weekend" not in manifest["scaler"]

    def test_lag_1_equals_previous_actual(self, manifest, pipeline_db):
        import pandas as pd

        conn = connect(pipeline_db, read_only=True)
        rows = conn.execute(
            "SELECT business_date, target_sales_cents, payload FROM features ORDER BY business_date"
        ).fetchall()
        conn.close()

        frame = pd.DataFrame(
            [
                {"date": r["business_date"], "target": r["target_sales_cents"], **json.loads(r["payload"])}
                for r in rows
            ]
        )
        expected = frame["target"].shift(1)
        assert (frame["sales_lag_1"].iloc[1:] == expected.iloc[1:]).all()

    def test_cyclical_encoding_wraps(self, manifest, pipeline_db):
        conn = connect(pipeline_db, read_only=True)
        payloads = [
            json.loads(r["payload"])
            for r in conn.execute("SELECT payload FROM features LIMIT 40")
        ]
        conn.close()
        for row in payloads:
            assert -1.001 <= row["dow_sin"] <= 1.001
            assert abs(row["dow_sin"] ** 2 + row["dow_cos"] ** 2 - 1) < 0.01

    def test_future_rows_have_no_target(self, manifest, pipeline_db):
        conn = connect(pipeline_db, read_only=True)
        rows = conn.execute(
            "SELECT target_sales_cents FROM features WHERE split = 'future'"
        ).fetchall()
        conn.close()
        assert rows
        assert all(r["target_sales_cents"] is None for r in rows)

    def test_arrays_are_finite_and_aligned(self, manifest):
        from pipelines.build_features import load_dataset

        data = load_dataset(Path(manifest["dataset_file"]))
        for split in ("train", "val", "test"):
            assert np.isfinite(data[f"X_{split}"]).all()
            assert len(data[f"X_{split}"]) == len(data[f"y_{split}"])
            assert len(data[f"dates_{split}"]) == len(data[f"X_{split}"])
        assert data["X_train"].shape[1] == manifest["feature_count"]

    def test_manifest_records_what_inference_needs(self, manifest):
        assert manifest["target"] == "target_sales_cents"
        assert len(manifest["feature_names"]) == manifest["feature_count"]
        assert manifest["scaler"]  # non-empty
        assert "split_config" in manifest


class TestSplitSizing:
    def test_explicit_sizes_are_respected(self):
        assert auto_split_sizes(500, 28, 28) == (28, 28)

    def test_short_history_gets_small_holdouts(self):
        val, test = auto_split_sizes(60, 0, 0)
        assert val == test == 9
        assert 60 - val - test > 30

    def test_floor_and_cap(self):
        assert auto_split_sizes(10, 0, 0) == (7, 7)      # floor
        assert auto_split_sizes(1000, 0, 0) == (28, 28)  # cap


class TestDaypartConstants:
    def test_boundaries_are_coherent(self):
        assert summary_mod.LUNCH_START < summary_mod.LUNCH_END <= summary_mod.DINNER_START
        assert summary_mod.DINNER_START < summary_mod.DINNER_END
