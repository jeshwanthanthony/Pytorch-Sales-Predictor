"""Tests for the warehouse, using synthetic Square-shaped raw files.

The Square half of the collector can't be exercised without a live token, so
these fixtures stand in for it: the records have the exact shape square_api.py
emits, which is what the loader contracts against.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from collector.calendar_api import CalendarCollector
from collector.config import SiteConfig
from database.db import connect
from database.load import BusinessDay, load_all
from database.queries import feature_frame, prediction_rows

SITE = SiteConfig(38.8816, -77.0910, "America/New_York", "US", "VA")

LOCATION = {
    "location_id": "LOC1",
    "name": "Arlington",
    "status": "ACTIVE",
    "currency": "USD",
    "country": "US",
    "state": "VA",
    "city": "Arlington",
    "postal_code": "22201",
    "timezone": "America/New_York",
}


def write_raw(raw_dir: Path, entity: str, rows: list[dict], run: str = "20260801T000000Z") -> None:
    directory = raw_dir / entity
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{entity}-{run}.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def order(order_id: str, created_at: str, revenue: int, **overrides) -> dict:
    row = {
        "order_id": order_id,
        "location_id": "LOC1",
        "created_at": created_at,
        "updated_at": created_at,
        "closed_at": created_at,
        "state": "COMPLETED",
        "currency": "USD",
        "revenue_cents": revenue,
        "discount_cents": 0,
        "tax_cents": 0,
        "tip_cents": 0,
        "service_charge_cents": 0,
        "net_sales_cents": revenue,
        "source": "Square POS",
        "fulfillment_type": None,
        "fulfillment_state": None,
        "customer_id": None,
        "payment_types": ["CARD"],
        "tender_count": 1,
        "line_item_count": 1,
        "item_quantity": 1.0,
        "has_returns": False,
        "ticket_name": None,
        "version": 1,
    }
    row.update(overrides)
    return row


@pytest.fixture
def warehouse(tmp_path: Path):
    """A loaded database over a small synthetic history.

    Three trading days at one location, plus the calendar spine the feature
    view needs.
    """
    raw = tmp_path / "raw"
    db = tmp_path / "test.db"

    calendar = CalendarCollector(SITE)
    write_raw(raw, "calendar", list(calendar.fetch(date(2026, 7, 25), date(2026, 8, 5))))
    write_raw(raw, "locations", [LOCATION])
    write_raw(
        raw,
        "weather",
        [
            {"date": "2026-07-30", "temp_max_f": 91.0, "precipitation_in": 0.0, "is_rainy": False},
            {"date": "2026-07-31", "temp_max_f": 78.0, "precipitation_in": 0.9, "is_rainy": True},
            {"date": "2026-08-01", "temp_max_f": 84.0, "precipitation_in": 0.0, "is_rainy": False},
        ],
    )
    write_raw(
        raw,
        "orders",
        [
            # 18:15 EDT on the 30th
            order("O1", "2026-07-30T22:15:00Z", 4_000),
            order("O2", "2026-07-30T23:00:00Z", 2_000),
            # 19:00 EDT on the 31st
            order("O3", "2026-07-31T23:00:00Z", 5_000),
            # 01:30 EDT on Aug 1 — still the 31st's business, before the 4am cutoff
            order("O4", "2026-08-01T05:30:00Z", 1_000),
            # 12:00 EDT on Aug 1
            order("O5", "2026-08-01T16:00:00Z", 3_000),
            # canceled: never revenue
            order("O6", "2026-08-01T17:00:00Z", 9_999, state="CANCELED"),
        ],
    )
    write_raw(
        raw,
        "order_items",
        [
            {
                "order_id": "O1",
                "line_item_uid": "LI1",
                "line_number": 0,
                "location_id": "LOC1",
                "created_at": "2026-07-30T22:15:00Z",
                "catalog_object_id": "VAR_BC",
                "item_name": "Butter Chicken",
                "variation_name": "Regular",
                "quantity": 2.0,
                "base_price_cents": 1600,
                "gross_sales_cents": 3200,
                "discount_cents": 0,
                "tax_cents": 0,
                "total_cents": 3200,
                "modifiers": [
                    {"uid": "M1", "catalog_object_id": "MOD_SPICY", "name": "Extra spicy", "price_cents": 0},
                    {"uid": "M2", "catalog_object_id": "MOD_RICE", "name": "Add rice", "price_cents": 300},
                ],
                "modifier_names": ["Extra spicy", "Add rice"],
            },
            {
                "order_id": "O3",
                "line_item_uid": "LI1",
                "line_number": 0,
                "location_id": "LOC1",
                "created_at": "2026-07-31T23:00:00Z",
                "catalog_object_id": "VAR_NAAN",
                "item_name": "Garlic Naan",
                "quantity": 3.0,
                "base_price_cents": 400,
                "total_cents": 1200,
                "modifiers": [],
            },
        ],
    )
    write_raw(
        raw,
        "catalog",
        [
            {
                "catalog_object_id": "ITEM_BC",
                "item_name": "Butter Chicken",
                "category_id": "CAT1",
                "category_name": "Mains",
                "is_archived": False,
                "is_deleted": False,
                "modifier_list_ids": ["ML1"],
                "variations": [
                    {"id": "VAR_BC", "name": "Regular", "sku": "BC-R", "price_cents": 1600},
                    {"id": "VAR_BC_L", "name": "Large", "sku": "BC-L", "price_cents": 2100},
                ],
            }
        ],
    )
    write_raw(
        raw,
        "payments",
        [
            {
                "payment_id": "P1",
                "order_id": "O1",
                "location_id": "LOC1",
                "created_at": "2026-07-30T22:15:00Z",
                "status": "COMPLETED",
                "amount_cents": 4000,
                "source_type": "CARD",
                "card_brand": "VISA",
                "processing_fee_cents": 116,
            },
            {
                "payment_id": "P2",
                "order_id": "O2",
                "location_id": "LOC1",
                "created_at": "2026-07-30T23:00:00Z",
                "status": "COMPLETED",
                "amount_cents": 2000,
                "source_type": "CASH",
            },
        ],
    )
    write_raw(
        raw,
        "shifts",
        [
            {
                "shift_id": "S1",
                "team_member_id": "TM1",
                "location_id": "LOC1",
                "start_at": "2026-07-30T20:00:00Z",
                "end_at": "2026-07-31T02:00:00Z",
                "hours": 6.0,
                "status": "CLOSED",
                "job_title": "Server",
                "hourly_rate_cents": 1500,
                "labor_cost_cents": 9000,
                "declared_tips_cents": 0,
            }
        ],
    )

    load_all(db, raw_dir=raw)
    return db, raw


class TestBusinessDay:
    def test_late_night_belongs_to_previous_day(self):
        calendar = BusinessDay({"LOC1": "America/New_York"}, cutoff_hour=4)
        # 05:30 UTC = 01:30 EDT, before the 4am cutoff
        assert calendar.of("2026-08-01T05:30:00Z", "LOC1") == ("2026-07-31", 1)

    def test_afternoon_is_its_own_day(self):
        calendar = BusinessDay({"LOC1": "America/New_York"}, cutoff_hour=4)
        assert calendar.of("2026-08-01T16:00:00Z", "LOC1") == ("2026-08-01", 12)

    def test_evening_utc_rolls_back_to_local_day(self):
        calendar = BusinessDay({"LOC1": "America/New_York"}, cutoff_hour=4)
        # 22:15 UTC = 18:15 EDT the same day
        assert calendar.of("2026-07-30T22:15:00Z", "LOC1") == ("2026-07-30", 18)

    def test_per_location_timezone(self):
        calendar = BusinessDay({"E": "America/New_York", "W": "America/Los_Angeles"}, cutoff_hour=4)
        # 06:00 UTC = 02:00 EDT (before cutoff, so the 31st) but 23:00 PDT on the 31st
        assert calendar.of("2026-08-01T06:00:00Z", "E")[0] == "2026-07-31"
        assert calendar.of("2026-08-01T06:00:00Z", "W")[0] == "2026-07-31"

    def test_missing_timestamp(self):
        calendar = BusinessDay({}, cutoff_hour=4)
        assert calendar.of(None, "LOC1") == (None, None)


class TestLoad:
    def test_orders_land_on_the_right_business_date(self, warehouse):
        db, _ = warehouse
        conn = connect(db, read_only=True)
        rows = {
            r["order_id"]: r["business_date"]
            for r in conn.execute("SELECT order_id, business_date FROM orders")
        }
        conn.close()
        assert rows["O1"] == "2026-07-30"
        # 01:30am order counts as the night before
        assert rows["O4"] == "2026-07-31"
        assert rows["O5"] == "2026-08-01"

    def test_modifiers_become_rows(self, warehouse):
        db, _ = warehouse
        conn = connect(db, read_only=True)
        names = [r["name"] for r in conn.execute("SELECT name FROM order_item_modifiers ORDER BY name")]
        conn.close()
        assert names == ["Add rice", "Extra spicy"]

    def test_variations_become_rows(self, warehouse):
        db, _ = warehouse
        conn = connect(db, read_only=True)
        rows = conn.execute(
            "SELECT variation_id, price_cents FROM catalog_variations ORDER BY variation_id"
        ).fetchall()
        conn.close()
        assert [r["variation_id"] for r in rows] == ["VAR_BC", "VAR_BC_L"]
        assert [r["price_cents"] for r in rows] == [1600, 2100]

    def test_json_columns_round_trip(self, warehouse):
        db, _ = warehouse
        conn = connect(db, read_only=True)
        raw = conn.execute("SELECT payment_types FROM orders WHERE order_id = 'O1'").fetchone()[0]
        conn.close()
        assert json.loads(raw) == ["CARD"]

    def test_money_stays_integer(self, warehouse):
        db, _ = warehouse
        conn = connect(db, read_only=True)
        value = conn.execute("SELECT revenue_cents FROM orders WHERE order_id = 'O1'").fetchone()[0]
        conn.close()
        assert isinstance(value, int)


class TestIdempotency:
    def test_reloading_the_same_files_changes_nothing(self, warehouse):
        db, raw = warehouse
        conn = connect(db, read_only=True)
        before = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        conn.close()

        load_all(db, raw_dir=raw)          # skipped, already ingested
        load_all(db, raw_dir=raw, force=True)  # re-read every file

        conn = connect(db, read_only=True)
        after = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        conn.close()
        assert before == after == 6

    def test_a_later_file_updates_the_row_in_place(self, warehouse):
        db, raw = warehouse
        # Square re-sends O1 with a refund applied.
        write_raw(
            raw,
            "orders",
            [order("O1", "2026-07-30T22:15:00Z", 1_000, updated_at="2026-08-02T10:00:00Z", has_returns=True)],
            run="20260802T000000Z",
        )
        load_all(db, raw_dir=raw)

        conn = connect(db, read_only=True)
        row = conn.execute("SELECT revenue_cents, has_returns FROM orders WHERE order_id = 'O1'").fetchone()
        count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        conn.close()

        assert row["revenue_cents"] == 1_000  # updated, not duplicated
        assert row["has_returns"] == 1
        assert count == 6


class TestViews:
    def test_daily_sales_excludes_canceled_orders(self, warehouse):
        db, _ = warehouse
        conn = connect(db, read_only=True)
        rows = {
            r["business_date"]: r
            for r in conn.execute("SELECT * FROM daily_sales ORDER BY business_date")
        }
        conn.close()

        assert rows["2026-07-30"]["net_sales_cents"] == 6_000   # O1 + O2
        assert rows["2026-07-31"]["net_sales_cents"] == 6_000   # O3 + the 1:30am O4
        assert rows["2026-08-01"]["net_sales_cents"] == 3_000   # O5 only; O6 canceled
        assert rows["2026-08-01"]["order_count"] == 1

    def test_daily_payments_splits_by_tender(self, warehouse):
        db, _ = warehouse
        conn = connect(db, read_only=True)
        row = conn.execute("SELECT * FROM daily_payments WHERE business_date = '2026-07-30'").fetchone()
        conn.close()
        assert row["card_cents"] == 4000
        assert row["cash_cents"] == 2000
        assert row["visa_cents"] == 4000

    def test_daily_labor_rolls_up_shifts(self, warehouse):
        db, _ = warehouse
        conn = connect(db, read_only=True)
        row = conn.execute("SELECT * FROM daily_labor WHERE business_date = '2026-07-30'").fetchone()
        conn.close()
        assert row["labor_hours"] == 6.0
        assert row["labor_cost_cents"] == 9000
        assert row["staff_count"] == 1

    def test_daily_item_sales_joins_the_category(self, warehouse):
        db, _ = warehouse
        conn = connect(db, read_only=True)
        row = conn.execute(
            "SELECT * FROM daily_item_sales WHERE item_name = 'Butter Chicken'"
        ).fetchone()
        conn.close()
        assert row["quantity"] == 2.0
        assert row["category_name"] == "Mains"  # joined via variation -> item


class TestFeatureTable:
    def test_one_row_per_observed_day(self, warehouse):
        db, _ = warehouse
        frame = feature_frame(db)
        assert list(frame["business_date"]) == [
            d.isoformat() for d in (date(2026, 7, 25) + timedelta(days=i) for i in range(8))
        ]
        # Days before the first order are real zeros, not gaps.
        assert frame.set_index("business_date").loc["2026-07-28", "target_sales_cents"] == 0

    def test_targets_match_daily_sales(self, warehouse):
        db, _ = warehouse
        frame = feature_frame(db).set_index("business_date")
        assert frame.loc["2026-07-30", "target_sales_cents"] == 6_000
        assert frame.loc["2026-07-31", "target_sales_cents"] == 6_000
        assert frame.loc["2026-08-01", "target_sales_cents"] == 3_000

    def test_lags_look_backwards(self, warehouse):
        db, _ = warehouse
        frame = feature_frame(db).set_index("business_date")
        assert frame.loc["2026-07-31", "sales_lag_1_cents"] == 6_000   # the 30th
        assert frame.loc["2026-08-01", "sales_lag_1_cents"] == 6_000   # the 31st
        assert frame.loc["2026-08-01", "sales_lag_2_cents"] == 6_000   # the 30th

    def test_rolling_average_excludes_today(self, warehouse):
        db, _ = warehouse
        frame = feature_frame(db).set_index("business_date")
        # Prior 7 days of the 1st: 25th-31st = 0,0,0,0,0,6000,6000 -> 12000/7
        assert frame.loc["2026-08-01", "sales_avg_7_cents"] == 12_000 // 7

    def test_weather_joins_on_business_date(self, warehouse):
        db, _ = warehouse
        frame = feature_frame(db).set_index("business_date")
        assert frame.loc["2026-07-31", "is_rainy"] == 1
        assert frame.loc["2026-07-30", "temp_max_f"] == 91.0

    def test_calendar_features_present(self, warehouse):
        db, _ = warehouse
        frame = feature_frame(db).set_index("business_date")
        assert frame.loc["2026-08-01", "day_of_week"] == 6   # Saturday
        assert frame.loc["2026-08-01", "is_weekend"] == 1

    def test_future_rows_are_for_inference_only(self, warehouse):
        db, _ = warehouse
        future = prediction_rows(db)
        assert len(future) == 4  # Aug 2..5, beyond the last observed sale
        assert future["target_sales_cents"].isna().all()
        # Calendar still known for future days — that's the point.
        assert future["day_of_week"].notna().all()

    def test_no_future_rows_leak_into_training(self, warehouse):
        db, _ = warehouse
        frame = feature_frame(db)
        assert frame["business_date"].max() == "2026-08-01"
        assert frame["target_sales_cents"].notna().all()
