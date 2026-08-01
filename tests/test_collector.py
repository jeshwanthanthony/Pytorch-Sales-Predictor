"""Tests for the parts of the collector that don't need a live Square token:
payload normalization, pagination, and the calendar/events sources.
"""

from __future__ import annotations

from datetime import date

import pytest

from collector.calendar_api import CalendarCollector
from collector.config import SiteConfig
from collector.events import fetch_events, fetch_promotions
from collector.http import paginate
from collector.square_api import SquareCollector, _hours_between
from collector.weather_api import _weather_flags

SITE = SiteConfig(38.8816, -77.0910, "America/New_York", "US", "VA")

ORDER = {
    "id": "ORDER1",
    "location_id": "LOC1",
    "created_at": "2026-07-30T18:15:00Z",
    "updated_at": "2026-07-30T18:42:00Z",
    "closed_at": "2026-07-30T18:40:00Z",
    "state": "COMPLETED",
    "source": {"name": "Square Online"},
    "total_money": {"amount": 5432, "currency": "USD"},
    "total_tax_money": {"amount": 432, "currency": "USD"},
    "total_tip_money": {"amount": 800, "currency": "USD"},
    "total_discount_money": {"amount": 300, "currency": "USD"},
    "tenders": [{"type": "CARD"}, {"type": "CASH"}],
    "fulfillments": [
        {
            "type": "PICKUP",
            "state": "COMPLETED",
            "pickup_details": {"recipient": {"customer_id": "CUST9"}},
        }
    ],
    "line_items": [
        {
            "uid": "LI1",
            "name": "Butter Chicken",
            "variation_name": "Regular",
            "quantity": "2",
            "catalog_object_id": "VAR_BC",
            "base_price_money": {"amount": 1600},
            "gross_sales_money": {"amount": 3200},
            "total_money": {"amount": 3200},
            "modifiers": [{"uid": "M1", "name": "Extra spicy", "base_price_money": {"amount": 0}}],
        },
        {
            "uid": "LI2",
            "name": "Garlic Naan",
            "quantity": "3",
            "base_price_money": {"amount": 400},
            "total_money": {"amount": 1200},
        },
    ],
}


class TestNormalizeOrder:
    def test_money_and_net_sales(self):
        row = SquareCollector.normalize_order(ORDER)
        assert row["revenue_cents"] == 5432
        assert row["tax_cents"] == 432
        assert row["tip_cents"] == 800
        # Net sales strips tax and tip — what the kitchen actually sold.
        assert row["net_sales_cents"] == 5432 - 432 - 800

    def test_customer_id_falls_back_to_fulfillment_recipient(self):
        assert SquareCollector.normalize_order(ORDER)["customer_id"] == "CUST9"

    def test_order_shape(self):
        row = SquareCollector.normalize_order(ORDER)
        assert row["payment_types"] == ["CARD", "CASH"]
        assert row["fulfillment_type"] == "PICKUP"
        assert row["source"] == "Square Online"
        assert row["line_item_count"] == 2
        assert row["item_quantity"] == 5.0

    def test_missing_money_is_zero_not_none(self):
        row = SquareCollector.normalize_order({"id": "X", "location_id": "L"})
        assert row["revenue_cents"] == 0
        assert row["tip_cents"] == 0
        assert row["order_id"] == "X"


class TestOrderItems:
    def test_one_row_per_line_item(self):
        items = list(SquareCollector.fetch_order_items([ORDER]))
        assert len(items) == 2
        assert [i["item_name"] for i in items] == ["Butter Chicken", "Garlic Naan"]

    def test_carries_order_context_and_modifiers(self):
        first = list(SquareCollector.fetch_order_items([ORDER]))[0]
        assert first["order_id"] == "ORDER1"
        assert first["location_id"] == "LOC1"
        assert first["created_at"] == "2026-07-30T18:15:00Z"
        assert first["quantity"] == 2.0
        assert first["modifier_names"] == ["Extra spicy"]
        # Category is joined from the catalog later, so it's absent here.
        assert "category_id" not in first


class TestPagination:
    def test_follows_cursor_until_exhausted(self):
        pages = [
            {"orders": [{"id": 1}, {"id": 2}], "cursor": "c1"},
            {"orders": [{"id": 3}], "cursor": "c2"},
            {"orders": [{"id": 4}]},
        ]
        seen_cursors = []

        def fetch(cursor):
            seen_cursors.append(cursor)
            return pages[len(seen_cursors) - 1]

        rows = list(paginate(fetch, "orders"))
        assert [r["id"] for r in rows] == [1, 2, 3, 4]
        assert seen_cursors == [None, "c1", "c2"]

    def test_empty_response_terminates(self):
        assert list(paginate(lambda cursor: {}, "orders")) == []


class TestCalendar:
    def test_federal_holiday_is_flagged(self):
        row = CalendarCollector(SITE).describe(date(2026, 7, 4))
        assert row["is_holiday"] is True
        assert "Independence Day" in row["holiday_name"]

    def test_holiday_eve_and_day_after(self):
        calendar = CalendarCollector(SITE)
        assert calendar.describe(date(2026, 7, 3))["is_holiday_eve"] is True
        assert calendar.describe(date(2026, 7, 5))["is_day_after_holiday"] is True

    def test_weekend_and_weekend_night(self):
        calendar = CalendarCollector(SITE)
        friday = calendar.describe(date(2026, 7, 31))
        assert friday["is_weekend"] is False
        # Friday trades like a weekend even though it's a workday.
        assert friday["is_weekend_night"] is True
        assert calendar.describe(date(2026, 8, 1))["is_weekend"] is True

    def test_observance_not_a_day_off(self):
        row = CalendarCollector(SITE).describe(date(2026, 2, 14))
        assert row["observance"] == "Valentines Day"
        assert row["is_holiday"] is False

    def test_covers_future_dates(self):
        rows = list(CalendarCollector(SITE).fetch(date(2027, 1, 1), date(2027, 1, 7)))
        assert len(rows) == 7
        assert rows[0]["holiday_name"] == "New Year's Day"


class TestReferenceFiles:
    def test_multi_day_event_expands_to_one_row_per_day(self):
        rows = [r for r in fetch_events(date(2026, 8, 1), date(2026, 9, 1)) if "Fair" in r["name"]]
        assert len(rows) == 5  # 2026-08-19 .. 2026-08-23
        assert rows[0]["day_index"] == 0
        assert rows[-1]["day_index"] == 4
        assert all(r["is_multi_day"] for r in rows)

    def test_window_clips_events(self):
        rows = list(fetch_events(date(2026, 8, 20), date(2026, 8, 21)))
        assert {r["date"] for r in rows} == {"2026-08-20", "2026-08-21"}

    def test_promotions_parse_numbers(self):
        rows = [
            r
            for r in fetch_promotions(date(2026, 7, 15), date(2026, 7, 16))
            if r["channel"] == "ubereats"
        ]
        assert rows and rows[0]["discount_value"] == 20.0
        assert rows[0]["discount_type"] == "percent"


class TestHelpers:
    @pytest.mark.parametrize(
        "code,precip,snow,expected",
        [
            (0, 0.0, 0.0, {"is_rainy": False, "is_snowy": False, "is_stormy": False}),
            (61, 0.4, 0.0, {"is_rainy": True, "is_snowy": False, "is_stormy": False}),
            (73, 0.2, 1.5, {"is_rainy": True, "is_snowy": True, "is_stormy": False}),
            (95, 0.8, 0.0, {"is_rainy": True, "is_snowy": False, "is_stormy": True}),
        ],
    )
    def test_weather_flags(self, code, precip, snow, expected):
        assert _weather_flags(code, precip, snow) == expected

    def test_shift_hours(self):
        assert _hours_between("2026-07-30T16:00:00Z", "2026-07-30T22:30:00Z") == 6.5

    def test_open_shift_has_no_hours(self):
        assert _hours_between("2026-07-30T16:00:00Z", None) == 0.0
