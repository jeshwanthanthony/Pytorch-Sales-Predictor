"""Load data/raw/*.jsonl into the warehouse.

The mapping from raw record to table row lives here and nowhere else. Two
transformations happen on the way in, and both are here because they should
happen once per row rather than once per query:

  1. business_date — a UTC timestamp becomes the local trading day it belongs
     to, with an early-morning cutoff so a 1:30am order counts as the night
     before.
  2. flattening — order line-item modifiers and catalog variations become their
     own rows instead of JSON blobs, because they get grouped by.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .db import DB_PATH, connect, init_schema, optimize, table_counts, transaction, upsert

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

log = logging.getLogger(__name__)

# a restaurant day does not end at midnight, orders before this hour count as
# the day before
DEFAULT_CUTOFF_HOUR = int(os.environ.get("BUSINESS_DAY_CUTOFF_HOUR", 4))
DEFAULT_TIMEZONE = os.environ.get("SITE_TIMEZONE", "America/New_York")


class BusinessDay:
    """Turns a UTC timestamp into the local business date and hour.

    Timezones are per-location, so a second location in another zone stays
    correct without touching any query.
    """

    def __init__(self, timezones: dict[str, str], cutoff_hour: int = DEFAULT_CUTOFF_HOUR):
        self.cutoff_hour = cutoff_hour
        self._zones: dict[str | None, ZoneInfo] = {}
        self._names = timezones
        self._default = self._zone_for_name(DEFAULT_TIMEZONE)

    def _zone_for_name(self, name: str) -> ZoneInfo:
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            log.warning("unknown timezone %r, using UTC", name)
            return ZoneInfo("UTC")

    def zone(self, location_id: str | None) -> ZoneInfo:
        if location_id not in self._zones:
            name = self._names.get(location_id or "")
            self._zones[location_id] = self._zone_for_name(name) if name else self._default
        return self._zones[location_id]

    def of(self, timestamp: str | None, location_id: str | None) -> tuple[str | None, int | None]:
        if not timestamp:
            return None, None
        try:
            moment = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            log.warning("unparseable timestamp %r", timestamp)
            return None, None

        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)

        local = moment.astimezone(self.zone(location_id))
        business = local.date() - timedelta(days=1) if local.hour < self.cutoff_hour else local.date()
        return business.isoformat(), local.hour


@dataclass
class Spec:
    """How one raw entity becomes one or more table rows."""

    entity: str
    table: str
    keys: Sequence[str]
    columns: Sequence[str]
    # Optional row rewrite, e.g. to attach business_date.
    transform: Callable[[dict[str, Any], BusinessDay], dict[str, Any]] | None = None
    # Optional child rows extracted from the same record.
    children: list["Spec"] = field(default_factory=list)
    expand: Callable[[dict[str, Any]], Iterator[dict[str, Any]]] | None = None


def _with_business_date(
    row: dict[str, Any], calendar: BusinessDay, field_name: str = "created_at"
) -> dict[str, Any]:
    date, hour = calendar.of(row.get(field_name), row.get("location_id"))
    return {**row, "business_date": date, "business_hour": hour}


def _expand_modifiers(row: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for modifier in row.get("modifiers") or []:
        yield {
            "order_id": row.get("order_id"),
            "line_item_uid": row.get("line_item_uid"),
            "modifier_uid": modifier.get("uid"),
            "catalog_object_id": modifier.get("catalog_object_id"),
            "name": modifier.get("name"),
            "price_cents": modifier.get("price_cents", 0),
        }


def _expand_variations(row: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for variation in row.get("variations") or []:
        if not variation.get("id"):
            continue
        yield {
            "variation_id": variation["id"],
            "catalog_object_id": row.get("catalog_object_id"),
            "name": variation.get("name"),
            "sku": variation.get("sku"),
            "price_cents": variation.get("price_cents", 0),
        }


SPECS: list[Spec] = [
    Spec(
        entity="locations",
        table="locations",
        keys=["location_id"],
        columns=[
            "location_id", "name", "status", "currency", "country", "state", "city",
            "postal_code", "address_line_1", "latitude", "longitude", "timezone",
            "business_name", "type", "created_at",
        ],
    ),
    Spec(
        entity="calendar",
        table="calendar_days",
        keys=["date"],
        columns=[
            "date", "day_of_week", "day_name", "month", "month_name", "year",
            "day_of_month", "day_of_year", "week_of_year", "quarter", "is_weekend",
            "is_weekend_night", "is_holiday", "holiday_name", "is_holiday_eve",
            "is_day_after_holiday", "observance", "is_observance", "school_break",
            "is_school_break", "is_month_start", "is_month_end", "is_payday_window",
        ],
    ),
    Spec(
        entity="weather",
        table="weather",
        keys=["date"],
        columns=[
            "date", "latitude", "longitude", "source", "temp_max_f", "temp_min_f",
            "temp_mean_f", "feels_like_max_f", "feels_like_min_f", "precipitation_in",
            "rain_in", "snowfall_in", "precipitation_hours", "wind_max_mph",
            "wind_gust_mph", "humidity_mean", "weather_code", "sunrise", "sunset",
            "is_rainy", "is_snowy", "is_stormy",
        ],
    ),
    Spec(
        entity="events",
        table="events",
        keys=["date", "name"],
        columns=[
            "date", "name", "category", "venue", "distance_miles",
            "expected_attendance", "start_time", "is_multi_day", "day_index", "notes",
        ],
    ),
    Spec(
        entity="promotions",
        table="promotions",
        keys=["date", "name"],
        columns=[
            "date", "name", "channel", "discount_type", "discount_value", "spend_usd",
            "applies_to", "day_index", "notes",
        ],
    ),
    Spec(
        entity="catalog",
        table="catalog_items",
        keys=["catalog_object_id"],
        columns=[
            "catalog_object_id", "item_name", "description", "category_id",
            "category_name", "product_type", "is_archived", "is_deleted", "updated_at",
            "version", "modifier_list_ids",
        ],
        children=[
            Spec(
                entity="catalog",
                table="catalog_variations",
                keys=["variation_id"],
                columns=["variation_id", "catalog_object_id", "name", "sku", "price_cents"],
                expand=_expand_variations,
            )
        ],
    ),
    Spec(
        entity="orders",
        table="orders",
        keys=["order_id"],
        columns=[
            "order_id", "location_id", "created_at", "updated_at", "closed_at",
            "business_date", "business_hour", "state", "currency", "revenue_cents",
            "discount_cents", "tax_cents", "tip_cents", "service_charge_cents",
            "net_sales_cents", "source", "fulfillment_type", "fulfillment_state",
            "customer_id", "payment_types", "tender_count", "line_item_count",
            "item_quantity", "has_returns", "ticket_name", "version", "ingested_at",
        ],
        transform=lambda row, cal: {
            **_with_business_date(row, cal),
            "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    ),
    Spec(
        entity="order_items",
        table="order_items",
        keys=["order_id", "line_item_uid"],
        columns=[
            "order_id", "line_item_uid", "line_number", "location_id", "created_at",
            "business_date", "catalog_object_id", "item_name", "variation_name",
            "quantity", "unit", "base_price_cents", "gross_sales_cents",
            "discount_cents", "tax_cents", "total_cents", "item_type", "note",
        ],
        transform=_with_business_date,
        children=[
            Spec(
                entity="order_items",
                table="order_item_modifiers",
                keys=["order_id", "line_item_uid", "modifier_uid"],
                columns=[
                    "order_id", "line_item_uid", "modifier_uid", "catalog_object_id",
                    "name", "price_cents",
                ],
                expand=_expand_modifiers,
            )
        ],
    ),
    Spec(
        entity="payments",
        table="payments",
        keys=["payment_id"],
        columns=[
            "payment_id", "order_id", "location_id", "customer_id", "created_at",
            "updated_at", "business_date", "status", "amount_cents", "tip_cents",
            "app_fee_cents", "refunded_cents", "approved_cents", "currency",
            "processing_fee_cents", "source_type", "card_brand", "card_type", "last_4",
            "entry_method", "receipt_number", "team_member_id",
        ],
        transform=_with_business_date,
    ),
    Spec(
        entity="refunds",
        table="refunds",
        keys=["refund_id"],
        columns=[
            "refund_id", "payment_id", "order_id", "location_id", "created_at",
            "updated_at", "business_date", "status", "amount_cents", "currency",
            "processing_fee_cents", "reason", "destination_type", "team_member_id",
        ],
        transform=_with_business_date,
    ),
    Spec(
        entity="customers",
        table="customers",
        keys=["customer_id"],
        columns=[
            "customer_id", "created_at", "updated_at", "given_name", "family_name",
            "email", "phone", "birthday", "reference_id", "company_name",
            "creation_source", "group_ids", "segment_ids", "email_unsubscribed",
            "postal_code", "note",
        ],
    ),
    Spec(
        entity="inventory",
        table="inventory_counts",
        keys=["catalog_object_id", "location_id", "state"],
        columns=[
            "catalog_object_id", "location_id", "state", "catalog_object_type",
            "quantity", "calculated_at",
        ],
    ),
    Spec(
        entity="team_members",
        table="team_members",
        keys=["team_member_id"],
        columns=[
            "team_member_id", "reference_id", "status", "given_name", "family_name",
            "is_owner", "created_at", "updated_at", "assignment_type", "location_ids",
        ],
    ),
    Spec(
        entity="shifts",
        table="shifts",
        keys=["shift_id"],
        columns=[
            "shift_id", "team_member_id", "location_id", "start_at", "end_at",
            "business_date", "hours", "status", "job_id", "job_title",
            "hourly_rate_cents", "labor_cost_cents", "declared_tips_cents", "timezone",
        ],
        transform=lambda row, cal: _with_business_date(row, cal, "start_at"),
    ),
]

# locations first: every other entity's business_date depends on its timezone.
LOAD_ORDER = [spec.entity for spec in SPECS]


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("%s:%d is not valid JSON, skipping", path.name, number)


def raw_files(entity: str, raw_dir: Path = RAW_DIR) -> list[Path]:
    """Every run's file for an entity, oldest first.

    Order matters: a later run's version of a row must win the upsert.
    """
    directory = raw_dir / entity
    if not directory.exists():
        return []
    return sorted(directory.glob(f"{entity}-*.jsonl"))


def already_loaded(conn: sqlite3.Connection) -> set[str]:
    return {row["path"] for row in conn.execute("SELECT path FROM ingested_files")}


def location_timezones(conn: sqlite3.Connection) -> dict[str, str]:
    return {
        row["location_id"]: row["timezone"]
        for row in conn.execute("SELECT location_id, timezone FROM locations WHERE timezone IS NOT NULL")
    }


def load_spec(
    conn: sqlite3.Connection,
    spec: Spec,
    calendar: BusinessDay,
    force: bool,
    raw_dir: Path = RAW_DIR,
) -> tuple[int, int]:
    """Load one entity. Returns (rows written, files processed)."""
    seen = set() if force else already_loaded(conn)
    rows_total = 0
    files_done = 0

    for path in raw_files(spec.entity, raw_dir):
        # just the filename, it is already unique and it keeps working if the
        # project folder moves
        key = path.name
        if key in seen:
            continue

        records = list(read_jsonl(path))
        if spec.transform:
            records = [spec.transform(record, calendar) for record in records]

        with transaction(conn):
            written = upsert(conn, spec.table, spec.columns, spec.keys, records)
            for child in spec.children:
                child_rows = (out for record in records for out in child.expand(record))
                upsert(conn, child.table, child.columns, child.keys, child_rows)

            conn.execute(
                "INSERT INTO ingested_files (path, entity, rows, loaded_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET rows = excluded.rows, loaded_at = excluded.loaded_at",
                (key, spec.entity, written, datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )

        rows_total += written
        files_done += 1

    return rows_total, files_done


def load_all(
    db_path: Path = DB_PATH,
    only: Iterable[str] | None = None,
    force: bool = False,
    cutoff_hour: int = DEFAULT_CUTOFF_HOUR,
    raw_dir: Path = RAW_DIR,
    on_progress: Callable[[str, int], None] | None = None,
) -> dict[str, int]:
    wanted = set(only) if only else None
    conn = connect(db_path)
    init_schema(conn)

    results: dict[str, int] = {}
    try:
        # locations go first, everything else needs their timezone
        for spec in SPECS:
            if wanted and spec.entity not in wanted:
                continue

            calendar = BusinessDay(location_timezones(conn), cutoff_hour)
            rows, files = load_spec(conn, spec, calendar, force, raw_dir)
            results[spec.entity] = rows
            if on_progress:
                on_progress(spec.entity, rows)

            if files:
                log.info("%-14s %7d rows from %d file(s)", spec.entity, rows, files)
            else:
                log.debug("%-14s nothing new", spec.entity)

        optimize(conn)
    finally:
        conn.close()

    return results


def summarize(db_path: Path = DB_PATH) -> dict[str, Any]:
    conn = connect(db_path, read_only=True)
    try:
        counts = table_counts(conn)
        span = conn.execute(
            "SELECT MIN(business_date) AS first, MAX(business_date) AS last, "
            "COUNT(*) AS days FROM daily_sales"
        ).fetchone()
        size_mb = Path(db_path).stat().st_size / 1_000_000 if Path(db_path).exists() else 0
        return {
            "counts": counts,
            "first_sales_date": span["first"],
            "last_sales_date": span["last"],
            "days_with_sales": span["days"],
            "size_mb": round(size_mb, 2),
        }
    finally:
        conn.close()
