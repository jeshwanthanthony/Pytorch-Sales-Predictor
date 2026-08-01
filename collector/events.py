"""Local events and marketing promotions — the two signals no API will hand you.

Both are CSV-driven on purpose. Ticketed-event APIs (Ticketmaster, SeatGeek)
need keys and still miss the school fundraiser two blocks away, and no API knows
you ran a $5-off Instagram promo last Tuesday. A CSV the owner actually keeps
current beats a feed that's 60% right.

Add a paid feed later behind `fetch_events` and nothing downstream changes.
"""

from __future__ import annotations

import csv
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator

from .config import REFERENCE_DIR

log = logging.getLogger(__name__)

EVENTS_FILE = REFERENCE_DIR / "events.csv"
PROMOTIONS_FILE = REFERENCE_DIR / "promotions.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        log.warning("%s not found — no rows from it", path.name)
        return []
    with path.open(encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if any(row.values())]


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def fetch_events(start: date, end: date) -> Iterator[dict[str, Any]]:
    """One row per event *day* — a three-day festival becomes three rows.

    That shape means the daily feature join is a plain group-by on date.
    """
    for row in _read_csv(EVENTS_FILE):
        begins = _parse_date(row.get("start_date"))
        ends = _parse_date(row.get("end_date")) or begins
        if not begins:
            log.warning("skipping event with no start_date: %s", row.get("name"))
            continue

        day = max(begins, start)
        last = min(ends, end)
        while day <= last:
            yield {
                "date": day.isoformat(),
                "name": row.get("name"),
                "category": row.get("category"),  # sports / concert / festival / civic
                "venue": row.get("venue"),
                "distance_miles": _to_float(row.get("distance_miles")),
                "expected_attendance": _to_int(row.get("expected_attendance")),
                "start_time": row.get("start_time") or None,
                "is_multi_day": begins != ends,
                "day_index": (day - begins).days,
                "notes": row.get("notes") or None,
            }
            day += timedelta(days=1)


def fetch_promotions(start: date, end: date) -> Iterator[dict[str, Any]]:
    """One row per day a promotion was live."""
    for row in _read_csv(PROMOTIONS_FILE):
        begins = _parse_date(row.get("start_date"))
        ends = _parse_date(row.get("end_date")) or begins
        if not begins:
            log.warning("skipping promotion with no start_date: %s", row.get("name"))
            continue

        day = max(begins, start)
        last = min(ends, end)
        while day <= last:
            yield {
                "date": day.isoformat(),
                "name": row.get("name"),
                "channel": row.get("channel"),  # facebook_ads / ubereats / in_store / email
                "discount_type": row.get("discount_type"),  # percent / amount / bogo / none
                "discount_value": _to_float(row.get("discount_value")),
                "spend_usd": _to_float(row.get("spend_usd")),
                "applies_to": row.get("applies_to") or None,
                "day_index": (day - begins).days,
                "notes": row.get("notes") or None,
            }
            day += timedelta(days=1)


def _to_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def _to_int(value: str | None) -> int | None:
    parsed = _to_float(value)
    return int(parsed) if parsed is not None else None
