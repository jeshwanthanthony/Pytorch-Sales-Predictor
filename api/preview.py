"""A quick look at what a restaurant actually has in Square.

Runs straight after they connect, before any training. It answers the only
question that matters at that point: is there enough history here to forecast
anything? Cheap on purpose — a couple of API calls, no downloading everything.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

from collector.square_api import SquareCollector

from .pipeline_runner import MIN_DAYS_TO_TRAIN
from .workspace import Workspace

log = logging.getLogger("preview")

# how far back to look when sizing up an account
LOOKBACK_DAYS = 730


def summarize_square(workspace: Workspace) -> dict:
    """Count what is there, and say whether it is enough."""
    auth = workspace.auth()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)

    orders = 0
    cancelled = 0
    revenue_cents = 0
    days: defaultdict[str, int] = defaultdict(int)
    first_date: str | None = None
    last_date: str | None = None
    locations: list[dict] = []

    with SquareCollector(auth) as square:
        try:
            locations = square.fetch_locations()
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read locations: %s", exc)

        for order in square.search_orders(start, end):
            row = SquareCollector.normalize_order(order)
            orders += 1

            if row.get("state") == "CANCELED":
                cancelled += 1
                continue

            revenue_cents += row.get("net_sales_cents") or 0
            stamp = (row.get("closed_at") or row.get("created_at") or "")[:10]
            if stamp:
                days[stamp] += 1
                first_date = stamp if first_date is None or stamp < first_date else first_date
                last_date = stamp if last_date is None or stamp > last_date else last_date

    trading_days = len(days)
    return {
        "orders": orders,
        "cancelled_orders": cancelled,
        "sales_orders": orders - cancelled,
        "trading_days": trading_days,
        "first_sale": first_date,
        "last_sale": last_date,
        "total_sales": round(revenue_cents / 100, 2),
        "average_day": round(revenue_cents / 100 / trading_days, 2) if trading_days else 0.0,
        "locations": [
            {"name": loc.get("name"), "city": loc.get("city"), "currency": loc.get("currency")}
            for loc in locations
        ],
        "enough_to_forecast": trading_days >= MIN_DAYS_TO_TRAIN,
        "days_needed": max(0, MIN_DAYS_TO_TRAIN - trading_days),
        "minimum_days": MIN_DAYS_TO_TRAIN,
        "checked_at": date.today().isoformat(),
    }
