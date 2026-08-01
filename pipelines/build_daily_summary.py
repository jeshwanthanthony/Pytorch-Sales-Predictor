"""Collapse orders into one clean row per trading day per location.

The database's `daily_sales` view already sums orders by day. This goes further,
and lands the result in a real table:

  * every date in the calendar appears, including days the restaurant was shut,
    so a closure reads as a zero rather than a hole in the series
  * dayparts — lunch / dinner / late — from each order's local hour
  * new vs returning customers, from each customer's first ever order
  * derived operating ratios (sales per labor hour, refund rate)

Written as a table rather than a view because build_features reads it several
times and the daypart and first-visit logic are the expensive parts.

    python -m pipelines.build_daily_summary
    python -m pipelines.build_daily_summary --since 2026-01-01
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Iterator

from database.db import DB_PATH, connect, optimize, transaction, upsert

log = logging.getLogger("daily_summary")

# local hour boundaries, anything before LUNCH_START already counts as
# yesterday, so "late" means the tail of the same evening
LUNCH_START, LUNCH_END = 11, 16
DINNER_START, DINNER_END = 16, 22

TABLE = "daily_summary"

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    business_date            TEXT NOT NULL,
    location_id              TEXT NOT NULL,

    is_closed                INTEGER NOT NULL DEFAULT 0,
    order_count              INTEGER NOT NULL DEFAULT 0,
    revenue_cents            INTEGER NOT NULL DEFAULT 0,
    net_sales_cents          INTEGER NOT NULL DEFAULT 0,
    discount_cents           INTEGER NOT NULL DEFAULT 0,
    tax_cents                INTEGER NOT NULL DEFAULT 0,
    tip_cents                INTEGER NOT NULL DEFAULT 0,
    service_charge_cents     INTEGER NOT NULL DEFAULT 0,
    avg_ticket_cents         INTEGER NOT NULL DEFAULT 0,

    item_count               REAL NOT NULL DEFAULT 0,
    distinct_items           INTEGER NOT NULL DEFAULT 0,

    customer_count           INTEGER NOT NULL DEFAULT 0,
    new_customer_count       INTEGER NOT NULL DEFAULT 0,
    returning_customer_count INTEGER NOT NULL DEFAULT 0,

    lunch_orders             INTEGER NOT NULL DEFAULT 0,
    dinner_orders            INTEGER NOT NULL DEFAULT 0,
    late_orders              INTEGER NOT NULL DEFAULT 0,
    lunch_sales_cents        INTEGER NOT NULL DEFAULT 0,
    dinner_sales_cents       INTEGER NOT NULL DEFAULT 0,
    late_sales_cents         INTEGER NOT NULL DEFAULT 0,
    peak_hour                INTEGER,

    pickup_orders            INTEGER NOT NULL DEFAULT 0,
    delivery_orders          INTEGER NOT NULL DEFAULT 0,
    dine_in_orders           INTEGER NOT NULL DEFAULT 0,

    cash_cents               INTEGER NOT NULL DEFAULT 0,
    card_cents               INTEGER NOT NULL DEFAULT 0,
    gift_card_cents          INTEGER NOT NULL DEFAULT 0,
    processing_fee_cents     INTEGER NOT NULL DEFAULT 0,

    refund_count             INTEGER NOT NULL DEFAULT 0,
    refund_cents             INTEGER NOT NULL DEFAULT 0,
    refund_rate_bps          INTEGER NOT NULL DEFAULT 0,

    labor_hours              REAL NOT NULL DEFAULT 0,
    labor_cost_cents         INTEGER NOT NULL DEFAULT 0,
    staff_count              INTEGER NOT NULL DEFAULT 0,
    sales_per_labor_hour_cents INTEGER NOT NULL DEFAULT 0,
    labor_cost_ratio_bps     INTEGER NOT NULL DEFAULT 0,

    built_at                 TEXT NOT NULL,
    PRIMARY KEY (business_date, location_id)
) STRICT;

CREATE INDEX IF NOT EXISTS ix_summary_location ON {TABLE} (location_id, business_date);
"""

COLUMNS = [
    "business_date", "location_id", "is_closed", "order_count", "revenue_cents",
    "net_sales_cents", "discount_cents", "tax_cents", "tip_cents", "service_charge_cents",
    "avg_ticket_cents", "item_count", "distinct_items", "customer_count",
    "new_customer_count", "returning_customer_count", "lunch_orders", "dinner_orders",
    "late_orders", "lunch_sales_cents", "dinner_sales_cents", "late_sales_cents",
    "peak_hour", "pickup_orders", "delivery_orders", "dine_in_orders", "cash_cents",
    "card_cents", "gift_card_cents", "processing_fee_cents", "refund_count",
    "refund_cents", "refund_rate_bps", "labor_hours", "labor_cost_cents", "staff_count",
    "sales_per_labor_hour_cents", "labor_cost_ratio_bps", "built_at",
]

# the adding up stays in SQL, one pass over the indexes beats pulling thousands
# of orders into python. the spine is calendar x locations so closed days still
# get a row of zeros
BUILD_SQL = f"""
WITH spine AS (
    SELECT c.date AS business_date, l.location_id
    FROM calendar_days c
    CROSS JOIN locations l
    WHERE c.date >= :since
      AND c.date <= (SELECT MAX(business_date) FROM orders WHERE business_date IS NOT NULL)
),
-- each customer's first ever order, so we can split new from returning
first_seen AS (
    SELECT customer_id, MIN(business_date) AS first_date
    FROM orders
    WHERE customer_id IS NOT NULL AND state <> 'CANCELED' AND business_date IS NOT NULL
    GROUP BY customer_id
),
order_rollup AS (
    SELECT
        o.business_date,
        o.location_id,
        COUNT(*)                                    AS order_count,
        SUM(o.revenue_cents)                        AS revenue_cents,
        SUM(o.net_sales_cents)                      AS net_sales_cents,
        SUM(o.discount_cents)                       AS discount_cents,
        SUM(o.tax_cents)                            AS tax_cents,
        SUM(o.tip_cents)                            AS tip_cents,
        SUM(o.service_charge_cents)                 AS service_charge_cents,
        CAST(AVG(o.revenue_cents) AS INTEGER)       AS avg_ticket_cents,
        SUM(o.item_quantity)                        AS item_count,
        COUNT(DISTINCT o.customer_id)               AS customer_count,
        COUNT(DISTINCT CASE WHEN f.first_date = o.business_date THEN o.customer_id END)
                                                    AS new_customer_count,
        SUM(CASE WHEN o.business_hour >= {LUNCH_START} AND o.business_hour < {LUNCH_END}
                 THEN 1 ELSE 0 END)                 AS lunch_orders,
        SUM(CASE WHEN o.business_hour >= {DINNER_START} AND o.business_hour < {DINNER_END}
                 THEN 1 ELSE 0 END)                 AS dinner_orders,
        SUM(CASE WHEN o.business_hour >= {DINNER_END} OR o.business_hour < {LUNCH_START}
                 THEN 1 ELSE 0 END)                 AS late_orders,
        SUM(CASE WHEN o.business_hour >= {LUNCH_START} AND o.business_hour < {LUNCH_END}
                 THEN o.net_sales_cents ELSE 0 END) AS lunch_sales_cents,
        SUM(CASE WHEN o.business_hour >= {DINNER_START} AND o.business_hour < {DINNER_END}
                 THEN o.net_sales_cents ELSE 0 END) AS dinner_sales_cents,
        SUM(CASE WHEN o.business_hour >= {DINNER_END} OR o.business_hour < {LUNCH_START}
                 THEN o.net_sales_cents ELSE 0 END) AS late_sales_cents,
        SUM(CASE WHEN o.fulfillment_type = 'PICKUP'   THEN 1 ELSE 0 END) AS pickup_orders,
        SUM(CASE WHEN o.fulfillment_type = 'DELIVERY' THEN 1 ELSE 0 END) AS delivery_orders,
        SUM(CASE WHEN o.fulfillment_type IS NULL      THEN 1 ELSE 0 END) AS dine_in_orders
    FROM orders o
    LEFT JOIN first_seen f ON f.customer_id = o.customer_id
    WHERE o.state <> 'CANCELED' AND o.business_date IS NOT NULL AND o.business_date >= :since
    GROUP BY o.business_date, o.location_id
),
-- busiest hour of the day, useful later for staffing
peak AS (
    SELECT business_date, location_id, business_hour AS peak_hour
    FROM (
        SELECT business_date, location_id, business_hour,
               ROW_NUMBER() OVER (
                   PARTITION BY business_date, location_id ORDER BY COUNT(*) DESC, business_hour
               ) AS rank
        FROM orders
        WHERE state <> 'CANCELED' AND business_date IS NOT NULL AND business_hour IS NOT NULL
        GROUP BY business_date, location_id, business_hour
    )
    WHERE rank = 1
),
items AS (
    SELECT business_date, location_id, COUNT(DISTINCT item_name) AS distinct_items
    FROM order_items
    WHERE business_date IS NOT NULL
    GROUP BY business_date, location_id
)
SELECT
    s.business_date,
    s.location_id,
    (o.order_count IS NULL)                        AS is_closed,
    COALESCE(o.order_count, 0)                     AS order_count,
    COALESCE(o.revenue_cents, 0)                   AS revenue_cents,
    COALESCE(o.net_sales_cents, 0)                 AS net_sales_cents,
    COALESCE(o.discount_cents, 0)                  AS discount_cents,
    COALESCE(o.tax_cents, 0)                       AS tax_cents,
    COALESCE(o.tip_cents, 0)                       AS tip_cents,
    COALESCE(o.service_charge_cents, 0)            AS service_charge_cents,
    COALESCE(o.avg_ticket_cents, 0)                AS avg_ticket_cents,
    COALESCE(o.item_count, 0)                      AS item_count,
    COALESCE(i.distinct_items, 0)                  AS distinct_items,
    COALESCE(o.customer_count, 0)                  AS customer_count,
    COALESCE(o.new_customer_count, 0)              AS new_customer_count,
    COALESCE(o.customer_count, 0) - COALESCE(o.new_customer_count, 0)
                                                   AS returning_customer_count,
    COALESCE(o.lunch_orders, 0)                    AS lunch_orders,
    COALESCE(o.dinner_orders, 0)                   AS dinner_orders,
    COALESCE(o.late_orders, 0)                     AS late_orders,
    COALESCE(o.lunch_sales_cents, 0)               AS lunch_sales_cents,
    COALESCE(o.dinner_sales_cents, 0)              AS dinner_sales_cents,
    COALESCE(o.late_sales_cents, 0)                AS late_sales_cents,
    pk.peak_hour                                   AS peak_hour,
    COALESCE(o.pickup_orders, 0)                   AS pickup_orders,
    COALESCE(o.delivery_orders, 0)                 AS delivery_orders,
    COALESCE(o.dine_in_orders, 0)                  AS dine_in_orders,
    COALESCE(pm.cash_cents, 0)                     AS cash_cents,
    COALESCE(pm.card_cents, 0)                     AS card_cents,
    COALESCE(pm.gift_card_cents, 0)                AS gift_card_cents,
    COALESCE(pm.processing_fee_cents, 0)           AS processing_fee_cents,
    COALESCE(rf.refund_count, 0)                   AS refund_count,
    COALESCE(rf.refund_cents, 0)                   AS refund_cents,
    COALESCE(lb.labor_hours, 0)                    AS labor_hours,
    COALESCE(lb.labor_cost_cents, 0)               AS labor_cost_cents,
    COALESCE(lb.staff_count, 0)                    AS staff_count
FROM spine s
LEFT JOIN order_rollup o  ON o.business_date = s.business_date AND o.location_id = s.location_id
LEFT JOIN peak pk         ON pk.business_date = s.business_date AND pk.location_id = s.location_id
LEFT JOIN items i         ON i.business_date = s.business_date AND i.location_id = s.location_id
LEFT JOIN daily_payments pm ON pm.business_date = s.business_date AND pm.location_id = s.location_id
LEFT JOIN daily_refunds rf  ON rf.business_date = s.business_date AND rf.location_id = s.location_id
LEFT JOIN daily_labor lb    ON lb.business_date = s.business_date AND lb.location_id = s.location_id
ORDER BY s.location_id, s.business_date
"""


def _ratios(row: dict[str, Any], built_at: str) -> dict[str, Any]:
    """Operating ratios, in basis points so they stay integers.

    Division belongs here rather than in SQL because the zero cases (a closed
    day, a day with no shifts recorded) need a deliberate answer, not a NULL
    that silently becomes NaN in a tensor.
    """
    net = row["net_sales_cents"]
    labor_hours = row["labor_hours"]
    labor_cost = row["labor_cost_cents"]
    refunds = row["refund_cents"]

    return {
        **row,
        "sales_per_labor_hour_cents": int(net / labor_hours) if labor_hours else 0,
        "labor_cost_ratio_bps": int(labor_cost / net * 10_000) if net else 0,
        "refund_rate_bps": int(refunds / net * 10_000) if net else 0,
        "built_at": built_at,
    }


def build(db_path: Path | str = DB_PATH, since: str = "0000-01-01") -> int:
    """Rebuild the summary for every date on or after `since`. Idempotent."""
    from datetime import datetime, timezone

    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)

        def rows() -> Iterator[dict[str, Any]]:
            for record in conn.execute(BUILD_SQL, {"since": since}):
                yield _ratios(dict(record), built_at)

        with transaction(conn):
            written = upsert(conn, TABLE, COLUMNS, ["business_date", "location_id"], rows())

        closed = conn.execute(
            f"SELECT COUNT(*) FROM {TABLE} WHERE is_closed = 1 AND business_date >= ?", (since,)
        ).fetchone()[0]
        optimize(conn)
    finally:
        conn.close()

    log.info("%s: %d day-rows (%d with no sales)", TABLE, written, closed)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the daily summary table.")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--since", default="0000-01-01", help="rebuild from this date forward")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    rows = build(args.db, args.since)
    if not rows:
        log.warning("nothing built — is there any order data? try `python -m database stats`")
        return 1

    conn = connect(args.db, read_only=True)
    try:
        sample = conn.execute(
            f"SELECT business_date, order_count, net_sales_cents, lunch_orders, dinner_orders, "
            f"peak_hour, is_closed FROM {TABLE} ORDER BY business_date DESC LIMIT 7"
        ).fetchall()
    finally:
        conn.close()

    print(f"\n{'date':<12} {'orders':>7} {'net sales':>11} {'lunch':>6} {'dinner':>7} {'peak':>5}")
    for row in sample:
        marker = "  (closed)" if row["is_closed"] else ""
        print(
            f"{row['business_date']:<12} {row['order_count']:>7} "
            f"${row['net_sales_cents']/100:>10,.0f} {row['lunch_orders']:>6} "
            f"{row['dinner_orders']:>7} {str(row['peak_hour'] or '-'):>5}{marker}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
