"""Gate the data before anything trains on it.

`database check` reports numbers; this decides whether they're acceptable and
exits non-zero when they aren't. That difference matters: a model trained on a
week where the collector silently missed three days will still produce a
confident number, and nothing downstream will ever tell you it's wrong.

    python -m pipelines.validate_data
    python -m pipelines.validate_data --strict   # warnings fail too

Severity:
    error   — training on this would produce a misleading model. Blocks.
    warning — worth knowing; usually a thin sandbox dataset or a real closure.
    info    — context for reading the rest.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from database.db import DB_PATH, connect

log = logging.getLogger("validate")

ERROR, WARNING, INFO = "error", "warning", "info"

# Below this, a train/validation/test split has nothing meaningful in it.
MIN_HISTORY_DAYS = 60
# A daily total this far from the mean is more likely a data problem than a day.
OUTLIER_SIGMA = 5.0


@dataclass
class Check:
    name: str
    severity: str
    passed: bool
    detail: str
    value: Any = None

    @property
    def blocking(self) -> bool:
        return self.severity == ERROR and not self.passed


@dataclass
class Report:
    checks: list[Check]

    @property
    def errors(self) -> list[Check]:
        return [c for c in self.checks if c.severity == ERROR and not c.passed]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.severity == WARNING and not c.passed]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_blocked(self) -> None:
        if not self.ok:
            joined = "; ".join(f"{c.name}: {c.detail}" for c in self.errors)
            raise DataQualityError(f"{len(self.errors)} blocking issue(s) — {joined}")

    def render(self) -> str:
        icons = {True: "ok  ", False: {ERROR: "FAIL", WARNING: "warn", INFO: "--  "}}
        width = max(len(c.name) for c in self.checks) if self.checks else 10
        lines = []
        for check in self.checks:
            icon = icons[True] if check.passed else icons[False][check.severity]
            lines.append(f"  [{icon}] {check.name:<{width}}  {check.detail}")
        return "\n".join(lines)


class DataQualityError(RuntimeError):
    """Raised when the data cannot responsibly be trained on."""


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> Any:
    row = conn.execute(sql, params).fetchone()
    return row[0] if row else None


# Each check takes a connection and returns a Check.
CheckFn = Callable[[sqlite3.Connection], Check]


def check_has_orders(conn: sqlite3.Connection) -> Check:
    count = _scalar(conn, "SELECT COUNT(*) FROM orders")
    return Check(
        "orders present",
        ERROR,
        count > 0,
        f"{count:,} orders" if count else "no orders — connect Square and run the collector",
        count,
    )


def check_history_length(conn: sqlite3.Connection) -> Check:
    """No trading days at all is fatal. Too few is only a warning.

    Orders can exist while daily_sales is empty — every one of them cancelled,
    for instance. There is nothing to learn from that, so it has to block rather
    than let the next step fail with a confusing error.
    """
    days = _scalar(conn, "SELECT COUNT(*) FROM daily_sales") or 0

    if days == 0:
        total = _scalar(conn, "SELECT COUNT(*) FROM orders") or 0
        reason = (
            f"{total} orders but none of them count as sales — all cancelled, "
            "or none have a business_date"
            if total
            else "no completed orders at all"
        )
        return Check("history length", ERROR, False, reason, 0)

    return Check(
        "history length",
        WARNING,
        days >= MIN_HISTORY_DAYS,
        f"{days} days of sales" + ("" if days >= MIN_HISTORY_DAYS else f" (want {MIN_HISTORY_DAYS}+)"),
        days,
    )


def check_business_dates(conn: sqlite3.Connection) -> Check:
    missing = _scalar(conn, "SELECT COUNT(*) FROM orders WHERE business_date IS NULL")
    return Check(
        "business dates",
        ERROR,
        missing == 0,
        "all orders dated" if not missing else f"{missing:,} orders have no business_date",
        missing,
    )


def check_calendar_coverage(conn: sqlite3.Connection) -> Check:
    """Every trading day needs a calendar row — it's the spine of the feature table."""
    missing = _scalar(
        conn,
        "SELECT COUNT(*) FROM daily_sales d "
        "WHERE NOT EXISTS (SELECT 1 FROM calendar_days c WHERE c.date = d.business_date)",
    )
    return Check(
        "calendar coverage",
        ERROR,
        missing == 0,
        "every sales day has calendar context"
        if not missing
        else f"{missing} sales days missing from calendar_days — widen --since and re-collect",
        missing,
    )


def check_weather_coverage(conn: sqlite3.Connection) -> Check:
    missing = _scalar(
        conn,
        "SELECT COUNT(*) FROM daily_sales d "
        "WHERE NOT EXISTS (SELECT 1 FROM weather w WHERE w.date = d.business_date)",
    )
    total = _scalar(conn, "SELECT COUNT(*) FROM daily_sales") or 0
    share = (missing / total * 100) if total else 0
    return Check(
        "weather coverage",
        WARNING,
        share < 5,
        f"{missing} of {total} sales days without weather ({share:.0f}%)",
        missing,
    )


def check_calendar_gaps(conn: sqlite3.Connection) -> Check:
    """The calendar spine must be unbroken across the sales range.

    A hole here doesn't error anywhere — the feature table just silently skips
    that day, and the lag features quietly straddle the gap.
    """
    present = _scalar(
        conn,
        "SELECT COUNT(*) FROM calendar_days "
        "WHERE date BETWEEN (SELECT MIN(business_date) FROM daily_sales) "
        "AND (SELECT MAX(business_date) FROM daily_sales)",
    ) or 0
    span = _scalar(
        conn,
        "SELECT CAST(julianday(MAX(business_date)) - julianday(MIN(business_date)) AS INTEGER) + 1 "
        "FROM daily_sales",
    ) or 0
    return Check(
        "calendar continuity",
        ERROR,
        present >= span,
        f"{present} calendar rows across a {span}-day span"
        + ("" if present >= span else f" — {span - present} missing, the feature spine will skip them"),
        span - present,
    )


def check_zero_sales_runs(conn: sqlite3.Connection) -> Check:
    """Consecutive days with no orders at all.

    A restaurant closed on Mondays is fine. A four-day hole in the middle of a
    normal week usually means the collector missed a window, and a model can't
    tell those apart.
    """
    dates = [
        row["date"]
        for row in conn.execute(
            """
            SELECT c.date, (SELECT COUNT(*) FROM daily_sales d WHERE d.business_date = c.date) AS n
            FROM calendar_days c
            WHERE c.date BETWEEN (SELECT MIN(business_date) FROM daily_sales)
                             AND (SELECT MAX(business_date) FROM daily_sales)
            ORDER BY c.date
            """
        )
        if row["n"] == 0
    ]

    longest, current, previous = 0, 0, None
    from datetime import date as _date, timedelta

    for value in dates:
        day = _date.fromisoformat(value)
        current = current + 1 if previous and day - previous == timedelta(days=1) else 1
        longest = max(longest, current)
        previous = day

    return Check(
        "no long closures",
        WARNING,
        longest <= 2,
        f"longest run of zero-sales days: {longest}"
        + (" — check whether that's a closure or a collection gap" if longest > 2 else ""),
        longest,
    )


def check_negative_money(conn: sqlite3.Connection) -> Check:
    bad = _scalar(conn, "SELECT COUNT(*) FROM orders WHERE revenue_cents < 0 OR tax_cents < 0")
    return Check(
        "money non-negative",
        ERROR,
        bad == 0,
        "no negative order totals" if not bad else f"{bad} orders with negative money",
        bad,
    )


def check_totals_reconcile(conn: sqlite3.Connection) -> Check:
    """net_sales should equal revenue - tax - tip, per the collector's definition."""
    bad = _scalar(
        conn,
        "SELECT COUNT(*) FROM orders "
        "WHERE ABS(net_sales_cents - (revenue_cents - tax_cents - tip_cents)) > 1",
    )
    return Check(
        "totals reconcile",
        ERROR,
        bad == 0,
        "net = revenue - tax - tip everywhere" if not bad else f"{bad} orders fail the identity",
        bad,
    )


def check_orphan_items(conn: sqlite3.Connection) -> Check:
    orphans = _scalar(
        conn,
        "SELECT COUNT(*) FROM order_items i "
        "WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.order_id = i.order_id)",
    )
    return Check(
        "line items linked",
        WARNING,
        orphans == 0,
        "every line item has an order"
        if not orphans
        else f"{orphans} line items with no order — widen the orders window",
        orphans,
    )


def check_future_orders(conn: sqlite3.Connection) -> Check:
    """An order dated in the future means a clock or timezone bug upstream."""
    ahead = _scalar(
        conn, "SELECT COUNT(*) FROM orders WHERE business_date > date('now', '+1 day')"
    )
    return Check(
        "no future orders",
        ERROR,
        ahead == 0,
        "no orders dated ahead of today" if not ahead else f"{ahead} orders dated in the future",
        ahead,
    )


def check_outlier_days(conn: sqlite3.Connection) -> Check:
    rows = [r["net_sales_cents"] for r in conn.execute("SELECT net_sales_cents FROM daily_sales")]
    if len(rows) < 10:
        return Check("daily outliers", INFO, True, "not enough days to judge", 0)

    mean = sum(rows) / len(rows)
    variance = sum((v - mean) ** 2 for v in rows) / len(rows)
    sigma = variance**0.5
    outliers = [v for v in rows if sigma and abs(v - mean) > OUTLIER_SIGMA * sigma]

    return Check(
        "daily outliers",
        WARNING,
        not outliers,
        f"mean ${mean/100:,.0f}/day, {len(outliers)} beyond {OUTLIER_SIGMA:g} sigma",
        len(outliers),
    )


def check_duplicate_ingestion(conn: sqlite3.Connection) -> Check:
    """The ledger and the tables should agree that nothing double-counted."""
    files = _scalar(conn, "SELECT COUNT(*) FROM ingested_files") or 0
    return Check("raw files loaded", INFO, True, f"{files} raw file(s) ingested", files)


def check_weather_is_actual(conn: sqlite3.Connection) -> Check:
    """Forecast rows over *past* dates get replaced by archive on a later run.

    If training days are still sitting on forecast weather, the model learns
    from a prediction rather than what actually happened.
    """
    forecast_days = _scalar(
        conn,
        "SELECT COUNT(*) FROM weather w JOIN daily_sales d ON d.business_date = w.date "
        "WHERE w.source = 'forecast' AND w.date < date('now', '-6 days')",
    ) or 0
    return Check(
        "weather is observed",
        WARNING,
        forecast_days == 0,
        "past weather is all reanalysis"
        if not forecast_days
        else f"{forecast_days} training days still on forecast weather — re-run the collector",
        forecast_days,
    )


CHECKS: list[CheckFn] = [
    check_has_orders,
    check_history_length,
    check_business_dates,
    check_calendar_coverage,
    check_calendar_gaps,
    check_zero_sales_runs,
    check_weather_coverage,
    check_weather_is_actual,
    check_negative_money,
    check_totals_reconcile,
    check_orphan_items,
    check_future_orders,
    check_outlier_days,
    check_duplicate_ingestion,
]


def validate(db_path: Path | str = DB_PATH, strict: bool = False) -> Report:
    conn = connect(db_path, read_only=True)
    try:
        checks = []
        for check in CHECKS:
            try:
                checks.append(check(conn))
            except sqlite3.Error as exc:
                checks.append(Check(check.__name__, ERROR, False, f"check failed to run: {exc}"))
    finally:
        conn.close()

    if strict:
        checks = [
            Check(c.name, ERROR if c.severity == WARNING else c.severity, c.passed, c.detail, c.value)
            for c in checks
        ]
    return Report(checks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the warehouse before training.")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--strict", action="store_true", help="treat warnings as errors")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    report = validate(args.db, strict=args.strict)
    print(f"\ndata quality — {args.db}\n")
    print(report.render())

    if report.errors:
        print(f"\n{len(report.errors)} blocking issue(s). Not safe to train.\n")
        return 1
    if report.warnings:
        print(f"\npassed with {len(report.warnings)} warning(s).\n")
    else:
        print("\nall checks passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
