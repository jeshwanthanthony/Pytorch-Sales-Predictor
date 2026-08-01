"""The read side: the queries training and inference will actually run.

This is the boundary the next phase sits on. Feature engineering imports
`feature_frame()` and gets a DataFrame; it never writes SQL and never touches
the raw JSONL.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from .db import DB_PATH, connect

# the columns a model can train on, all numeric and all backwards looking
FEATURE_COLUMNS = [
    # calendar
    "day_of_week", "month", "day_of_month", "week_of_year", "quarter",
    "is_weekend", "is_weekend_night", "is_holiday", "is_holiday_eve",
    "is_day_after_holiday", "is_observance", "is_school_break",
    "is_month_start", "is_month_end", "is_payday_window",
    # weather
    "temp_max_f", "temp_min_f", "temp_mean_f", "feels_like_max_f",
    "precipitation_in", "snowfall_in", "precipitation_hours", "wind_max_mph",
    "humidity_mean", "is_rainy", "is_snowy", "is_stormy",
    # local context
    "event_count", "event_attendance", "sports_events",
    "promotion_active", "promotion_count", "promotion_spend_usd", "max_percent_off",
    "has_facebook_ads", "has_delivery_promo",
    # history
    "sales_lag_1_cents", "sales_lag_2_cents", "sales_lag_7_cents",
    "sales_lag_14_cents", "sales_avg_7_cents", "sales_avg_30_cents",
    "orders_lag_1", "customers_lag_1", "orders_avg_7",
]

TARGET_COLUMN = "target_sales_cents"


def _read(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def feature_frame(
    db_path: Path = DB_PATH,
    location_id: str | None = None,
    observed_only: bool = True,
    drop_warmup: bool = False,
) -> pd.DataFrame:
    """The training table: one row per business day, ready for tensors.

    `observed_only` drops future dates — those rows exist for inference, where
    the target is unknown by definition.

    `drop_warmup` drops the opening rows whose history features are still null.
    The 14-day lag means the first 14 days of any history have nothing to look
    back at; those NULLs are honest, but they become silent NaNs in a tensor.
    """
    conn = connect(db_path, read_only=True)
    try:
        clauses = []
        params: list[Any] = []
        if observed_only:
            clauses.append("is_observed = 1 AND target_sales_cents IS NOT NULL")
        if location_id:
            clauses.append("location_id = ?")
            params.append(location_id)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        frame = _read(
            conn,
            f"SELECT * FROM daily_forecast_features {where} ORDER BY location_id, business_date",
            tuple(params),
        )
    finally:
        conn.close()

    if drop_warmup and len(frame):
        history = [c for c in FEATURE_COLUMNS if "lag" in c or "avg" in c]
        frame = frame.dropna(subset=history).reset_index(drop=True)

    return frame


def prediction_rows(db_path: Path = DB_PATH, location_id: str | None = None) -> pd.DataFrame:
    """Future dates — the rows a trained model is asked about.

    Weather is present here as a forecast rather than an observation, which is
    exactly why the collector reaches past today.
    """
    conn = connect(db_path, read_only=True)
    try:
        where = "WHERE is_observed = 0"
        params: tuple[Any, ...] = ()
        if location_id:
            where += " AND location_id = ?"
            params = (location_id,)
        return _read(
            conn,
            f"SELECT * FROM daily_forecast_features {where} ORDER BY location_id, business_date",
            params,
        )
    finally:
        conn.close()


def daily_sales(db_path: Path = DB_PATH, limit: int | None = None) -> pd.DataFrame:
    conn = connect(db_path, read_only=True)
    try:
        sql = "SELECT * FROM daily_sales ORDER BY business_date DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return _read(conn, sql)
    finally:
        conn.close()


def top_items(db_path: Path = DB_PATH, days: int = 30, limit: int = 20) -> pd.DataFrame:
    """Best sellers over the recent window — the menu-demand model's target list."""
    conn = connect(db_path, read_only=True)
    try:
        return _read(
            conn,
            """
            SELECT item_name, category_name,
                   SUM(quantity)      AS quantity,
                   SUM(revenue_cents) AS revenue_cents,
                   COUNT(DISTINCT business_date) AS days_sold
            FROM daily_item_sales
            WHERE business_date >= date((SELECT MAX(business_date) FROM daily_item_sales), ?)
            GROUP BY item_name, category_name
            ORDER BY quantity DESC
            LIMIT ?
            """,
            (f"-{int(days)} days", limit),
        )
    finally:
        conn.close()


def data_quality(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Gaps and nulls that would quietly poison training.

    Worth running before every training job: a model can't tell the difference
    between "closed on Mondays" and "we forgot to collect Mondays".
    """
    conn = connect(db_path, read_only=True)
    try:
        return _read(
            conn,
            """
            SELECT
                'days in range'          AS metric, COUNT(*) AS value FROM calendar_days
                WHERE date BETWEEN (SELECT MIN(business_date) FROM daily_sales)
                              AND (SELECT MAX(business_date) FROM daily_sales)
            UNION ALL
            SELECT 'days with sales', COUNT(*) FROM daily_sales
            UNION ALL
            SELECT 'days missing weather', COUNT(*) FROM daily_sales d
                WHERE NOT EXISTS (SELECT 1 FROM weather w WHERE w.date = d.business_date)
            UNION ALL
            SELECT 'orders without business_date', COUNT(*) FROM orders WHERE business_date IS NULL
            UNION ALL
            SELECT 'order items without an order', COUNT(*) FROM order_items i
                WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.order_id = i.order_id)
            UNION ALL
            SELECT 'payments without an order', COUNT(*) FROM payments
                WHERE order_id IS NULL OR order_id NOT IN (SELECT order_id FROM orders)
            """,
        )
    finally:
        conn.close()
