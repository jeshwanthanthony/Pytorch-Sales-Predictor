"""Turn the daily summary into the exact arrays PyTorch will train on.

This is the step that can't be a SQL view, for one reason: **the scaler must be
fit on the training split alone**. Standardising over the whole dataset leaks
the test period's mean into training and quietly flatters every metric you go on
to report. So the split happens first, statistics come from train, and the same
numbers are saved for inference to reuse.

Everything else here is feature engineering SQL is simply clumsy at — cyclical
encodings so December sits next to January, distance to the nearest holiday,
weather interactions, and same-weekday history.

    python -m pipelines.build_features
    python -m pipelines.build_features --test-days 28 --val-days 28

Outputs:
    data/features/dataset.npz     X/y arrays per split, plus dates
    data/features/manifest.json   feature names, scaler stats, split boundaries
    features table in the database (unscaled, for inspection and the dashboard)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from database.db import DB_PATH, connect, transaction, upsert

log = logging.getLogger("features")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "features"
DATASET_FILE = OUTPUT_DIR / "dataset.npz"
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"

TARGET = "target_sales_cents"
SECONDARY_TARGETS = ["target_orders", "target_customers"]

# things you only know after the day is over, so they can never be inputs.
# listing them by hand is what makes the leakage check mean anything
SAME_DAY_OUTCOMES = {
    "order_count", "revenue_cents", "net_sales_cents", "discount_cents", "tax_cents",
    "tip_cents", "service_charge_cents", "avg_ticket_cents", "item_count",
    "distinct_items", "customer_count", "new_customer_count", "returning_customer_count",
    "lunch_orders", "dinner_orders", "late_orders", "lunch_sales_cents",
    "dinner_sales_cents", "late_sales_cents", "peak_hour", "pickup_orders",
    "delivery_orders", "dine_in_orders", "cash_cents", "card_cents", "gift_card_cents",
    "processing_fee_cents", "refund_count", "refund_cents", "refund_rate_bps",
    "labor_hours", "labor_cost_cents", "staff_count", "sales_per_labor_hour_cents",
    "labor_cost_ratio_bps", "is_closed",
}

# leave 0/1 flags alone, scaling them gains nothing and makes them harder to read
BINARY_PREFIXES = ("is_", "has_", "promotion_active")

SOURCE_SQL = """
SELECT
    -- Date and location come from the spine, not the summary: future rows have
    -- no summary, and a NULL key would strand them out of every lag.
    c.date        AS business_date,
    l.location_id AS location_id,
    s.is_closed,
    s.order_count, s.net_sales_cents, s.customer_count, s.avg_ticket_cents,
    s.item_count, s.labor_hours,

    c.day_of_week, c.month, c.day_of_month, c.day_of_year, c.week_of_year, c.quarter,
    c.is_weekend, c.is_weekend_night, c.is_holiday, c.is_holiday_eve,
    c.is_day_after_holiday, c.is_observance, c.is_school_break, c.is_month_start,
    c.is_month_end, c.is_payday_window,

    w.temp_max_f, w.temp_min_f, w.temp_mean_f, w.feels_like_max_f,
    w.precipitation_in, w.snowfall_in, w.precipitation_hours, w.wind_max_mph,
    w.humidity_mean, w.is_rainy, w.is_snowy, w.is_stormy,

    COALESCE(e.event_count, 0)          AS event_count,
    COALESCE(e.event_attendance, 0)     AS event_attendance,
    COALESCE(e.sports_events, 0)        AS sports_events,
    COALESCE(e.nearest_event_miles, 99) AS nearest_event_miles,

    COALESCE(p.promotion_count, 0)      AS promotion_count,
    COALESCE(p.promotion_spend_usd, 0)  AS promotion_spend_usd,
    COALESCE(p.max_percent_off, 0)      AS max_percent_off,
    COALESCE(p.has_facebook_ads, 0)     AS has_facebook_ads,
    COALESCE(p.has_delivery_promo, 0)   AS has_delivery_promo
FROM calendar_days c
CROSS JOIN locations l
LEFT JOIN daily_summary s     ON s.business_date = c.date AND s.location_id = l.location_id
LEFT JOIN weather w           ON w.date = c.date
LEFT JOIN daily_events e      ON e.date = c.date
LEFT JOIN daily_promotions p  ON p.date = c.date
WHERE c.date >= (SELECT MIN(business_date) FROM daily_summary)
  AND c.date <= date((SELECT MAX(business_date) FROM daily_summary), :horizon)
  AND EXISTS (
      SELECT 1 FROM orders o
      WHERE o.location_id = l.location_id
        AND o.state <> 'CANCELED'
        AND o.net_sales_cents > 0
  )
ORDER BY l.location_id, c.date
"""


class LeakageError(RuntimeError):
    """Raised when a feature column would tell the model the answer."""


def load_source(db_path: Path | str, horizon_days: int) -> pd.DataFrame:
    conn = connect(db_path, read_only=True)
    try:
        frame = pd.read_sql_query(
            SOURCE_SQL, conn, params={"horizon": f"+{int(horizon_days)} days"}
        )
    finally:
        conn.close()

    if frame.empty:
        raise RuntimeError("no rows — run `python -m pipelines.build_daily_summary` first")

    # rows past the last trading day have no summary, those are the future
    frame["location_id"] = frame["location_id"].astype(str)
    frame["is_future"] = frame["net_sales_cents"].isna().astype(int)
    frame["is_closed"] = frame["is_closed"].fillna(0).astype(int)
    return frame


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    frame[TARGET] = frame["net_sales_cents"]
    frame["target_orders"] = frame["order_count"]
    frame["target_customers"] = frame["customer_count"]
    return frame


def add_cyclical(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode wraparound so Sunday is adjacent to Monday, December to January.

    A raw day_of_week of 7 vs 1 looks like a distance of 6 to a network; as a
    point on a circle it's a distance of 1, which is what it actually is.
    """
    two_pi = 2 * np.pi
    frame["dow_sin"] = np.sin(two_pi * frame["day_of_week"] / 7)
    frame["dow_cos"] = np.cos(two_pi * frame["day_of_week"] / 7)
    frame["month_sin"] = np.sin(two_pi * frame["month"] / 12)
    frame["month_cos"] = np.cos(two_pi * frame["month"] / 12)
    frame["doy_sin"] = np.sin(two_pi * frame["day_of_year"] / 365.25)
    frame["doy_cos"] = np.cos(two_pi * frame["day_of_year"] / 365.25)
    return frame


def add_holiday_distance(frame: pd.DataFrame) -> pd.DataFrame:
    """Days to the next holiday and since the last one.

    The week before Thanksgiving doesn't trade like an ordinary week, and a
    binary is_holiday can't express that.
    """
    dates = pd.to_datetime(frame["business_date"])
    holidays = dates[frame["is_holiday"] == 1].sort_values().unique()

    if len(holidays) == 0:
        frame["days_to_holiday"] = 99
        frame["days_since_holiday"] = 99
        return frame

    holiday_values = np.array(holidays, dtype="datetime64[ns]")
    day_values = dates.to_numpy(dtype="datetime64[ns]")

    after = np.searchsorted(holiday_values, day_values, side="left")
    before = after - 1

    to_next = np.where(
        after < len(holiday_values),
        (holiday_values[np.clip(after, 0, len(holiday_values) - 1)] - day_values)
        / np.timedelta64(1, "D"),
        99,
    )
    since_last = np.where(
        before >= 0,
        (day_values - holiday_values[np.clip(before, 0, len(holiday_values) - 1)])
        / np.timedelta64(1, "D"),
        99,
    )

    frame["days_to_holiday"] = np.clip(to_next, 0, 99)
    frame["days_since_holiday"] = np.clip(since_last, 0, 99)
    return frame


def add_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Lags and rolling statistics, per location, strictly backwards-looking.

    Every window is applied to an already-shifted series, so today's value can
    never appear in today's own feature.
    """
    grouped = frame.groupby("location_id", sort=False)

    for lag in (1, 2, 3, 7, 14, 28):
        frame[f"sales_lag_{lag}"] = grouped[TARGET].shift(lag)

    frame["orders_lag_1"] = grouped["target_orders"].shift(1)
    frame["orders_lag_7"] = grouped["target_orders"].shift(7)
    frame["customers_lag_1"] = grouped["target_customers"].shift(1)
    frame["customers_lag_7"] = grouped["target_customers"].shift(7)
    frame["avg_ticket_lag_7"] = grouped["avg_ticket_cents"].shift(7)
    frame["labor_hours_lag_7"] = grouped["labor_hours"].shift(7)

    shifted = grouped[TARGET].shift(1)
    by_location = shifted.groupby(frame["location_id"], sort=False)
    frame["sales_roll_mean_7"] = by_location.transform(lambda s: s.rolling(7, min_periods=7).mean())
    frame["sales_roll_mean_28"] = by_location.transform(lambda s: s.rolling(28, min_periods=28).mean())
    frame["sales_roll_std_7"] = by_location.transform(lambda s: s.rolling(7, min_periods=7).std())
    frame["sales_roll_max_7"] = by_location.transform(lambda s: s.rolling(7, min_periods=7).max())

    # same weekday over the past few weeks, usually the strongest single signal
    # and a plain 7 day mean hides it
    frame["same_dow_mean_4"] = (
        frame[["sales_lag_7", "sales_lag_14", "sales_lag_28"]].mean(axis=1, skipna=False)
    )

    # Momentum: is the last week running above or below the last month?
    frame["momentum_7_28"] = frame["sales_roll_mean_7"] / frame["sales_roll_mean_28"].replace(0, np.nan)
    frame["momentum_7_28"] = frame["momentum_7_28"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    return frame


def add_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """A few crosses worth stating outright rather than hoping the net finds them."""
    frame["rain_x_weekend"] = frame["is_rainy"].fillna(0) * frame["is_weekend"]
    frame["promotion_active"] = (frame["promotion_count"] > 0).astype(int)
    frame["promo_x_weekend"] = frame["promotion_active"] * frame["is_weekend"]

    # How unusual is today's temperature for this time of year?
    grouped = frame.groupby("location_id", sort=False)["temp_max_f"]
    normal = grouped.transform(lambda s: s.shift(1).rolling(30, min_periods=10).mean())
    frame["temp_vs_normal_f"] = (frame["temp_max_f"] - normal).fillna(0)

    frame["days_since_start"] = (
        pd.to_datetime(frame["business_date"]) - pd.to_datetime(frame["business_date"]).min()
    ).dt.days
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Every column that is a legitimate input, in stable order."""
    excluded = SAME_DAY_OUTCOMES | {
        "business_date", "location_id", "is_future", TARGET, *SECONDARY_TARGETS,
    }
    return [c for c in frame.columns if c not in excluded]


def assert_no_leakage(columns: list[str]) -> None:
    leaked = sorted(set(columns) & SAME_DAY_OUTCOMES)
    if leaked:
        raise LeakageError(
            f"same-day outcome(s) present as model inputs: {', '.join(leaked)}. "
            "These are only knowable after the day ends."
        )


def auto_split_sizes(observed_days: int, val_days: int, test_days: int) -> tuple[int, int]:
    """Size the holdouts to the history available.

    A fixed 28/28 is right for two years of data and absurd for three months —
    it would leave almost nothing to train on. 0 means "choose for me": 15% each,
    floored at 7 days and capped at 28.
    """
    if val_days and test_days:
        return val_days, test_days
    auto = max(7, min(28, int(observed_days * 0.15)))
    return (val_days or auto), (test_days or auto)


def split_by_time(frame: pd.DataFrame, val_days: int, test_days: int) -> pd.DataFrame:
    """Split by date. Never random — a random split lets the model see the future."""
    observed = frame[frame["is_future"] == 0]
    dates = sorted(observed["business_date"].unique())

    if len(dates) - val_days - test_days < 30:
        log.warning(
            "only %d observed days; val=%d test=%d leaves %d for training",
            len(dates), val_days, test_days, len(dates) - val_days - test_days,
        )

    test_start = dates[-test_days] if test_days and len(dates) > test_days else None
    val_start = (
        dates[-(test_days + val_days)] if val_days and len(dates) > test_days + val_days else None
    )

    def label(row: pd.Series) -> str:
        if row["is_future"]:
            return "future"
        date = row["business_date"]
        if test_start and date >= test_start:
            return "test"
        if val_start and date >= val_start:
            return "val"
        return "train"

    frame["split"] = frame.apply(label, axis=1)
    return frame


def fit_scaler(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    """Mean and standard deviation from the training split only.

    Saved as plain JSON rather than a pickled sklearn object so inference has no
    version coupling to whatever fitted it.
    """
    train = frame[frame["split"] == "train"]
    if train.empty:
        raise RuntimeError("no training rows — widen the history or shrink --val-days/--test-days")

    stats: dict[str, dict[str, float]] = {}
    for column in columns:
        if column.startswith(BINARY_PREFIXES):
            continue
        series = train[column].astype(float)
        std = float(series.std())
        stats[column] = {"mean": float(series.mean()), "std": std if std > 1e-9 else 1.0}
    return stats


def apply_scaler(frame: pd.DataFrame, stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    scaled = frame.copy()
    for column, values in stats.items():
        scaled[column] = (scaled[column].astype(float) - values["mean"]) / values["std"]
    return scaled


def build(
    db_path: Path | str = DB_PATH,
    val_days: int = 0,
    test_days: int = 0,
    horizon_days: int = 14,
    write_db: bool = True,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    frame = load_source(db_path, horizon_days)
    frame = add_targets(frame)
    frame = add_cyclical(frame)
    frame = add_holiday_distance(frame)
    frame = add_history(frame)
    frame = add_interactions(frame)

    columns = feature_columns(frame)
    assert_no_leakage(columns)

    # the first rows have no 28 day history yet, and future rows past tomorrow
    # have no lag_1. drop them, a made up lag is a made up fact about the past
    before = len(frame)
    frame = frame.dropna(subset=columns).reset_index(drop=True)
    log.info("dropped %d rows with incomplete history, %d remain", before - len(frame), len(frame))

    observed = frame[frame["is_future"] == 0]
    if observed.empty:
        raise RuntimeError("no observed rows survived — not enough history yet")

    val_days, test_days = auto_split_sizes(
        observed["business_date"].nunique(), val_days, test_days
    )
    frame = split_by_time(frame, val_days, test_days)
    stats = fit_scaler(frame, columns)
    scaled = apply_scaler(frame, stats)

    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    dataset_file = output_dir / DATASET_FILE.name
    manifest_file = output_dir / MANIFEST_FILE.name
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"feature_names": np.array(columns)}
    counts: dict[str, int] = {}
    spans: dict[str, list[str]] = {}

    for name in ("train", "val", "test", "future"):
        subset = scaled[scaled["split"] == name]
        counts[name] = len(subset)
        if subset.empty:
            continue
        spans[name] = [subset["business_date"].min(), subset["business_date"].max()]
        arrays[f"X_{name}"] = subset[columns].to_numpy(dtype=np.float32)
        arrays[f"dates_{name}"] = subset["business_date"].to_numpy(dtype=object).astype(str)
        if name != "future":
            arrays[f"y_{name}"] = subset[TARGET].to_numpy(dtype=np.float32).reshape(-1, 1)

    np.savez_compressed(dataset_file, **arrays)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "database": str(db_path),
        "target": TARGET,
        "feature_count": len(columns),
        "feature_names": columns,
        "binary_features": [c for c in columns if c.startswith(BINARY_PREFIXES)],
        "row_counts": counts,
        "date_spans": spans,
        "split_config": {"val_days": val_days, "test_days": test_days, "horizon_days": horizon_days},
        "scaler": stats,
        "notes": "Scaler fit on the train split only. Apply these exact stats at inference.",
        "dataset_file": str(dataset_file),
    }
    manifest_file.write_text(json.dumps(manifest, indent=2))

    if write_db:
        _write_features_table(db_path, frame, columns)

    log.info(
        "features: %d columns | train %d, val %d, test %d, future %d",
        len(columns), counts["train"], counts["val"], counts["test"], counts["future"],
    )
    return manifest


FEATURES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS features (
    business_date TEXT NOT NULL,
    location_id   TEXT NOT NULL,
    split         TEXT NOT NULL,
    target_sales_cents INTEGER,
    payload       TEXT NOT NULL,   -- JSON: the unscaled feature vector
    built_at      TEXT NOT NULL,
    PRIMARY KEY (business_date, location_id)
) STRICT;

CREATE INDEX IF NOT EXISTS ix_features_split ON features (split, business_date);
"""


def _write_features_table(db_path: Path | str, frame: pd.DataFrame, columns: list[str]) -> None:
    """Keep an unscaled, queryable copy in the database.

    The npz is what trains; this is what you look at when a prediction seems
    wrong and you want to know what the model was actually shown.
    """
    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn = connect(db_path)
    try:
        conn.executescript(FEATURES_TABLE_SQL)
        rows = (
            {
                "business_date": record["business_date"],
                "location_id": record["location_id"],
                "split": record["split"],
                "target_sales_cents": None
                if pd.isna(record[TARGET])
                else int(record[TARGET]),
                "payload": json.dumps(
                    {c: _jsonable(record[c]) for c in columns}, separators=(",", ":")
                ),
                "built_at": built_at,
            }
            for record in frame.to_dict("records")
        )
        with transaction(conn):
            upsert(
                conn,
                "features",
                ["business_date", "location_id", "split", "target_sales_cents", "payload", "built_at"],
                ["business_date", "location_id"],
                rows,
            )
    finally:
        conn.close()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 4)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def load_dataset(path: Path = DATASET_FILE) -> dict[str, np.ndarray]:
    """Read back what build() wrote — this is what training/ will call."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m pipelines.build_features`")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the model-ready feature set.")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--val-days", type=int, default=0, help="0 = size to the history")
    parser.add_argument("--test-days", type=int, default=0, help="0 = size to the history")
    parser.add_argument("--horizon-days", type=int, default=14, help="future rows to prepare")
    parser.add_argument("--no-db", action="store_true", help="skip writing the features table")
    parser.add_argument("--output-dir", type=Path, default=None, help="where to write dataset.npz")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    manifest = build(
        args.db,
        val_days=args.val_days,
        test_days=args.test_days,
        horizon_days=args.horizon_days,
        write_db=not args.no_db,
        output_dir=args.output_dir,
    )

    print(f"\n{manifest['feature_count']} features -> {manifest['dataset_file']}")
    for name, count in manifest["row_counts"].items():
        span = manifest["date_spans"].get(name)
        window = f"{span[0]} .. {span[1]}" if span else "-"
        print(f"  {name:<7} {count:>5} rows   {window}")
    print(f"\nmanifest: {Path(manifest['dataset_file']).parent / MANIFEST_FILE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
