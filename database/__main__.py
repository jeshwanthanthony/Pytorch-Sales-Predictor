"""CLI for the warehouse.

    python -m database init          create the schema
    python -m database load          load everything new from data/raw/
    python -m database load --force  re-read every raw file
    python -m database stats         row counts, date span, file size
    python -m database check         data-quality report
    python -m database features      preview the training table
"""

from __future__ import annotations

import argparse
import logging
import sys

from .db import DB_PATH, connect, init_schema
from .load import LOAD_ORDER, load_all, summarize
from .queries import FEATURE_COLUMNS, data_quality, daily_sales, feature_frame, top_items

log = logging.getLogger("database")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="database", description="Forecast warehouse.")
    parser.add_argument("command", choices=["init", "load", "stats", "check", "features", "sales", "items"])
    parser.add_argument("--db", default=str(DB_PATH), help="database file")
    parser.add_argument("--only", help=f"comma-separated entities: {', '.join(LOAD_ORDER)}")
    parser.add_argument("--force", action="store_true", help="re-load raw files already ingested")
    parser.add_argument("--cutoff-hour", type=int, default=4, help="business-day cutoff, local hour")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "init":
        conn = connect(args.db)
        init_schema(conn)
        conn.close()
        log.info("schema ready at %s", args.db)
        return 0

    if args.command == "load":
        only = args.only.split(",") if args.only else None
        results = load_all(args.db, only=only, force=args.force, cutoff_hour=args.cutoff_hour)
        total = sum(results.values())
        log.info("loaded %d rows across %d entities", total, len([r for r in results.values() if r]))
        _print_stats(args.db)
        return 0

    if args.command == "stats":
        _print_stats(args.db)
        return 0

    if args.command == "check":
        print(data_quality(args.db).to_string(index=False))
        return 0

    if args.command == "sales":
        frame = daily_sales(args.db, limit=args.limit)
        print(frame.to_string(index=False) if len(frame) else "no sales yet")
        return 0

    if args.command == "items":
        frame = top_items(args.db, limit=args.limit)
        print(frame.to_string(index=False) if len(frame) else "no item sales yet")
        return 0

    if args.command == "features":
        frame = feature_frame(args.db)
        if not len(frame):
            print("no observed days yet — load Square orders first")
            return 0
        print(f"{len(frame)} rows x {len(frame.columns)} columns "
              f"({len(FEATURE_COLUMNS)} model features)")
        preview = ["business_date", "day_name", "is_rainy", "promotion_active",
                   "sales_lag_1_cents", "sales_lag_7_cents", "sales_avg_7_cents",
                   "target_sales_cents"]
        print(frame[[c for c in preview if c in frame.columns]].tail(args.limit).to_string(index=False))
        return 0

    return 1


def _print_stats(db_path: str) -> None:
    summary = summarize(db_path)
    print(f"\n{db_path}  ({summary['size_mb']} MB)")
    print(f"sales span: {summary['first_sales_date']} .. {summary['last_sales_date']} "
          f"({summary['days_with_sales']} days)\n")
    width = max(len(name) for name in summary["counts"]) if summary["counts"] else 10
    for name, count in summary["counts"].items():
        if count:
            print(f"  {name:<{width}}  {count:>8,}")
    empty = [n for n, c in summary["counts"].items() if not c]
    if empty:
        print(f"\n  empty: {', '.join(empty)}")


if __name__ == "__main__":
    sys.exit(main())
