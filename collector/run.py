"""CLI entry point: pull everything, land it in data/raw/, update the watermarks.

    python -m collector.run --since 2024-01-01     # first full backfill
    python -m collector.run                        # incremental, since last run
    python -m collector.run --only weather,calendar
    python -m collector.run --check                # what can this token read?

Each source is independent: one failing pull is logged and the rest continue,
because a missing scope shouldn't cost you the other seven entities.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from .calendar_api import CalendarCollector
from .config import REQUIRED_SCOPES, ConfigError, SiteConfig, load_square_auth
from .events import fetch_events, fetch_promotions
from .square_api import SquareCollector, probe_scopes
from .storage import RawWriter, State
from .weather_api import WeatherCollector, geocode

log = logging.getLogger("collector")

SQUARE_ENTITIES = [
    "locations",
    "orders",
    "order_items",
    "payments",
    "refunds",
    "customers",
    "catalog",
    "inventory",
    "team_members",
    "shifts",
]
EXTERNAL_ENTITIES = ["weather", "calendar", "events", "promotions"]
ALL_ENTITIES = SQUARE_ENTITIES + EXTERNAL_ENTITIES

# how far back to go the very first time, when we have no watermark yet
DEFAULT_BACKFILL_DAYS = 730
# go back a bit further than the watermark, square can back-date an update
# and duplicates cost nothing because the database dedupes on the id
OVERLAP = timedelta(hours=6)


def resolve_site(square: SquareCollector | None) -> SiteConfig | None:
    """Where the restaurant is: .env first, then the Square location, then geocode."""
    site = SiteConfig.from_env()
    if site:
        return site
    if square is None:
        return None

    try:
        locations = square.fetch_locations()
    except Exception as exc:  # noqa: BLE001 - any failure here is non-fatal
        log.warning("could not read locations for site config: %s", exc)
        return None

    active = next((loc for loc in locations if loc.get("status") == "ACTIVE"), None) or (
        locations[0] if locations else None
    )
    if not active:
        return None

    if active.get("latitude") and active.get("longitude"):
        return SiteConfig(
            latitude=float(active["latitude"]),
            longitude=float(active["longitude"]),
            timezone=active.get("timezone") or "America/New_York",
            country=active.get("country") or "US",
            subdivision=active.get("state"),
        )

    # try the most specific query first, then fall back. open-meteo's geocoder
    # searches place names, so a bare postal code usually finds nothing.
    attempts = [
        ", ".join(filter(None, [active.get("city"), active.get("state"), active.get("country")])),
        ", ".join(filter(None, [active.get("city"), active.get("country")])),
        active.get("postal_code"),
    ]

    hit = None
    for query in [q for q in attempts if q]:
        hit = geocode(query)
        if hit:
            break

    if not hit:
        log.warning(
            "could not geocode any of %r — set SITE_LATITUDE/SITE_LONGITUDE in .env",
            [q for q in attempts if q],
        )
        return None

    log.info("geocoded %r -> %.4f, %.4f", query, hit["latitude"], hit["longitude"])
    return SiteConfig(
        latitude=hit["latitude"],
        longitude=hit["longitude"],
        timezone=hit["timezone"],
        country=active.get("country") or hit.get("country_code") or "US",
        subdivision=active.get("state"),
    )


def run_entity(
    name: str,
    state: State,
    run_id: str,
    produce: Callable[[], Any],
    watermark_field: str | None = None,
    dry_run: bool = False,
) -> int:
    """Write one entity's rows, recording how it went. Never raises."""
    try:
        records = produce()
        if dry_run:
            rows = sum(1 for _ in records)
            log.info("%-14s %6d rows (dry run, nothing written)", name, rows)
            return rows

        writer = RawWriter(name, run_id=run_id, watermark_field=watermark_field)
        count = writer.write_all(records)
        state.record_run(name, count)
        if writer.watermark:
            state.set(name, writer.watermark)
        return count
    except Exception as exc:  # noqa: BLE001 - one bad source must not stop the run
        log.error("%-14s FAILED: %s", name, exc)
        state.record_run(name, -1)
        return 0


def window_for(state: State, entity: str, since: date | None, until: date) -> tuple[datetime, datetime]:
    """The time range to ask for, honouring --since or the stored watermark."""
    end = datetime.combine(until, datetime.max.time()).replace(tzinfo=timezone.utc)

    if since:
        return datetime.combine(since, datetime.min.time()).replace(tzinfo=timezone.utc), end

    watermark = state.get(entity)
    if watermark:
        try:
            start = datetime.fromisoformat(str(watermark).replace("Z", "+00:00")) - OVERLAP
            return start, end
        except ValueError:
            log.warning("unparseable watermark for %s (%r) — falling back", entity, watermark)

    return end - timedelta(days=DEFAULT_BACKFILL_DAYS), end


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect raw data for the forecast model.")
    parser.add_argument("--since", type=date.fromisoformat, help="YYYY-MM-DD; overrides watermarks")
    parser.add_argument("--until", type=date.fromisoformat, default=date.today(), help="YYYY-MM-DD")
    parser.add_argument("--only", help=f"comma-separated subset of: {', '.join(ALL_ENTITIES)}")
    parser.add_argument("--forecast-days", type=int, default=14, help="days of future weather/calendar")
    parser.add_argument("--check", action="store_true", help="probe which pulls the token allows, then exit")
    parser.add_argument("--dry-run", action="store_true", help="count rows without writing files")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # httpx logs every request at INFO, which drowns out the run summary.
    logging.getLogger("httpx").setLevel(logging.DEBUG if args.verbose else logging.WARNING)

    wanted = set(args.only.split(",")) if args.only else set(ALL_ENTITIES)
    unknown = wanted - set(ALL_ENTITIES)
    if unknown:
        parser.error(f"unknown entities: {', '.join(sorted(unknown))}")

    state = State()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # -- Square -------------------------------------------------------------
    square: SquareCollector | None = None
    if wanted & set(SQUARE_ENTITIES) or args.check:
        try:
            square = SquareCollector(load_square_auth())
        except ConfigError as exc:
            log.error("%s", exc)
            if args.check or not (wanted & set(EXTERNAL_ENTITIES)):
                return 1
            log.warning("continuing with external sources only")

    if args.check:
        if square is None:
            return 1
        log.info("probing scopes (expects: %s)", ", ".join(REQUIRED_SCOPES))
        for entity, allowed in probe_scopes(square).items():
            log.info("  %-14s %s", entity, "ok" if allowed else "DENIED - missing scope")
        return 0

    # Resolved inside the Square session below, since it may need a live lookup.
    site: SiteConfig | None = SiteConfig.from_env()

    if square:
        with square:
            if site is None and wanted & set(EXTERNAL_ENTITIES):
                site = resolve_site(square)

            if "locations" in wanted:
                run_entity("locations", state, run_id, square.fetch_locations, dry_run=args.dry_run)

            if "orders" in wanted or "order_items" in wanted:
                start, end = window_for(state, "orders", args.since, args.until)
                log.info("orders window: %s .. %s", start.date(), end.date())
                # one download, two tables, line items only exist on the raw payload
                raw_orders = list(square.search_orders(start, end))
                log.info("fetched %d raw orders", len(raw_orders))

                if "orders" in wanted:
                    run_entity(
                        "orders",
                        state,
                        run_id,
                        lambda: (SquareCollector.normalize_order(o) for o in raw_orders),
                        watermark_field="updated_at",
                        dry_run=args.dry_run,
                    )
                if "order_items" in wanted:
                    run_entity(
                        "order_items",
                        state,
                        run_id,
                        lambda: square.fetch_order_items(raw_orders),
                        dry_run=args.dry_run,
                    )

            if "payments" in wanted:
                start, end = window_for(state, "payments", args.since, args.until)
                run_entity(
                    "payments", state, run_id,
                    lambda: square.fetch_payments(start, end),
                    watermark_field="updated_at", dry_run=args.dry_run,
                )

            if "refunds" in wanted:
                start, end = window_for(state, "refunds", args.since, args.until)
                run_entity(
                    "refunds", state, run_id,
                    lambda: square.fetch_refunds(start, end),
                    watermark_field="updated_at", dry_run=args.dry_run,
                )

            if "customers" in wanted:
                run_entity(
                    "customers", state, run_id, square.fetch_customers,
                    watermark_field="updated_at", dry_run=args.dry_run,
                )

            catalog_rows: list[dict[str, Any]] = []
            if "catalog" in wanted or "inventory" in wanted:
                catalog_rows = list(square.fetch_catalog())
                if "catalog" in wanted:
                    run_entity(
                        "catalog", state, run_id, lambda: iter(catalog_rows),
                        watermark_field="updated_at", dry_run=args.dry_run,
                    )

            if "inventory" in wanted:
                # stock is counted per variation, not per item
                variation_ids = [
                    variation["id"]
                    for item in catalog_rows
                    for variation in item.get("variations") or []
                    if variation.get("id")
                ]
                run_entity(
                    "inventory", state, run_id,
                    lambda: square.fetch_inventory(variation_ids), dry_run=args.dry_run,
                )

            if "team_members" in wanted:
                run_entity(
                    "team_members", state, run_id, square.fetch_team_members,
                    watermark_field="updated_at", dry_run=args.dry_run,
                )

            if "shifts" in wanted:
                start, end = window_for(state, "shifts", args.since, args.until)
                run_entity(
                    "shifts", state, run_id,
                    lambda: square.fetch_shifts(start, end),
                    watermark_field="start_at", dry_run=args.dry_run,
                )

    # -- external sources ---------------------------------------------------
    if wanted & set(EXTERNAL_ENTITIES):
        if site is None:
            log.error(
                "no site location — set SITE_LATITUDE / SITE_LONGITUDE in .env "
                "(skipping weather and calendar)"
            )
        else:
            log.info(
                "site: %.4f, %.4f  %s  %s/%s",
                site.latitude, site.longitude, site.timezone, site.country, site.subdivision or "-",
            )
            # go past today, predicting tomorrow needs tomorrow's weather and calendar
            history_start = args.since or (args.until - timedelta(days=DEFAULT_BACKFILL_DAYS))
            future_end = args.until + timedelta(days=args.forecast_days)

            if "weather" in wanted:
                with WeatherCollector(site) as weather:
                    run_entity(
                        "weather", state, run_id,
                        lambda: weather.fetch(history_start, future_end),
                        watermark_field="date", dry_run=args.dry_run,
                    )

            if "calendar" in wanted:
                calendar = CalendarCollector(site)
                run_entity(
                    "calendar", state, run_id,
                    lambda: calendar.fetch(history_start, future_end),
                    watermark_field="date", dry_run=args.dry_run,
                )

            if "events" in wanted:
                run_entity(
                    "events", state, run_id,
                    lambda: fetch_events(history_start, future_end),
                    watermark_field="date", dry_run=args.dry_run,
                )

            if "promotions" in wanted:
                run_entity(
                    "promotions", state, run_id,
                    lambda: fetch_promotions(history_start, future_end),
                    watermark_field="date", dry_run=args.dry_run,
                )

    if not args.dry_run:
        state.save()
        log.info("run %s complete; state saved", run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
