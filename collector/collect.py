"""Collect everything for one restaurant into one folder.

collector/run.py is the command line version and always writes to data/raw.
This is the same work, but you tell it whose token to use and where to put the
files — which is what the web app needs, because every restaurant that connects
gets its own folder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .calendar_api import CalendarCollector
from .config import SiteConfig, SquareAuth
from .events import fetch_events, fetch_promotions
from .square_api import SquareCollector
from .storage import RawWriter, State, read_entity
from .weather_api import WeatherCollector, geocode

log = logging.getLogger("collect")

# how far ahead to collect weather and calendar, so tomorrow can be predicted
FORECAST_DAYS = 14


@dataclass
class CollectionResult:
    rows: dict[str, int] = field(default_factory=dict)
    site: SiteConfig | None = None
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def order_count(self) -> int:
        return self.rows.get("orders", 0)


def site_from_locations(locations: list[dict[str, Any]]) -> SiteConfig | None:
    """Work out where the restaurant is, for weather and holidays."""
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

    # square did not give coordinates, so look the address up
    attempts = [
        ", ".join(filter(None, [active.get("city"), active.get("state"), active.get("country")])),
        ", ".join(filter(None, [active.get("city"), active.get("country")])),
        active.get("postal_code"),
    ]
    for query in [q for q in attempts if q]:
        hit = geocode(query)
        if hit:
            log.info("geocoded %r -> %.4f, %.4f", query, hit["latitude"], hit["longitude"])
            return SiteConfig(
                latitude=hit["latitude"],
                longitude=hit["longitude"],
                timezone=hit["timezone"],
                country=active.get("country") or hit.get("country_code") or "US",
                subdivision=active.get("state"),
            )

    log.warning("could not work out where this restaurant is, skipping weather")
    return None


def collect_all(
    auth: SquareAuth,
    raw_dir: Path,
    state_file: Path,
    since: date,
    until: date | None = None,
    forecast_days: int = FORECAST_DAYS,
    on_progress: Callable[[str, int], None] | None = None,
    forecast_only: bool = False,
    resume_existing: bool = False,
) -> CollectionResult:
    """Pull everything for one account. One bad source never stops the rest."""
    until = until or date.today()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    state = State(state_file)
    result = CollectionResult()

    def existing_count(entity: str) -> int | None:
        if not resume_existing:
            return None
        directory = Path(raw_dir) / entity
        paths = sorted(directory.glob(f"{entity}-*.jsonl")) if directory.exists() else []
        if not paths:
            return None
        with paths[-1].open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def write(entity: str, records, watermark: str | None = None) -> int:
        try:
            cached = existing_count(entity)
            if cached is not None:
                result.rows[entity] = cached
                state.record_run(entity, cached)
                if on_progress:
                    on_progress(entity, cached)
                log.info("%-14s %6d rows (resumed existing file)", entity, cached)
                return cached

            seen = 0

            def reporting_records():
                nonlocal seen
                if on_progress:
                    on_progress(entity, 0)
                for record in records:
                    seen += 1
                    if on_progress and (seen == 1 or seen % 1_000 == 0):
                        on_progress(entity, seen)
                    yield record

            writer = RawWriter(entity, run_id, watermark, raw_dir)
            count = writer.write_all(reporting_records())
            state.record_run(entity, count)
            if writer.watermark:
                state.set(entity, writer.watermark)
            result.rows[entity] = count
            if on_progress:
                on_progress(entity, count)
            return count
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            log.error("%s failed: %s", entity, exc)
            result.failures[entity] = str(exc)
            result.rows[entity] = 0
            return 0

    start = datetime.combine(since, datetime.min.time()).replace(tzinfo=timezone.utc)
    end = datetime.combine(until, datetime.max.time()).replace(tzinfo=timezone.utc)

    with SquareCollector(auth) as square:
        locations = read_entity("locations", raw_dir) if existing_count("locations") is not None else []
        if not locations:
            try:
                locations = square.fetch_locations()
            except Exception as exc:  # noqa: BLE001
                result.failures["locations"] = str(exc)
        write("locations", iter(locations))

        # one download, two tables, line items only live on the raw payload
        raw_orders: list[dict[str, Any]] = []
        orders_cached = existing_count("orders") is not None
        items_cached = existing_count("order_items") is not None
        if not (orders_cached and items_cached):
            try:
                raw_orders = list(square.search_orders(start, end))
            except Exception as exc:  # noqa: BLE001
                result.failures["orders"] = str(exc)

        write("orders", (SquareCollector.normalize_order(o) for o in raw_orders), "updated_at")
        write("order_items", SquareCollector.fetch_order_items(raw_orders))
        write("payments", square.fetch_payments(start, end), "updated_at")
        write("refunds", square.fetch_refunds(start, end), "updated_at")
        if not forecast_only:
            write("customers", square.fetch_customers(), "updated_at")

            catalog_rows: list[dict[str, Any]] = []
            try:
                catalog_rows = list(square.fetch_catalog())
            except Exception as exc:  # noqa: BLE001
                result.failures["catalog"] = str(exc)
            write("catalog", iter(catalog_rows), "updated_at")

            variation_ids = [
                variation["id"]
                for item in catalog_rows
                for variation in item.get("variations") or []
                if variation.get("id")
            ]
            write("inventory", square.fetch_inventory(variation_ids))
            write("team_members", square.fetch_team_members(), "updated_at")
            write("shifts", square.fetch_shifts(start, end), "start_at")

        result.site = site_from_locations(locations)

    # the outside world: weather, holidays, events, promotions
    if result.site:
        future_end = until + timedelta(days=forecast_days)
        with WeatherCollector(result.site) as weather:
            write("weather", weather.fetch(since, future_end), "date")
        write("calendar", CalendarCollector(result.site).fetch(since, future_end), "date")
        write("events", fetch_events(since, future_end), "date")
        write("promotions", fetch_promotions(since, future_end), "date")

    state.save()
    return result
