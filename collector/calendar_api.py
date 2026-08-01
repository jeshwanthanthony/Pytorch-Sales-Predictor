"""Calendar context for every date: holidays, day-of-week, school terms, paydays.

Named calendar_api rather than holidays.py on purpose — a module called
holidays.py inside this package would be a confusing neighbour to the `holidays`
PyPI package it imports.

This is the one source that needs no network at all, so it can generate rows for
future dates. That matters: predicting next Friday needs next Friday's calendar.
"""

from __future__ import annotations

import csv
import logging
from datetime import date, timedelta
from typing import Any, Iterator

import holidays

from .config import REFERENCE_DIR, SiteConfig

log = logging.getLogger(__name__)

# Holidays that move restaurant traffic even though they aren't days off.
OBSERVANCES = {
    (2, 14): "Valentines Day",
    (3, 17): "St Patricks Day",
    (5, 5): "Cinco de Mayo",
    (10, 31): "Halloween",
    (12, 24): "Christmas Eve",
    (12, 31): "New Years Eve",
}


def _load_school_terms() -> list[tuple[date, date, str]]:
    """Break periods from reference/school_calendar.csv (start,end,label)."""
    path = REFERENCE_DIR / "school_calendar.csv"
    if not path.exists():
        return []
    terms = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                terms.append(
                    (
                        date.fromisoformat(row["start_date"]),
                        date.fromisoformat(row["end_date"]),
                        row.get("label", "break"),
                    )
                )
            except (KeyError, ValueError):
                log.warning("skipping malformed school_calendar row: %s", row)
    return terms


class CalendarCollector:
    def __init__(self, site: SiteConfig):
        self.site = site
        self._school_terms = _load_school_terms()
        self._cache: dict[int, Any] = {}

    def _holidays_for(self, year: int) -> Any:
        if year not in self._cache:
            try:
                self._cache[year] = holidays.country_holidays(
                    self.site.country, subdiv=self.site.subdivision, years=year
                )
            except NotImplementedError:
                # unknown state, just use the national holidays
                self._cache[year] = holidays.country_holidays(self.site.country, years=year)
        return self._cache[year]

    def fetch(self, start: date, end: date) -> Iterator[dict[str, Any]]:
        """One row per date in [start, end]. Works for past and future dates."""
        day = start
        while day <= end:
            yield self.describe(day)
            day += timedelta(days=1)

    def describe(self, day: date) -> dict[str, Any]:
        calendar = self._holidays_for(day.year)
        holiday_name = calendar.get(day)

        # a holiday changes trade the night before and the day after too
        eve_of = calendar.get(day + timedelta(days=1))
        day_after = calendar.get(day - timedelta(days=1))
        observance = OBSERVANCES.get((day.month, day.day))

        school_break = next(
            (label for s, e, label in self._school_terms if s <= day <= e),
            None,
        )

        return {
            "date": day.isoformat(),
            "day_of_week": day.isoweekday(),  # 1 = Monday
            "day_name": day.strftime("%A"),
            "month": day.month,
            "month_name": day.strftime("%B"),
            "year": day.year,
            "day_of_month": day.day,
            "day_of_year": day.timetuple().tm_yday,
            "week_of_year": day.isocalendar().week,
            "quarter": (day.month - 1) // 3 + 1,
            "is_weekend": day.isoweekday() >= 6,
            # friday night trades like a weekend even though friday is a workday
            "is_weekend_night": day.isoweekday() in (5, 6),
            "is_holiday": holiday_name is not None,
            "holiday_name": holiday_name,
            "is_holiday_eve": eve_of is not None,
            "is_day_after_holiday": day_after is not None,
            "observance": observance,
            "is_observance": observance is not None,
            "school_break": school_break,
            "is_school_break": school_break is not None,
            "is_month_start": day.day == 1,
            "is_month_end": (day + timedelta(days=1)).day == 1,
            # most people get paid on the 1st and 15th and spend right after
            "is_payday_window": day.day in (1, 2, 15, 16),
        }
