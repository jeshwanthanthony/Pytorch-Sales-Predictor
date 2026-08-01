"""Daily weather from Open-Meteo — no API key, no rate limit worth worrying about.

Two endpoints, because they cover different windows:
  * archive  — reanalysis, authoritative, but lags ~5 days behind today
  * forecast — the last ~92 days plus 16 days ahead

History pulls stitch them together; a prediction for tomorrow needs the
forecast half, which is why both live here.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Iterator

from .config import SiteConfig
from .http import HttpClient

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

# the archive only has settled data up to about 6 days ago
ARCHIVE_LAG_DAYS = 6
# forecast covers 16 days counting today, so the last date it takes is today + 15
MAX_FORECAST_DAYS = 15

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "weather_code",
    "sunrise",
    "sunset",
]

# there is no daily humidity, so we average the hourly numbers ourselves
HOURLY_VARS = ["relative_humidity_2m"]

# wmo codes: 51+ is some kind of rain, 71+ is snow
RAIN_CODES = range(51, 70)
SNOW_CODES = range(71, 80)
STORM_CODES = range(95, 100)


def geocode(query: str, client: HttpClient | None = None) -> dict[str, Any] | None:
    """Resolve a city/postal string to coordinates. Used once, then cached in .env."""
    owns = client is None
    client = client or HttpClient()
    try:
        body = client.get(GEOCODE_URL, params={"name": query, "count": 1})
        results = body.get("results") or []
        if not results:
            return None
        top = results[0]
        return {
            "latitude": top["latitude"],
            "longitude": top["longitude"],
            "timezone": top.get("timezone", "America/New_York"),
            "name": top.get("name"),
            "admin1": top.get("admin1"),
            "country_code": top.get("country_code"),
        }
    finally:
        if owns:
            client.__exit__()


class WeatherCollector:
    def __init__(self, site: SiteConfig, client: HttpClient | None = None):
        self.site = site
        self._client = client or HttpClient()

    def __enter__(self) -> WeatherCollector:
        return self

    def __exit__(self, *exc: object) -> None:
        self._client.__exit__(*exc)

    def fetch(self, start: date, end: date) -> Iterator[dict[str, Any]]:
        """Daily weather for [start, end], drawing from whichever source covers it.

        A request past the 16-day forecast horizon is clipped rather than
        rejected — asking for 30 days of "future weather" is a reasonable thing
        to want and an error nobody can act on.
        """
        today = date.today()
        cutoff = today - timedelta(days=ARCHIVE_LAG_DAYS)
        horizon = today + timedelta(days=MAX_FORECAST_DAYS)

        if end > horizon:
            log.info("clipping weather request from %s to the %s forecast horizon", end, horizon)
            end = horizon

        if start <= cutoff:
            yield from self._request(ARCHIVE_URL, start, min(end, cutoff))
        if end > cutoff:
            yield from self._request(FORECAST_URL, max(start, cutoff + timedelta(days=1)), end)

    def fetch_forecast(self, days_ahead: int = 7) -> Iterator[dict[str, Any]]:
        """Tomorrow onward — the weather half of a live prediction."""
        today = date.today()
        end = today + timedelta(days=min(days_ahead, MAX_FORECAST_DAYS))
        yield from self._request(FORECAST_URL, today, end)

    def _request(self, url: str, start: date, end: date) -> Iterator[dict[str, Any]]:
        if start > end:
            return

        params = {
            "latitude": self.site.latitude,
            "longitude": self.site.longitude,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": ",".join(DAILY_VARS),
            "hourly": ",".join(HOURLY_VARS),
            "timezone": self.site.timezone,
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "inch",
            "wind_speed_unit": "mph",
        }
        body = self._client.get(url, params=params)

        daily = body.get("daily") or {}
        dates = daily.get("time") or []
        humidity = self._daily_humidity(body.get("hourly") or {})
        source = "archive" if url == ARCHIVE_URL else "forecast"
        log.info("weather %s: %s..%s (%d days)", source, start, end, len(dates))

        for index, day in enumerate(dates):
            code = _at(daily.get("weather_code"), index)
            row = {
                "date": day,
                "latitude": self.site.latitude,
                "longitude": self.site.longitude,
                "source": source,
                "temp_max_f": _at(daily.get("temperature_2m_max"), index),
                "temp_min_f": _at(daily.get("temperature_2m_min"), index),
                "temp_mean_f": _at(daily.get("temperature_2m_mean"), index),
                "feels_like_max_f": _at(daily.get("apparent_temperature_max"), index),
                "feels_like_min_f": _at(daily.get("apparent_temperature_min"), index),
                "precipitation_in": _at(daily.get("precipitation_sum"), index),
                "rain_in": _at(daily.get("rain_sum"), index),
                "snowfall_in": _at(daily.get("snowfall_sum"), index),
                "precipitation_hours": _at(daily.get("precipitation_hours"), index),
                "wind_max_mph": _at(daily.get("wind_speed_10m_max"), index),
                "wind_gust_mph": _at(daily.get("wind_gusts_10m_max"), index),
                "humidity_mean": humidity.get(day),
                "weather_code": code,
                "sunrise": _at(daily.get("sunrise"), index),
                "sunset": _at(daily.get("sunset"), index),
            }
            row.update(_weather_flags(code, row["precipitation_in"], row["snowfall_in"]))
            yield row

    @staticmethod
    def _daily_humidity(hourly: dict[str, Any]) -> dict[str, float]:
        """Mean relative humidity per calendar day, from hourly readings."""
        times = hourly.get("time") or []
        values = hourly.get("relative_humidity_2m") or []
        totals: dict[str, list[float]] = {}
        for stamp, value in zip(times, values):
            if value is None:
                continue
            totals.setdefault(stamp[:10], []).append(float(value))
        return {day: round(sum(v) / len(v), 1) for day, v in totals.items() if v}


def _at(series: list[Any] | None, index: int) -> Any:
    if not series or index >= len(series):
        return None
    return series[index]


def _weather_flags(code: Any, precipitation: Any, snowfall: Any) -> dict[str, bool]:
    """Booleans a model can use directly, rather than a raw WMO code."""
    code = int(code) if code is not None else -1
    return {
        "is_rainy": (precipitation or 0) > 0.01 or code in RAIN_CODES,
        "is_snowy": (snowfall or 0) > 0.01 or code in SNOW_CODES,
        "is_stormy": code in STORM_CODES,
    }
