"""Phase 1 of the forecast pipeline: collect everything, transform nothing.

Sources
    square_api    orders, line items, payments, refunds, customers, catalog,
                  inventory, team members, labor shifts
    weather_api   daily weather, history and forecast (Open-Meteo, no key)
    calendar_api  holidays, day/month parts, school breaks, paydays
    events        local events and promotions, from reference CSVs

Everything lands as JSONL in data/raw/<entity>/. The database step reads that
directory; nothing here writes to a database or engineers a feature.

    python -m collector.run --check
    python -m collector.run --since 2024-01-01
"""

from .config import ConfigError, SiteConfig, SquareAuth, load_square_auth

__all__ = ["ConfigError", "SiteConfig", "SquareAuth", "load_square_auth"]
