"""Phase 3: clean, combine, and engineer.

    collector/  →  database/  →  pipelines/  →  training/

The database stores what happened. These pipelines decide what is fit to learn
from, roll it up to one row per trading day, and turn that into the arrays a
network trains on.

    python -m pipelines.validate_data        gate the data, non-zero exit if unsafe
    python -m pipelines.build_daily_summary  one clean row per day per location
    python -m pipelines.build_features       lags, encodings, split, scaling

Run them in that order — build_features reads the summary, and the summary is
only trustworthy if validation passed.
"""

# imported lazily, otherwise `python -m pipelines.<mod>` runs the module twice
_LAZY = {
    "build_daily_summary": (".build_daily_summary", "build"),
    "build_features": (".build_features", "build"),
    "load_dataset": (".build_features", "load_dataset"),
    "validate": (".validate_data", "validate"),
    "DataQualityError": (".validate_data", "DataQualityError"),
}


def __getattr__(name: str):
    from importlib import import_module

    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _LAZY[name]
    return getattr(import_module(module, __name__), attribute)


__all__ = [
    "DataQualityError",
    "build_daily_summary",
    "build_features",
    "load_dataset",
    "validate",
]
