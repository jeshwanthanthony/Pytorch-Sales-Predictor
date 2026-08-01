"""Phase 4: teach the model.

    collector/ -> database/ -> pipelines/ -> training/

Reads the feature file pipelines/build_features.py wrote, trains a small
feed-forward network, and saves everything needed to repeat a prediction.

    python -m training.train
    python -m training.evaluate
    python -m training.predict
"""

# imported lazily so `python -m training.train` does not run the module twice
_LAZY = {
    "TrainConfig": (".config", "TrainConfig"),
    "load_splits": (".dataset", "load_splits"),
    "set_seed": (".dataset", "set_seed"),
    "SalesForecastNet": (".model", "SalesForecastNet"),
    "ModelConfig": (".model", "ModelConfig"),
    "train": (".train", "train"),
    "evaluate": (".evaluate", "evaluate"),
    "predict": (".predict", "predict"),
}


def __getattr__(name: str):
    from importlib import import_module

    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _LAZY[name]
    return getattr(import_module(module, __name__), attribute)


__all__ = [
    "ModelConfig",
    "TrainConfig",
    "SalesForecastNet",
    "evaluate",
    "load_splits",
    "predict",
    "set_seed",
    "train",
]
