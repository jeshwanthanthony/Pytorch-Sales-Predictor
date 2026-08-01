"""Phase 2: the warehouse.

Raw JSONL from collector/ goes in; a clean daily table comes out.

    python -m database load
    python -m database features

Nothing here learns anything — it stores, deduplicates, and organizes. PyTorch
reads `queries.feature_frame()` in phase 3.
"""

from .db import DB_PATH, connect, init_schema
from .load import load_all, summarize
from .queries import FEATURE_COLUMNS, TARGET_COLUMN, feature_frame, prediction_rows

__all__ = [
    "DB_PATH",
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "connect",
    "feature_frame",
    "init_schema",
    "load_all",
    "prediction_rows",
    "summarize",
]
