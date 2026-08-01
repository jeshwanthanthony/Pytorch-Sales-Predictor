"""Connection handling, pragmas, and the generic upsert.

SQLite rather than Postgres, deliberately: this is a single-writer batch
pipeline against a few hundred thousand rows. One file, no daemon, no port, and
the schema is plain enough to lift into Postgres the day concurrency matters.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "forecast.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

log = logging.getLogger(__name__)

# settings for every connection
#   WAL         readers do not block the writer
#   NORMAL sync a crash can lose the last transaction, but we can just re-pull it
#   64MB cache  the daily rollups keep hitting the same pages
PRAGMAS = [
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA foreign_keys = ON",
    "PRAGMA cache_size = -64000",
    "PRAGMA temp_store = MEMORY",
    "PRAGMA busy_timeout = 5000",
]


def connect(path: Path | str = DB_PATH, read_only: bool = False) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if read_only:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(path)

    conn.row_factory = sqlite3.Row
    for pragma in PRAGMAS:
        if read_only and "journal_mode" in pragma:
            continue
        conn.execute(pragma)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables, indexes, and views. Safe to run repeatedly."""
    conn.executescript(SCHEMA_PATH.read_text())
    conn.commit()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """One commit per batch — the difference between minutes and seconds."""
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def encode(value: Any) -> Any:
    """Python value -> something a STRICT column accepts.

    Lists and dicts become JSON text; bools become 0/1. Everything else is
    already an int, float, str, or None.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def column_defaults(conn: sqlite3.Connection, table: str) -> dict[str, Any]:
    """Defaults for NOT NULL columns, read from the schema.

    A missing key in a raw record becomes None, and an explicit None defeats a
    column DEFAULT — so the load would fail on a record where the upstream API
    simply omitted a field. Substituting the declared default here keeps that a
    zero instead of a crash, without duplicating the schema in Python.
    """
    defaults: dict[str, Any] = {}
    for row in conn.execute(f"PRAGMA table_info({table})"):
        if row["notnull"] and row["dflt_value"] is not None:
            raw = row["dflt_value"]
            try:
                defaults[row["name"]] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                defaults[row["name"]] = raw.strip("'")
    return defaults


def upsert(
    conn: sqlite3.Connection,
    table: str,
    columns: Sequence[str],
    keys: Sequence[str],
    rows: Iterable[dict[str, Any]],
    batch_size: int = 1000,
) -> int:
    """INSERT ... ON CONFLICT DO UPDATE, batched.

    Re-loading the same raw file is therefore a no-op, and an order that Square
    re-sent because it was refunded overwrites the stale copy in place. That
    property is the entire reason the primary keys are Square's ids.
    """
    placeholders = ", ".join("?" for _ in columns)
    updates = ", ".join(f"{c} = excluded.{c}" for c in columns if c not in keys)
    conflict = ", ".join(keys)

    sql = (
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT({conflict}) DO UPDATE SET {updates}"
        if updates
        else f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT({conflict}) DO NOTHING"
    )

    defaults = column_defaults(conn, table)

    def value_for(row: dict[str, Any], column: str) -> Any:
        value = encode(row.get(column))
        return defaults.get(column) if value is None and column in defaults else value

    total = 0
    batch: list[tuple[Any, ...]] = []
    for row in rows:
        batch.append(tuple(value_for(row, column) for column in columns))
        if len(batch) >= batch_size:
            conn.executemany(sql, batch)
            total += len(batch)
            batch.clear()

    if batch:
        conn.executemany(sql, batch)
        total += len(batch)

    return total


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Row count per table, for the stats command."""
    names = [
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    return {name: conn.execute(f"SELECT COUNT(*) AS n FROM {name}").fetchone()["n"] for name in names}


def optimize(conn: sqlite3.Connection) -> None:
    """Refresh the query planner's statistics after a big load."""
    conn.execute("ANALYZE")
    conn.execute("PRAGMA optimize")
    conn.commit()
