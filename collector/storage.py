"""The raw landing zone.

Every pull lands as immutable JSONL under data/raw/<entity>/, one file per run.
Nothing here dedupes or updates in place — the database step owns that, and
keeping the raw layer append-only means a bad transform is always replayable.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import RAW_DIR, STATE_FILE

log = logging.getLogger(__name__)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class RawWriter:
    """Writes one entity's records for one run."""

    def __init__(
        self,
        entity: str,
        run_id: str | None = None,
        watermark_field: str | None = None,
        raw_dir: Path = RAW_DIR,
    ):
        self.entity = entity
        self.run_id = run_id or _stamp()
        self.path = Path(raw_dir) / entity / f"{entity}-{self.run_id}.jsonl"
        self.count = 0
        # track the newest timestamp as we write, so we do not hold every row twice
        self.watermark_field = watermark_field
        self.watermark: str | None = None

    def write_all(self, records: Iterable[dict[str, Any]]) -> int:
        """Stream records to disk. Writes no file at all if there were none."""
        handle = None
        try:
            for record in records:
                if handle is None:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    handle = self.path.open("w", encoding="utf-8")
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                self.count += 1

                if self.watermark_field:
                    value = record.get(self.watermark_field)
                    if value and (self.watermark is None or value > self.watermark):
                        self.watermark = value
        finally:
            if handle is not None:
                handle.close()

        if self.count:
            log.info("%-14s %6d rows -> %s", self.entity, self.count, self.path.name)
        else:
            log.info("%-14s %6d rows (nothing new)", self.entity, 0)
        return self.count


def read_entity(entity: str, raw_dir: Path = RAW_DIR) -> list[dict[str, Any]]:
    """Read every run's records for an entity, oldest file first.

    Useful for eyeballing a pull; the database loader will use the same order so
    later versions of a row overwrite earlier ones.
    """
    directory = Path(raw_dir) / entity
    if not directory.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob(f"{entity}-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


class State:
    """High-water marks, so a second run only asks Square for what's new."""

    def __init__(self, state_file: Path = STATE_FILE) -> None:
        self._data: dict[str, Any] = {}
        self.state_file = Path(state_file)
        if self.state_file.exists():
            self._data = json.loads(self.state_file.read_text() or "{}")

    def get(self, entity: str, field: str = "watermark") -> Any:
        return self._data.get(entity, {}).get(field)

    def set(self, entity: str, value: Any, field: str = "watermark") -> None:
        self._data.setdefault(entity, {})[field] = value

    def record_run(self, entity: str, rows: int) -> None:
        bucket = self._data.setdefault(entity, {})
        bucket["last_run_at"] = datetime.now(timezone.utc).isoformat()
        bucket["last_row_count"] = rows

    def save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self._data, indent=2, sort_keys=True))

    def summary(self) -> dict[str, Any]:
        return dict(self._data)
