"""Runs the whole chain for one restaurant, in the background.

collect -> store -> check -> daily rows -> features -> train -> score

One job per restaurant at a time. Progress is reported step by step so the page
can show what is happening instead of a spinner.

If a step fails the run stops there and keeps the reason. The reason is usually
the useful part: "not enough sales history yet" is an answer, not a crash.
"""

from __future__ import annotations

import logging
import json
import threading
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable

from .workspace import Workspace

log = logging.getLogger("setup")

# how much history to ask Square for
BACKFILL_DAYS = 730
# a model needs at least this many trading days to be worth anything
MIN_DAYS_TO_TRAIN = 60


class StepFailed(RuntimeError):
    """A step stopped the run, with a reason worth showing the user."""


@dataclass
class Step:
    name: str
    label: str
    status: str = "waiting"  # waiting | running | done | failed
    detail: str = ""


@dataclass
class RunState:
    steps: list[Step] = field(default_factory=list)
    running: bool = False
    finished_at: str | None = None
    error: str | None = None
    started_at: str | None = None
    logs: list[dict] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "running": self.running,
            "finished_at": self.finished_at,
            "error": self.error,
            "started_at": self.started_at,
            "complete": bool(self.finished_at) and self.error is None,
            "steps": [vars(s) for s in self.steps],
            "logs": list(self.logs),
            "tables": list(self.tables),
        }

    @classmethod
    def from_dict(cls, data: dict) -> RunState:
        return cls(
            steps=[Step(**step) for step in data.get("steps", [])] or build_steps(),
            running=False,  # a process restart cannot leave a real worker running
            finished_at=data.get("finished_at"),
            error=data.get("error"),
            started_at=data.get("started_at"),
            logs=list(data.get("logs", [])),
            tables=list(data.get("tables", [])),
        )


def build_steps() -> list[Step]:
    return [
        Step("collect", "Reading your Square history"),
        Step("store", "Storing it"),
        Step("check", "Checking it is usable"),
        Step("daily", "Adding it up by day"),
        Step("features", "Preparing what the model learns from"),
        Step("train", "Training your model"),
        Step("score", "Checking it beats a simple guess"),
    ]


class PipelineRunner:
    """One run, for one restaurant."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.state = self._load_state()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _load_state(self) -> RunState:
        path = self.workspace.pipeline_state_file
        if not path.exists():
            return RunState(steps=build_steps())
        try:
            return RunState.from_dict(json.loads(path.read_text()))
        except (OSError, ValueError, TypeError):
            log.warning("could not restore pipeline status for %s", self.workspace.merchant_id)
            return RunState(steps=build_steps())

    def _persist(self) -> None:
        self.workspace.ensure()
        path = self.workspace.pipeline_state_file
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state.to_dict(), indent=2))
        temporary.replace(path)

    def start(self, on_finish: Callable[[], None] | None = None) -> bool:
        with self._lock:
            if self.state.running:
                return False
            self.state = RunState(
                steps=build_steps(),
                running=True,
                started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            self._log("system", "pipeline started")

        self._thread = threading.Thread(target=self._run, args=(on_finish,), daemon=True)
        self._thread.start()
        return True

    @property
    def is_running(self) -> bool:
        return self.state.running

    # -- the work ----------------------------------------------------------

    def _run(self, on_finish: Callable[[], None] | None) -> None:
        try:
            self._collect()
            self._store()
            self._check()
            self._daily()
            self._features()
            self._train()
            self._score()
        except StepFailed as exc:
            self.state.error = str(exc)
            self._log("error", str(exc))
            log.info("setup stopped for %s: %s", self.workspace.merchant_id, exc)
        except Exception as exc:  # noqa: BLE001 - never kill the thread silently
            self.state.error = f"something went wrong: {exc}"
            self._log("error", self.state.error)
            log.error("setup crashed:\n%s", traceback.format_exc())
        finally:
            self.state.running = False
            self.state.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._persist()
            if on_finish:
                try:
                    on_finish()
                except Exception:  # noqa: BLE001
                    log.exception("could not reload the model")

    def _step(self, name: str) -> Step:
        step = next(s for s in self.state.steps if s.name == name)
        step.status = "running"
        self._log("stage", f"{name} -> {step.label}")
        return step

    def _log(self, kind: str, message: str) -> None:
        self.state.logs.append(
            {
                "time": datetime.now().astimezone().strftime("%H:%M:%S"),
                "kind": kind,
                "message": message,
            }
        )
        # Keep status polling small even during a long training run.
        self.state.logs = self.state.logs[-500:]
        self._persist()

    def _table(self, title: str, columns: list[str], rows: list[list]) -> None:
        self.state.tables.append({"title": title, "columns": columns, "rows": rows})
        self.state.tables = self.state.tables[-4:]
        self._persist()

    def _fail(self, step: Step, message: str) -> None:
        step.status = "failed"
        step.detail = message
        raise StepFailed(message)

    def _collect(self) -> None:
        from collector.collect import collect_all

        step = self._step("collect")
        since = date.today() - timedelta(days=BACKFILL_DAYS)
        current_entity = ""
        last_logged: dict[str, int] = {}

        def collection_progress(entity: str, count: int) -> None:
            nonlocal current_entity
            step.detail = f"{entity}: {count:,} rows received"
            previous = last_logged.get(entity, -1)
            if entity != current_entity:
                current_entity = entity
                self._log("square", f"requesting {entity}")
            if count and (count < 1_000 or count - previous >= 10_000):
                last_logged[entity] = count
                self._log("square", f"{entity}: {count:,} rows")

        result = collect_all(
            auth=self.workspace.auth(),
            raw_dir=self.workspace.raw_dir,
            state_file=self.workspace.state_file,
            since=since,
            on_progress=collection_progress,
            forecast_only=True,
            # A failed first setup can retry transforms without downloading the
            # same two-year Square export again. Once a model exists, reruns do
            # fetch Square so "update latest sales" still means what it says.
            resume_existing=self.workspace.db_path.exists() and not self.workspace.has_model(),
        )

        if result.failures.get("orders"):
            self._fail(step, f"could not read your orders: {result.failures['orders']}")

        step.status = "done"
        step.detail = f"{result.order_count:,} orders found"
        self._log("done", f"Square download complete: {result.order_count:,} orders")
        self._table(
            "raw Square data",
            ["dataset", "rows"],
            [[name, f"{count:,}"] for name, count in result.rows.items()],
        )

    def _store(self) -> None:
        from database.load import load_all

        step = self._step("store")
        def store_progress(entity: str, count: int) -> None:
            step.detail = f"{entity}: {count:,} rows cleaned and stored"
            self._log("sqlite", f"{entity}: {count:,} rows upserted")

        results = load_all(
            self.workspace.db_path,
            raw_dir=self.workspace.raw_dir,
            on_progress=store_progress,
        )
        cutoff = (date.today() - timedelta(days=BACKFILL_DAYS)).isoformat()
        trimmed = self._trim_history(cutoff)
        if trimmed:
            self._log(
                "clean",
                f"excluded {trimmed:,} stale rows created before {cutoff}; "
                "Square returned them because they were updated recently",
            )
        total = sum(v for v in results.values() if v > 0)
        step.status = "done"
        step.detail = f"{total:,} rows stored"
        self._log("done", f"warehouse ready: {total:,} rows processed")

    def _trim_history(self, cutoff: str) -> int:
        """Keep the model window coherent when old orders were updated recently."""
        from database.db import connect, transaction

        conn = connect(self.workspace.db_path)
        try:
            old_orders = [
                row[0]
                for row in conn.execute(
                    "SELECT order_id FROM orders WHERE business_date < ?", (cutoff,)
                )
            ]
            with transaction(conn):
                before = conn.total_changes
                conn.execute(
                    "DELETE FROM order_item_modifiers WHERE order_id IN "
                    "(SELECT order_id FROM orders WHERE business_date < ?)",
                    (cutoff,),
                )
                conn.execute("DELETE FROM order_items WHERE business_date < ?", (cutoff,))
                conn.execute("DELETE FROM payments WHERE business_date < ?", (cutoff,))
                conn.execute("DELETE FROM refunds WHERE business_date < ?", (cutoff,))
                conn.execute("DELETE FROM shifts WHERE business_date < ?", (cutoff,))
                if old_orders:
                    conn.executemany("DELETE FROM orders WHERE order_id = ?", ((x,) for x in old_orders))
                return conn.total_changes - before
        finally:
            conn.close()

    def _check(self) -> None:
        from pipelines.validate_data import validate

        step = self._step("check")
        report = validate(self.workspace.db_path)
        if not report.ok:
            self._fail(step, "; ".join(c.detail for c in report.errors))

        from database.db import connect

        conn = connect(self.workspace.db_path, read_only=True)
        try:
            days = conn.execute(
                "SELECT COUNT(DISTINCT business_date) FROM daily_sales"
            ).fetchone()[0]
        finally:
            conn.close()
        if days < MIN_DAYS_TO_TRAIN:
            self._fail(
                step,
                f"only {days} days of sales so far. A forecast needs about "
                f"{MIN_DAYS_TO_TRAIN} days before it means anything.",
            )
        step.status = "done"
        step.detail = f"{days} days of sales"
        self._log("check", f"validation passed: {days} trading days")

    def _daily(self) -> None:
        from pipelines.build_daily_summary import build

        step = self._step("daily")
        rows = build(self.workspace.db_path)
        step.status = "done"
        step.detail = f"{rows:,} days"
        self._log("clean", f"daily aggregation built: {rows:,} restaurant-day rows")

        from database.db import connect

        conn = connect(self.workspace.db_path, read_only=True)
        try:
            sample = conn.execute(
                "SELECT business_date, location_id, order_count, net_sales_cents "
                "FROM daily_summary ORDER BY business_date DESC, location_id LIMIT 6"
            ).fetchall()
        finally:
            conn.close()
        self._table(
            "clean daily table",
            ["date", "location", "orders", "net sales"],
            [
                [row["business_date"], row["location_id"], row["order_count"],
                 f"${row['net_sales_cents'] / 100:,.2f}"]
                for row in sample
            ],
        )

    def _features(self) -> None:
        from pipelines.build_features import build

        step = self._step("features")
        try:
            manifest = build(self.workspace.db_path, output_dir=self.workspace.features_dir)
        except RuntimeError as exc:
            self._fail(step, str(exc))

        counts = manifest["row_counts"]
        if counts["train"] < 30:
            self._fail(
                step,
                f"only {counts['train']} days left to learn from after preparing history. "
                "More trading history is needed.",
            )
        step.status = "done"
        step.detail = (
            f"{manifest['feature_count']} things to learn from, "
            f"{counts['train']} days to learn, {counts['test']} days to test"
        )
        self._log(
            "pandas",
            f"DataFrame -> {manifest['feature_count']} features; "
            f"train={counts['train']}, val={counts['val']}, test={counts['test']}, "
            f"future={counts['future']}",
        )
        self._table(
            "pandas / PyTorch splits",
            ["split", "rows", "date range"],
            [
                [name, counts[name], " -> ".join(manifest.get("date_spans", {}).get(name, [])) or "-"]
                for name in ("train", "val", "test", "future")
            ],
        )

    def _train(self) -> None:
        from training.train import train

        step = self._step("train")

        def epoch_progress(epoch: dict) -> None:
            metrics = epoch["val"]
            step.detail = (
                f"epoch {epoch['epoch']}: train {epoch['train_loss']:.4f}, "
                f"val {epoch['val_loss']:.4f}, MAE ${metrics['mae_dollars']:,.0f}"
            )
            self._log(
                "torch",
                f"epoch {epoch['epoch']:03d} | train_loss={epoch['train_loss']:.4f} "
                f"val_loss={epoch['val_loss']:.4f} val_MAE=${metrics['mae_dollars']:,.2f}",
            )

        result = train(
            dataset_file=self.workspace.dataset_file,
            manifest_file=self.workspace.manifest_file,
            output_dir=self.workspace.models_dir,
            on_epoch=epoch_progress,
        )
        step.status = "done"
        step.detail = (
            f"best at round {result['best_epoch']}, "
            f"typically ${result['val_metrics']['mae_dollars']:,.0f} off per day"
        )
        self._log(
            "done",
            f"training complete: best epoch {result['best_epoch']}; "
            f"validation MAE ${result['val_metrics']['mae_dollars']:,.2f}",
        )

    def _score(self) -> None:
        from training.evaluate import evaluate

        step = self._step("score")
        payload = evaluate(
            self.workspace.checkpoint,
            self.workspace.dataset_file,
            self.workspace.manifest_file,
            self.workspace.models_dir,
        )
        verdict = payload["verdict"]
        step.status = "done"
        step.detail = (
            "beats both simple guesses"
            if verdict["beats_all_baselines"]
            else "a simple guess is still as good, treat this with care"
        )
        metrics = payload["splits"]["test"]["model"]
        self._log(
            "score",
            f"test MAE=${metrics['mae_dollars']:,.2f}; RMSE=${metrics['rmse_dollars']:,.2f}",
        )

        from .service import ModelService

        forecast = ModelService.for_workspace(self.workspace).predict_next()
        self._log(
            "prediction",
            f"{forecast['business_date']} predicted sales=${forecast['predicted_sales']:,.2f} "
            f"range=${forecast['interval_low']:,.2f}-${forecast['interval_high']:,.2f}",
        )
        self._table(
            "tomorrow's prediction",
            ["date", "prediction", "low", "high", "confidence"],
            [[
                forecast["business_date"],
                f"${forecast['predicted_sales']:,.2f}",
                f"${forecast['interval_low']:,.2f}",
                f"${forecast['interval_high']:,.2f}",
                f"{forecast['confidence'] * 100:.0f}%",
            ]],
        )


class RunnerRegistry:
    """One runner per restaurant, so two accounts never trip over each other."""

    def __init__(self) -> None:
        self._runners: dict[str, PipelineRunner] = {}
        self._lock = threading.Lock()

    def for_workspace(self, workspace: Workspace) -> PipelineRunner:
        with self._lock:
            runner = self._runners.get(workspace.merchant_id)
            if runner is None:
                runner = PipelineRunner(workspace)
                self._runners[workspace.merchant_id] = runner
            return runner

    def forget(self, merchant_id: str) -> None:
        with self._lock:
            self._runners.pop(merchant_id, None)
