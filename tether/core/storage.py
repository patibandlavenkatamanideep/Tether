"""Async SQLite storage layer for Tether.

Uses aiosqlite with WAL mode for safe concurrent reads during writes.
All public methods are async and open a fresh connection per operation
(connection pooling is a future enhancement — see TODO below).

TODO(v0.2): Replace per-operation connections with an aiosqlite connection pool
            to reduce connection-open overhead in high-frequency call scenarios.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import aiosqlite

from tether.core.logging import get_logger
from tether.core.models import (
    Checkpoint,
    FailureRecord,
    Provider,
    Run,
    RunStatus,
    Step,
    StepKind,
)
from tether.exceptions import StorageError

log = get_logger(__name__)

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    config_hash TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs (created_at);
CREATE INDEX IF NOT EXISTS idx_runs_status     ON runs (status);
"""

_CREATE_STEPS = """
CREATE TABLE IF NOT EXISTS steps (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    sequence_number INTEGER NOT NULL,
    kind            TEXT NOT NULL,
    provider        TEXT,
    model           TEXT,
    inputs          TEXT NOT NULL DEFAULT '{}',
    outputs         TEXT,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cost_usd        TEXT,
    latency_ms      REAL,
    error           TEXT,
    created_at      TEXT NOT NULL,
    completed_at    TEXT,
    UNIQUE (run_id, sequence_number)
);
CREATE INDEX IF NOT EXISTS idx_steps_run_seq ON steps (run_id, sequence_number);
"""

_CREATE_CHECKPOINTS = """
CREATE TABLE IF NOT EXISTS checkpoints (
    id            TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES runs(id),
    after_step_id TEXT NOT NULL REFERENCES steps(id),
    snapshot      TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_run_created ON checkpoints (run_id, created_at DESC);
"""

_CREATE_FAILURES = """
CREATE TABLE IF NOT EXISTS failures (
    id                 TEXT PRIMARY KEY,
    run_id             TEXT NOT NULL REFERENCES runs(id),
    step_id            TEXT NOT NULL REFERENCES steps(id),
    error_type         TEXT NOT NULL,
    error_message      TEXT NOT NULL,
    provider           TEXT NOT NULL,
    occurred_at        TEXT NOT NULL,
    recovery_action    TEXT,
    recovery_succeeded INTEGER
);
CREATE INDEX IF NOT EXISTS idx_failures_run ON failures (run_id);
"""


def _dt_to_str(dt: datetime) -> str:
    """Serialize a UTC datetime to an ISO 8601 string for SQLite storage."""
    return dt.isoformat()


def _str_to_dt(s: str) -> datetime:
    """Deserialize an ISO 8601 string from SQLite to a UTC-aware datetime."""
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _json_dumps(obj: Any) -> str:
    """JSON-serialize an object, handling Decimal and UUID types."""

    def _default(o: Any) -> Any:
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, UUID):
            return str(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    return json.dumps(obj, default=_default)


class SQLiteStorage:
    """Async SQLite-backed store for Runs, Steps, Checkpoints, and FailureRecords.

    Args:
        db_path: Path to the SQLite database file. Created if it does not exist.
                 Defaults to ``tether.db`` in the current working directory.

    Example:
        storage = SQLiteStorage("my_agent.db")
        await storage.initialize()
        await storage.create_run(run)
        await storage.close()
    """

    def __init__(self, db_path: str | Path = "tether.db") -> None:
        self._db_path = Path(db_path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create tables and enable SQLite pragmas required by Tether.

        Must be called once before any other method. Safe to call multiple
        times (all DDL uses IF NOT EXISTS).

        Raises:
            StorageError: If the database cannot be opened or pragmas cannot be set.
        """
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute("PRAGMA journal_mode=WAL")
                await db.execute("PRAGMA foreign_keys=ON")
                for ddl in [_CREATE_RUNS, _CREATE_STEPS, _CREATE_CHECKPOINTS, _CREATE_FAILURES]:
                    await db.executescript(ddl)
                await db.commit()
            log.info("storage_initialized", db_path=str(self._db_path))
        except Exception as exc:
            raise StorageError(
                f"Failed to initialize database at {self._db_path}: {exc}. "
                "Check that the directory exists and is writable.",
                original_error=exc,
            ) from exc

    async def close(self) -> None:
        """No-op — connections are opened and closed per operation.

        Provided for API symmetry and future connection-pool teardown.
        """

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    async def create_run(self, run: Run) -> None:
        """Persist a new Run to the database.

        Args:
            run: The Run to store. Its ``id`` must not already exist.

        Raises:
            StorageError: If insertion fails (e.g. duplicate ID).
        """
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute("PRAGMA foreign_keys=ON")
                await db.execute(
                    """
                    INSERT INTO runs
                        (id, name, status, created_at, updated_at, metadata, config_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(run.id),
                        run.name,
                        run.status.value,
                        _dt_to_str(run.created_at),
                        _dt_to_str(run.updated_at),
                        _json_dumps(run.metadata),
                        run.config_hash,
                    ),
                )
                await db.commit()
            log.debug("run_created", run_id=str(run.id), name=run.name)
        except Exception as exc:
            raise StorageError(
                f"Failed to create run {run.id}: {exc}",
                original_error=exc,
            ) from exc

    async def get_run(self, run_id: UUID) -> Run | None:
        """Fetch a Run by its ID.

        Args:
            run_id: The UUID of the Run to retrieve.

        Returns:
            The Run if found, or None if no such run exists.

        Raises:
            StorageError: If the query fails.
        """
        try:
            async with aiosqlite.connect(self._db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute("SELECT * FROM runs WHERE id = ?", (str(run_id),)) as cursor:
                    row = await cursor.fetchone()
            if row is None:
                return None
            return _row_to_run(row)
        except Exception as exc:
            raise StorageError(
                f"Failed to fetch run {run_id}: {exc}",
                original_error=exc,
            ) from exc

    async def update_run_status(self, run_id: UUID, status: RunStatus) -> None:
        """Update the status of an existing Run.

        Also updates ``updated_at`` to the current UTC time.

        Args:
            run_id: The UUID of the Run to update.
            status: The new lifecycle status.

        Raises:
            StorageError: If the update fails or the run does not exist.
        """
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    "UPDATE runs SET status = ?, updated_at = ? WHERE id = ?",
                    (status.value, _dt_to_str(datetime.now(UTC)), str(run_id)),
                )
                await db.commit()
            log.debug("run_status_updated", run_id=str(run_id), status=status.value)
        except Exception as exc:
            raise StorageError(
                f"Failed to update status for run {run_id}: {exc}",
                original_error=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    async def record_step(self, step: Step) -> None:
        """Persist a Step to the database.

        Args:
            step: The Step to store. (run_id, sequence_number) must be unique.

        Raises:
            StorageError: If insertion fails (e.g. duplicate sequence_number for this run).
        """
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute("PRAGMA foreign_keys=ON")
                await db.execute(
                    """
                    INSERT INTO steps (
                        id, run_id, sequence_number, kind, provider, model,
                        inputs, outputs, input_tokens, output_tokens,
                        cost_usd, latency_ms, error, created_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        outputs       = excluded.outputs,
                        input_tokens  = excluded.input_tokens,
                        output_tokens = excluded.output_tokens,
                        cost_usd      = excluded.cost_usd,
                        latency_ms    = excluded.latency_ms,
                        error         = excluded.error,
                        completed_at  = excluded.completed_at
                    """,
                    (
                        str(step.id),
                        str(step.run_id),
                        step.sequence_number,
                        step.kind.value,
                        step.provider.value if step.provider else None,
                        step.model,
                        _json_dumps(step.inputs),
                        _json_dumps(step.outputs) if step.outputs is not None else None,
                        step.input_tokens,
                        step.output_tokens,
                        str(step.cost_usd) if step.cost_usd is not None else None,
                        step.latency_ms,
                        _json_dumps(step.error) if step.error is not None else None,
                        _dt_to_str(step.created_at),
                        _dt_to_str(step.completed_at) if step.completed_at else None,
                    ),
                )
                await db.commit()
            log.debug(
                "step_recorded",
                step_id=str(step.id),
                run_id=str(step.run_id),
                seq=step.sequence_number,
            )
        except Exception as exc:
            raise StorageError(
                f"Failed to record step {step.id} (seq={step.sequence_number}) "
                f"for run {step.run_id}: {exc}",
                original_error=exc,
            ) from exc

    async def get_step(self, step_id: UUID) -> Step | None:
        """Fetch a Step by its ID.

        Args:
            step_id: The UUID of the Step to retrieve.

        Returns:
            The Step if found, or None.

        Raises:
            StorageError: If the query fails.
        """
        try:
            async with aiosqlite.connect(self._db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT * FROM steps WHERE id = ?", (str(step_id),)
                ) as cursor:
                    row = await cursor.fetchone()
            if row is None:
                return None
            return _row_to_step(row)
        except Exception as exc:
            raise StorageError(
                f"Failed to fetch step {step_id}: {exc}",
                original_error=exc,
            ) from exc

    async def get_steps_for_run(self, run_id: UUID) -> list[Step]:
        """Return all Steps for a Run, ordered by sequence_number ascending.

        Args:
            run_id: The UUID of the parent Run.

        Returns:
            A list of Steps in execution order. Empty list if none exist.

        Raises:
            StorageError: If the query fails.
        """
        try:
            async with aiosqlite.connect(self._db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT * FROM steps WHERE run_id = ? ORDER BY sequence_number ASC",
                    (str(run_id),),
                ) as cursor:
                    rows = await cursor.fetchall()
            return [_row_to_step(row) for row in rows]
        except Exception as exc:
            raise StorageError(
                f"Failed to fetch steps for run {run_id}: {exc}",
                original_error=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Persist a Checkpoint to the database.

        Args:
            checkpoint: The Checkpoint to store.

        Raises:
            StorageError: If insertion fails.
        """
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute("PRAGMA foreign_keys=ON")
                await db.execute(
                    """
                    INSERT INTO checkpoints (id, run_id, after_step_id, snapshot, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(checkpoint.id),
                        str(checkpoint.run_id),
                        str(checkpoint.after_step_id),
                        _json_dumps(checkpoint.snapshot),
                        _dt_to_str(checkpoint.created_at),
                    ),
                )
                await db.commit()
            log.debug(
                "checkpoint_saved",
                checkpoint_id=str(checkpoint.id),
                run_id=str(checkpoint.run_id),
            )
        except Exception as exc:
            raise StorageError(
                f"Failed to save checkpoint for run {checkpoint.run_id}: {exc}",
                original_error=exc,
            ) from exc

    async def get_latest_checkpoint(self, run_id: UUID) -> Checkpoint | None:
        """Return the most recently saved Checkpoint for a Run.

        Args:
            run_id: The UUID of the parent Run.

        Returns:
            The most recent Checkpoint, or None if no checkpoints exist.

        Raises:
            StorageError: If the query fails.
        """
        try:
            async with aiosqlite.connect(self._db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    """
                    SELECT * FROM checkpoints
                    WHERE run_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (str(run_id),),
                ) as cursor:
                    row = await cursor.fetchone()
            if row is None:
                return None
            return _row_to_checkpoint(row)
        except Exception as exc:
            raise StorageError(
                f"Failed to fetch latest checkpoint for run {run_id}: {exc}",
                original_error=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Failures
    # ------------------------------------------------------------------

    async def record_failure(self, failure: FailureRecord) -> None:
        """Persist a FailureRecord to the database.

        Args:
            failure: The FailureRecord to store.

        Raises:
            StorageError: If insertion fails.
        """
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute("PRAGMA foreign_keys=ON")
                await db.execute(
                    """
                    INSERT INTO failures (
                        id, run_id, step_id, error_type, error_message,
                        provider, occurred_at, recovery_action, recovery_succeeded
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(failure.id),
                        str(failure.run_id),
                        str(failure.step_id),
                        failure.error_type,
                        failure.error_message,
                        failure.provider.value,
                        _dt_to_str(failure.occurred_at),
                        failure.recovery_action,
                        (
                            1
                            if failure.recovery_succeeded is True
                            else 0
                            if failure.recovery_succeeded is False
                            else None
                        ),
                    ),
                )
                await db.commit()
            log.debug(
                "failure_recorded",
                failure_id=str(failure.id),
                run_id=str(failure.run_id),
                error_type=failure.error_type,
            )
        except Exception as exc:
            raise StorageError(
                f"Failed to record failure for run {failure.run_id}: {exc}",
                original_error=exc,
            ) from exc


# ------------------------------------------------------------------
# Row → model helpers (module-private)
# ------------------------------------------------------------------


def _row_to_run(row: aiosqlite.Row) -> Run:
    """Deserialize an aiosqlite Row into a Run model."""
    return Run(
        id=UUID(row["id"]),
        name=row["name"],
        status=RunStatus(row["status"]),
        created_at=_str_to_dt(row["created_at"]),
        updated_at=_str_to_dt(row["updated_at"]),
        metadata=json.loads(row["metadata"]),
        config_hash=row["config_hash"],
    )


def _row_to_step(row: aiosqlite.Row) -> Step:
    """Deserialize an aiosqlite Row into a Step model."""
    return Step(
        id=UUID(row["id"]),
        run_id=UUID(row["run_id"]),
        sequence_number=row["sequence_number"],
        kind=StepKind(row["kind"]),
        provider=Provider(row["provider"]) if row["provider"] else None,
        model=row["model"],
        inputs=json.loads(row["inputs"]),
        outputs=json.loads(row["outputs"]) if row["outputs"] else None,
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        cost_usd=Decimal(row["cost_usd"]) if row["cost_usd"] else None,
        latency_ms=row["latency_ms"],
        error=json.loads(row["error"]) if row["error"] else None,
        created_at=_str_to_dt(row["created_at"]),
        completed_at=_str_to_dt(row["completed_at"]) if row["completed_at"] else None,
    )


def _row_to_checkpoint(row: aiosqlite.Row) -> Checkpoint:
    """Deserialize an aiosqlite Row into a Checkpoint model."""
    return Checkpoint(
        id=UUID(row["id"]),
        run_id=UUID(row["run_id"]),
        after_step_id=UUID(row["after_step_id"]),
        snapshot=json.loads(row["snapshot"]),
        created_at=_str_to_dt(row["created_at"]),
    )


def _row_to_failure(row: aiosqlite.Row) -> FailureRecord:
    """Deserialize an aiosqlite Row into a FailureRecord model."""
    rs = row["recovery_succeeded"]
    return FailureRecord(
        id=UUID(row["id"]),
        run_id=UUID(row["run_id"]),
        step_id=UUID(row["step_id"]),
        error_type=row["error_type"],
        error_message=row["error_message"],
        provider=Provider(row["provider"]),
        occurred_at=_str_to_dt(row["occurred_at"]),
        recovery_action=row["recovery_action"],
        recovery_succeeded=None if rs is None else bool(rs),
    )
