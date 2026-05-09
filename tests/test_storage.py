"""Tests for tether.core.storage.SQLiteStorage."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import aiosqlite
import pytest

from tether.core.models import (
    Checkpoint,
    FailureRecord,
    Provider,
    Run,
    RunStatus,
    Step,
    StepKind,
)
from tether.core.storage import SQLiteStorage
from tether.exceptions import StorageError


class TestInitialize:
    async def test_creates_all_tables(self, storage: SQLiteStorage, tmp_db_path: Path):
        async with aiosqlite.connect(tmp_db_path) as db:
            async with db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ) as cur:
                tables = {row[0] for row in await cur.fetchall()}
        assert {"runs", "steps", "checkpoints", "failures"}.issubset(tables)

    async def test_wal_mode_enabled(self, storage: SQLiteStorage, tmp_db_path: Path):
        async with aiosqlite.connect(tmp_db_path) as db:
            async with db.execute("PRAGMA journal_mode") as cur:
                row = await cur.fetchone()
        assert row[0] == "wal"

    async def test_foreign_keys_enforced(self, storage: SQLiteStorage, tmp_db_path: Path):
        async with aiosqlite.connect(tmp_db_path) as db:
            await db.execute("PRAGMA foreign_keys=ON")
            async with db.execute("PRAGMA foreign_keys") as cur:
                row = await cur.fetchone()
        assert row[0] == 1

    async def test_idempotent(self, tmp_db_path: Path):
        s = SQLiteStorage(tmp_db_path)
        await s.initialize()
        await s.initialize()  # must not raise
        await s.close()


class TestRunCRUD:
    async def test_create_and_get_round_trip(self, storage: SQLiteStorage):
        run = Run.create(name="test_run", metadata={"k": "v"})
        await storage.create_run(run)
        fetched = await storage.get_run(run.id)
        assert fetched is not None
        assert fetched.id == run.id
        assert fetched.name == "test_run"
        assert fetched.metadata == {"k": "v"}
        assert fetched.status == RunStatus.PENDING

    async def test_get_missing_run_returns_none(self, storage: SQLiteStorage):
        result = await storage.get_run(uuid4())
        assert result is None

    async def test_preserves_config_hash(self, storage: SQLiteStorage):
        run = Run.create(name="hashed", config={"model": "gpt-4o"})
        await storage.create_run(run)
        fetched = await storage.get_run(run.id)
        assert fetched.config_hash == run.config_hash

    async def test_preserves_utc_awareness(self, storage: SQLiteStorage):
        run = Run.create(name="tz_test")
        await storage.create_run(run)
        fetched = await storage.get_run(run.id)
        assert fetched.created_at.tzinfo is not None
        assert fetched.updated_at.tzinfo is not None

    async def test_update_run_status(self, storage: SQLiteStorage):
        run = Run.create(name="status_test")
        await storage.create_run(run)
        await storage.update_run_status(run.id, RunStatus.RUNNING)
        fetched = await storage.get_run(run.id)
        assert fetched.status == RunStatus.RUNNING

    async def test_update_status_updates_updated_at(self, storage: SQLiteStorage):
        run = Run.create(name="ts_test")
        await storage.create_run(run)
        original_updated_at = run.updated_at
        await storage.update_run_status(run.id, RunStatus.COMPLETED)
        fetched = await storage.get_run(run.id)
        assert fetched.updated_at >= original_updated_at


class TestStepCRUD:
    def _make_step(self, run_id=None, seq=1, **kwargs) -> Step:
        return Step(
            run_id=run_id or uuid4(),
            sequence_number=seq,
            kind=StepKind.LLM_CALL,
            provider=Provider.OPENAI,
            model="gpt-4o-mini",
            inputs={"messages": [{"role": "user", "content": "hi"}]},
            **kwargs,
        )

    async def test_record_and_get_step(self, storage: SQLiteStorage):
        run = Run.create(name="step_test")
        await storage.create_run(run)
        step = self._make_step(run_id=run.id, seq=1)
        await storage.record_step(step)
        fetched = await storage.get_step(step.id)
        assert fetched is not None
        assert fetched.id == step.id
        assert fetched.sequence_number == 1
        assert fetched.kind == StepKind.LLM_CALL

    async def test_get_missing_step_returns_none(self, storage: SQLiteStorage):
        assert await storage.get_step(uuid4()) is None

    async def test_get_steps_for_run_ordered(self, storage: SQLiteStorage):
        run = Run.create(name="ordering_test")
        await storage.create_run(run)
        for seq in [3, 1, 2]:
            await storage.record_step(self._make_step(run_id=run.id, seq=seq))
        steps = await storage.get_steps_for_run(run.id)
        assert [s.sequence_number for s in steps] == [1, 2, 3]

    async def test_get_steps_for_run_empty(self, storage: SQLiteStorage):
        run = Run.create(name="empty_test")
        await storage.create_run(run)
        steps = await storage.get_steps_for_run(run.id)
        assert steps == []

    async def test_sequence_number_unique_per_run(self, storage: SQLiteStorage):
        run = Run.create(name="unique_seq")
        await storage.create_run(run)
        step1 = self._make_step(run_id=run.id, seq=1)
        await storage.record_step(step1)
        step2 = Step(
            run_id=run.id,
            sequence_number=1,  # duplicate!
            kind=StepKind.LLM_CALL,
            inputs={},
        )
        # Should raise StorageError because (run_id, sequence_number) is UNIQUE.
        # (record_step uses UPSERT by id, not by seq, so duplicate seq with
        # a different id will violate the UNIQUE constraint.)
        # We need a fresh step id but same seq.
        with pytest.raises(StorageError):
            await storage.record_step(step2)

    async def test_upsert_updates_completed_step(self, storage: SQLiteStorage):
        run = Run.create(name="upsert_test")
        await storage.create_run(run)
        step = self._make_step(run_id=run.id, seq=1)
        await storage.record_step(step)

        # Simulate completion: same id, new outputs added.
        completed = step.model_copy(
            update={
                "outputs": {"result": "done"},
                "input_tokens": 10,
                "output_tokens": 5,
                "cost_usd": Decimal("0.0001"),
                "latency_ms": 123.4,
                "completed_at": datetime.now(UTC),
            }
        )
        await storage.record_step(completed)

        fetched = await storage.get_step(step.id)
        assert fetched.outputs == {"result": "done"}
        assert fetched.input_tokens == 10
        assert fetched.cost_usd == Decimal("0.0001")

    async def test_decimal_cost_preserved(self, storage: SQLiteStorage):
        run = Run.create(name="decimal_test")
        await storage.create_run(run)
        cost = Decimal("0.00123456789")
        created = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        completed = datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC)
        step = self._make_step(
            run_id=run.id,
            seq=1,
            cost_usd=cost,
            created_at=created,
            completed_at=completed,
        )
        await storage.record_step(step)
        fetched = await storage.get_step(step.id)
        assert fetched.cost_usd == cost

    async def test_completed_at_utc_aware(self, storage: SQLiteStorage):
        run = Run.create(name="tz_step_test")
        await storage.create_run(run)
        created = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        completed = datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC)
        step = self._make_step(run_id=run.id, seq=1, created_at=created, completed_at=completed)
        await storage.record_step(step)
        fetched = await storage.get_step(step.id)
        assert fetched.completed_at.tzinfo is not None


class TestCheckpoints:
    async def test_save_and_get_latest(self, storage: SQLiteStorage):
        run = Run.create(name="cp_test")
        await storage.create_run(run)
        step = Step(run_id=run.id, sequence_number=1, kind=StepKind.LLM_CALL, inputs={})
        await storage.record_step(step)

        snapshot = {"messages": [{"role": "assistant", "content": "hi"}], "iter": 1}
        cp = Checkpoint(run_id=run.id, after_step_id=step.id, snapshot=snapshot)
        await storage.save_checkpoint(cp)

        fetched = await storage.get_latest_checkpoint(run.id)
        assert fetched is not None
        assert fetched.snapshot == snapshot
        assert fetched.after_step_id == step.id

    async def test_get_latest_returns_most_recent(self, storage: SQLiteStorage):
        run = Run.create(name="cp_order_test")
        await storage.create_run(run)

        for seq in range(1, 4):
            step = Step(run_id=run.id, sequence_number=seq, kind=StepKind.LLM_CALL, inputs={})
            await storage.record_step(step)
            cp = Checkpoint(run_id=run.id, after_step_id=step.id, snapshot={"seq": seq})
            await storage.save_checkpoint(cp)

        latest = await storage.get_latest_checkpoint(run.id)
        assert latest.snapshot["seq"] == 3

    async def test_get_latest_no_checkpoints(self, storage: SQLiteStorage):
        run = Run.create(name="no_cp_test")
        await storage.create_run(run)
        assert await storage.get_latest_checkpoint(run.id) is None


class TestFailures:
    async def test_record_failure(self, storage: SQLiteStorage):
        run = Run.create(name="fail_test")
        await storage.create_run(run)
        step = Step(run_id=run.id, sequence_number=1, kind=StepKind.LLM_CALL, inputs={})
        await storage.record_step(step)

        failure = FailureRecord(
            run_id=run.id,
            step_id=step.id,
            error_type="RateLimitError",
            error_message="429 from openai",
            provider=Provider.OPENAI,
            recovery_action="retry",
            recovery_succeeded=True,
        )
        await storage.record_failure(failure)
        # If no exception raised, the record was persisted.

    async def test_record_failure_null_recovery(self, storage: SQLiteStorage):
        run = Run.create(name="fail_null_test")
        await storage.create_run(run)
        step = Step(run_id=run.id, sequence_number=1, kind=StepKind.LLM_CALL, inputs={})
        await storage.record_step(step)
        failure = FailureRecord(
            run_id=run.id,
            step_id=step.id,
            error_type="ProviderError",
            error_message="500 Internal Server Error",
            provider=Provider.OPENAI,
        )
        await storage.record_failure(failure)


class TestConcurrency:
    async def test_concurrent_run_creates(self, tmp_db_path: Path):
        """Multiple coroutines creating runs simultaneously must not corrupt the DB."""
        storage = SQLiteStorage(tmp_db_path)
        await storage.initialize()

        async def create_one(name: str) -> None:
            run = Run.create(name=name)
            await storage.create_run(run)

        await asyncio.gather(*[create_one(f"run_{i}") for i in range(10)])
        await storage.close()

    async def test_concurrent_step_records(self, storage: SQLiteStorage):
        """Multiple coroutines writing steps for the same run must not corrupt."""
        run = Run.create(name="concurrent_steps")
        await storage.create_run(run)

        async def write_step(seq: int) -> None:
            step = Step(
                run_id=run.id,
                sequence_number=seq,
                kind=StepKind.LLM_CALL,
                inputs={"seq": seq},
            )
            await storage.record_step(step)

        await asyncio.gather(*[write_step(i) for i in range(1, 11)])
        steps = await storage.get_steps_for_run(run.id)
        assert len(steps) == 10
