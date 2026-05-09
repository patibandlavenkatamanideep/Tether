"""Tests for tether.core.models."""

from __future__ import annotations

from datetime import UTC, datetime, timezone
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from tether.core.models import (
    Checkpoint,
    FailureRecord,
    Provider,
    Run,
    RunStatus,
    Step,
    StepKind,
)


class TestRun:
    def test_auto_generates_id(self):
        run = Run(name="test")
        assert isinstance(run.id, UUID)

    def test_explicit_id_preserved(self):
        uid = uuid4()
        run = Run(id=uid, name="test")
        assert run.id == uid

    def test_default_status_is_pending(self):
        run = Run(name="test")
        assert run.status == RunStatus.PENDING

    def test_created_at_is_utc_aware(self):
        run = Run(name="test")
        assert run.created_at.tzinfo is not None
        assert run.created_at.tzinfo == UTC or run.created_at.utcoffset().total_seconds() == 0

    def test_updated_at_is_utc_aware(self):
        run = Run(name="test")
        assert run.updated_at.tzinfo is not None

    def test_rejects_naive_datetime(self):
        with pytest.raises(ValidationError):
            Run(name="test", created_at=datetime(2026, 1, 1))

    def test_create_factory_sets_config_hash(self):
        run = Run.create(name="test", config={"model": "gpt-4o"})
        assert len(run.config_hash) == 64  # sha256 hex

    def test_create_factory_empty_config_has_empty_hash(self):
        run = Run.create(name="test")
        assert run.config_hash == ""

    def test_with_status_returns_new_instance(self):
        run = Run.create(name="test")
        updated = run.with_status(RunStatus.RUNNING)
        assert updated.status == RunStatus.RUNNING
        assert run.status == RunStatus.PENDING  # original unchanged

    def test_with_status_updates_updated_at(self):
        run = Run.create(name="test")
        updated = run.with_status(RunStatus.COMPLETED)
        assert updated.updated_at >= run.updated_at

    def test_metadata_preserved(self):
        meta = {"agent": "researcher", "version": "1"}
        run = Run.create(name="test", metadata=meta)
        assert run.metadata == meta

    def test_frozen_prevents_mutation(self):
        run = Run(name="test")
        with pytest.raises(ValidationError):
            run.name = "other"  # type: ignore[misc]


class TestStep:
    def _make_step(self, **kwargs) -> Step:
        defaults = dict(
            run_id=uuid4(),
            sequence_number=1,
            kind=StepKind.LLM_CALL,
        )
        defaults.update(kwargs)
        return Step(**defaults)

    def test_auto_generates_id(self):
        step = self._make_step()
        assert isinstance(step.id, UUID)

    def test_sequence_number_must_be_positive(self):
        with pytest.raises(ValidationError):
            self._make_step(sequence_number=0)

    def test_created_at_is_utc_aware(self):
        step = self._make_step()
        assert step.created_at.tzinfo is not None

    def test_rejects_naive_created_at(self):
        with pytest.raises(ValidationError):
            self._make_step(created_at=datetime(2026, 1, 1))

    def test_rejects_completed_before_created(self):
        created = datetime(2026, 1, 2, tzinfo=UTC)
        completed = datetime(2026, 1, 1, tzinfo=UTC)
        with pytest.raises(ValidationError):
            self._make_step(created_at=created, completed_at=completed)

    def test_cost_usd_is_decimal(self):
        step = self._make_step(cost_usd=Decimal("0.001234"))
        assert isinstance(step.cost_usd, Decimal)

    def test_provider_and_model_optional(self):
        step = self._make_step(kind=StepKind.STATE_MUTATION)
        assert step.provider is None
        assert step.model is None

    def test_inputs_defaults_to_empty_dict(self):
        step = self._make_step()
        assert step.inputs == {}

    def test_error_dict_stored(self):
        step = self._make_step(error={"type": "RateLimitError", "message": "too many"})
        assert step.error["type"] == "RateLimitError"


class TestCheckpoint:
    def test_round_trips_complex_snapshot(self):
        snapshot = {
            "messages": [{"role": "user", "content": "hello"}],
            "iteration": 3,
            "context": {"urls": ["http://example.com"], "score": 0.9},
        }
        cp = Checkpoint(
            run_id=uuid4(),
            after_step_id=uuid4(),
            snapshot=snapshot,
        )
        assert cp.snapshot == snapshot

    def test_auto_generates_id(self):
        cp = Checkpoint(run_id=uuid4(), after_step_id=uuid4(), snapshot={})
        assert isinstance(cp.id, UUID)

    def test_created_at_is_utc_aware(self):
        cp = Checkpoint(run_id=uuid4(), after_step_id=uuid4(), snapshot={})
        assert cp.created_at.tzinfo is not None

    def test_rejects_naive_created_at(self):
        with pytest.raises(ValidationError):
            Checkpoint(
                run_id=uuid4(),
                after_step_id=uuid4(),
                snapshot={},
                created_at=datetime(2026, 1, 1),
            )


class TestFailureRecord:
    def _make_failure(self, **kwargs) -> FailureRecord:
        defaults = dict(
            run_id=uuid4(),
            step_id=uuid4(),
            error_type="RateLimitError",
            error_message="429 Too Many Requests",
            provider=Provider.OPENAI,
        )
        defaults.update(kwargs)
        return FailureRecord(**defaults)

    def test_auto_generates_id(self):
        f = self._make_failure()
        assert isinstance(f.id, UUID)

    def test_links_to_run_and_step(self):
        run_id = uuid4()
        step_id = uuid4()
        f = self._make_failure(run_id=run_id, step_id=step_id)
        assert f.run_id == run_id
        assert f.step_id == step_id

    def test_recovery_action_optional(self):
        f = self._make_failure()
        assert f.recovery_action is None
        assert f.recovery_succeeded is None

    def test_recovery_fields_set(self):
        f = self._make_failure(recovery_action="retry", recovery_succeeded=True)
        assert f.recovery_action == "retry"
        assert f.recovery_succeeded is True

    def test_occurred_at_is_utc_aware(self):
        f = self._make_failure()
        assert f.occurred_at.tzinfo is not None
