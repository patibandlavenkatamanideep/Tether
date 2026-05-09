"""Pydantic v2 data models for Tether's core domain.

All models use UTC-aware datetimes, Decimal for monetary values, and are
frozen by default to prevent accidental mutation after creation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(UTC)


def _new_uuid() -> UUID:
    """Return a new random UUID."""
    return uuid4()


class RunStatus(StrEnum):
    """Lifecycle state of an agent run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CRASHED = "crashed"


class StepKind(StrEnum):
    """Category of work performed in a single step."""

    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    STATE_MUTATION = "state_mutation"


class Provider(StrEnum):
    """LLM provider that executed a step."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class Run(BaseModel):
    """An agent run — the top-level durable execution unit.

    A Run groups all Steps, Checkpoints, and FailureRecords produced by a
    single invocation of an agent. Runs survive crashes: Tether replays from
    the latest Checkpoint when a run resumes.

    Attributes:
        id: Globally unique run identifier, auto-generated if not supplied.
        name: Human-readable label (e.g. "research_agent_run_1").
        status: Current lifecycle state of the run.
        created_at: UTC timestamp when the run was first created.
        updated_at: UTC timestamp of the last status change.
        metadata: Arbitrary caller-supplied key-value pairs (tags, version info, etc.).
        config_hash: SHA-256 of the run's configuration snapshot, for drift detection.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=_new_uuid, description="Unique run identifier.")
    name: str = Field(description="Human-readable run label.")
    status: RunStatus = Field(default=RunStatus.PENDING, description="Current lifecycle state.")
    created_at: datetime = Field(default_factory=_utcnow, description="UTC creation timestamp.")
    updated_at: datetime = Field(
        default_factory=_utcnow, description="UTC timestamp of last status change."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Caller-supplied tags and annotations."
    )
    config_hash: str = Field(default="", description="SHA-256 of the run config snapshot.")

    @field_validator("created_at", "updated_at", mode="after")
    @classmethod
    def _require_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware (UTC).")
        return v

    @classmethod
    def create(
        cls,
        name: str,
        metadata: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Run:
        """Factory that auto-hashes the config and sets sensible defaults.

        Args:
            name: Human-readable label for this run.
            metadata: Optional caller-supplied annotations.
            config: Optional config dict to hash for drift detection.

        Returns:
            A new Run instance in PENDING status.
        """
        config_hash = ""
        if config:
            config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        return cls(name=name, metadata=metadata or {}, config_hash=config_hash)

    def with_status(self, status: RunStatus) -> Run:
        """Return a copy of this Run with an updated status and updated_at.

        Args:
            status: The new lifecycle state.

        Returns:
            A new Run instance (frozen model — no mutation in place).
        """
        return self.model_copy(update={"status": status, "updated_at": _utcnow()})


class Step(BaseModel):
    """A single recorded operation within a Run.

    Each LLM call, tool invocation, or state mutation becomes one Step.
    Steps are immutable once completed_at is set.

    Attributes:
        id: Unique step identifier.
        run_id: Parent Run this step belongs to.
        sequence_number: Monotonically increasing integer per run (1-based).
        kind: Category of operation performed.
        provider: LLM provider, if this is an llm_call step.
        model: Provider model identifier (e.g. "gpt-4o").
        inputs: Full request payload sent to the provider.
        outputs: Full response payload received from the provider.
        input_tokens: Prompt token count reported by the provider.
        output_tokens: Completion token count reported by the provider.
        cost_usd: Estimated cost in USD as a Decimal (avoids float drift).
        latency_ms: Wall-clock time from request to response in milliseconds.
        error: Serialized exception info if the step failed.
        created_at: UTC timestamp when the step was initiated.
        completed_at: UTC timestamp when the step finished (or None if in-flight).
    """

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=_new_uuid, description="Unique step identifier.")
    run_id: UUID = Field(description="Parent run identifier.")
    sequence_number: int = Field(ge=1, description="Monotonically increasing per run, 1-based.")
    kind: StepKind = Field(description="Category of operation.")
    provider: Provider | None = Field(default=None, description="LLM provider for llm_call steps.")
    model: str | None = Field(default=None, description="Provider model identifier.")
    inputs: dict[str, Any] = Field(default_factory=dict, description="Full request payload.")
    outputs: dict[str, Any] | None = Field(default=None, description="Full response payload.")
    input_tokens: int | None = Field(default=None, description="Prompt token count from provider.")
    output_tokens: int | None = Field(
        default=None, description="Completion token count from provider."
    )
    cost_usd: Decimal | None = Field(
        default=None, description="Estimated cost in USD (Decimal, not float)."
    )
    latency_ms: float | None = Field(
        default=None, description="Wall-clock request latency in milliseconds."
    )
    error: dict[str, Any] | None = Field(
        default=None, description="Serialized exception info if step failed."
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="UTC timestamp when step was initiated."
    )
    completed_at: datetime | None = Field(
        default=None, description="UTC timestamp when step finished."
    )

    @field_validator("created_at", mode="after")
    @classmethod
    def _require_utc_created(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("created_at must be timezone-aware (UTC).")
        return v

    @field_validator("completed_at", mode="after")
    @classmethod
    def _require_utc_completed(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            raise ValueError("completed_at must be timezone-aware (UTC).")
        return v

    @model_validator(mode="after")
    def _completed_after_created(self) -> Step:
        if self.completed_at is not None and self.completed_at < self.created_at:
            raise ValueError("completed_at cannot be before created_at.")
        return self


class Checkpoint(BaseModel):
    """A point-in-time snapshot of an agent's state after a given step.

    Tether serializes checkpoints to SQLite so runs can resume from the
    most recent checkpoint after a crash.

    Attributes:
        id: Unique checkpoint identifier.
        run_id: Parent run this checkpoint belongs to.
        after_step_id: The Step that was just completed when this was saved.
        snapshot: JSON-serializable agent state (arbitrary dict).
        created_at: UTC timestamp when the checkpoint was saved.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=_new_uuid, description="Unique checkpoint identifier.")
    run_id: UUID = Field(description="Parent run identifier.")
    after_step_id: UUID = Field(
        description="The step that was just completed when this checkpoint was saved."
    )
    snapshot: dict[str, Any] = Field(
        description="JSON-serializable agent state at this point in time."
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="UTC timestamp when checkpoint was saved."
    )

    @field_validator("created_at", mode="after")
    @classmethod
    def _require_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("created_at must be timezone-aware (UTC).")
        return v


class FailureRecord(BaseModel):
    """A record of a provider failure and what Tether did about it.

    Attributes:
        id: Unique failure record identifier.
        run_id: Parent run where the failure occurred.
        step_id: The step that was in-flight when the failure occurred.
        error_type: Exception class name (e.g. "RateLimitError").
        error_message: Human-readable error message from the provider.
        provider: The provider that returned the error.
        occurred_at: UTC timestamp when the error was observed.
        recovery_action: What Tether attempted ("retry", "swap", "pause", "abort").
        recovery_succeeded: Whether the recovery action resolved the failure.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=_new_uuid, description="Unique failure record identifier.")
    run_id: UUID = Field(description="Parent run identifier.")
    step_id: UUID = Field(description="The step that was in-flight during the failure.")
    error_type: str = Field(description="Exception class name.")
    error_message: str = Field(description="Human-readable error message from the provider.")
    provider: Provider = Field(description="Provider that returned the error.")
    occurred_at: datetime = Field(
        default_factory=_utcnow, description="UTC timestamp when the error was observed."
    )
    recovery_action: str | None = Field(
        default=None,
        description='Recovery strategy attempted: "retry", "swap", "pause", or "abort".',
    )
    recovery_succeeded: bool | None = Field(
        default=None, description="Whether the recovery action resolved the failure."
    )

    @field_validator("occurred_at", mode="after")
    @classmethod
    def _require_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware (UTC).")
        return v
