"""TetheredAnthropic / AsyncTetheredAnthropic — Anthropic client wrappers.

Mirror the TetheredOpenAI pattern exactly. Every messages.create call is
intercepted, recorded to SQLite, and then forwarded to the real Anthropic
client. The return value is the unchanged Anthropic response object.

Usage (sync):
    from anthropic import Anthropic
    from tether import TetheredAnthropic, SQLiteStorage

    storage = SQLiteStorage("agent.db")
    await storage.initialize()

    client = Anthropic()  # uses ANTHROPIC_API_KEY from environment
    tethered = TetheredAnthropic(client=client, storage=storage, run_name="my_agent")

    response = tethered.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(response.content[0].text)

Usage (async):
    from anthropic import AsyncAnthropic
    from tether import AsyncTetheredAnthropic, SQLiteStorage

    tethered = AsyncTetheredAnthropic(client=AsyncAnthropic(), storage=storage, ...)
    response = await tethered.messages.create(model="claude-sonnet-4-6", ...)
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from decimal import Decimal
from threading import Lock
from typing import Any
from uuid import UUID

import anthropic

from tether.core.logging import get_logger
from tether.core.models import (
    FailureRecord,
    Provider,
    Run,
    RunStatus,
    Step,
    StepKind,
)
from tether.core.storage import SQLiteStorage
from tether.exceptions import CaptureError, RateLimitError

log = get_logger(__name__)

# Pricing in USD per 1K tokens: {model: (input_per_1k, output_per_1k)}
_PRICING: dict[str, tuple[Decimal, Decimal]] = {
    "claude-opus-4-6": (Decimal("0.015"), Decimal("0.075")),
    "claude-sonnet-4-6": (Decimal("0.003"), Decimal("0.015")),
    "claude-haiku-4-5": (Decimal("0.00025"), Decimal("0.00125")),
    "claude-haiku-4-5-20251001": (Decimal("0.00025"), Decimal("0.00125")),
    # Prior generation
    "claude-3-5-sonnet-20241022": (Decimal("0.003"), Decimal("0.015")),
    "claude-3-5-haiku-20241022": (Decimal("0.001"), Decimal("0.005")),
    "claude-3-opus-20240229": (Decimal("0.015"), Decimal("0.075")),
    "claude-3-haiku-20240307": (Decimal("0.00025"), Decimal("0.00125")),
}
_DEFAULT_PRICING = (Decimal("0.003"), Decimal("0.015"))  # sonnet-tier fallback


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    """Calculate estimated cost in USD for an Anthropic messages call."""
    if model not in _PRICING:
        log.warning(
            "unknown_model_pricing",
            model=model,
            fallback_input=str(_DEFAULT_PRICING[0]),
            fallback_output=str(_DEFAULT_PRICING[1]),
            advice="Add this model to tether/capture/anthropic.py _PRICING table.",
        )
    input_rate, output_rate = _PRICING.get(model, _DEFAULT_PRICING)
    return (Decimal(input_tokens) / Decimal(1000)) * input_rate + (
        Decimal(output_tokens) / Decimal(1000)
    ) * output_rate


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------


class _MessagesNamespace:
    """Mirrors anthropic.resources.Messages for the tethered sync client."""

    def __init__(self, owner: TetheredAnthropic) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        """Record and forward a messages.create request."""
        return self._owner._run_sync(self._owner._create_message(**kwargs))


class TetheredAnthropic:
    """Drop-in Anthropic client wrapper that records every LLM call to SQLite.

    Use exactly as you would ``anthropic.Anthropic`` — the return values and
    exceptions are identical. Pair with CostGuard ``POST /replay`` to compare
    production Claude traffic against alternate models with bootstrap CIs.

    Args:
        client: An initialized ``anthropic.Anthropic`` instance.
        storage: An initialized ``SQLiteStorage`` instance.
        run_name: Human-readable label for this agent run.
        run_id: Optional existing run UUID to resume.

    Example:
        tethered = TetheredAnthropic(
            client=Anthropic(),
            storage=storage,
            run_name="research_agent_v1",
        )
        response = tethered.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Summarise attention mechanisms"}],
        )
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        storage: SQLiteStorage,
        run_name: str,
        run_id: UUID | None = None,
    ) -> None:
        self._client = client
        self._storage = storage
        self._run_name = run_name
        self._run_id: UUID | None = run_id
        self._sequence_lock = Lock()
        self._sequence_counter: int = 0
        self.messages = _MessagesNamespace(self)

    @property
    def run_id(self) -> UUID | None:
        """The UUID of the current run, or None if no call has been made yet."""
        return self._run_id

    def _next_sequence_number(self) -> int:
        with self._sequence_lock:
            self._sequence_counter += 1
            return self._sequence_counter

    def _run_sync(self, coro: Any) -> Any:
        """Execute a coroutine from synchronous context."""
        try:
            asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            return asyncio.run(coro)

    async def _ensure_run(self) -> UUID:
        if self._run_id is not None:
            return self._run_id
        run = Run.create(name=self._run_name)
        await self._storage.create_run(run)
        await self._storage.update_run_status(run.id, RunStatus.RUNNING)
        self._run_id = run.id
        log.info("run_created", run_id=str(run.id), name=self._run_name)
        return run.id

    async def _create_message(self, **kwargs: Any) -> Any:
        """Core async implementation for messages.create."""
        run_id = await self._ensure_run()
        seq = self._next_sequence_number()
        model: str = kwargs.get("model", "unknown")

        step = Step(
            run_id=run_id,
            sequence_number=seq,
            kind=StepKind.LLM_CALL,
            provider=Provider.ANTHROPIC,
            model=model,
            inputs=dict(kwargs),
        )
        await self._storage.record_step(step)
        log.info(
            "llm_call_started",
            run_id=str(run_id),
            step_id=str(step.id),
            model=model,
            seq=seq,
        )

        start = time.perf_counter()
        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.RateLimitError as exc:
            await self._record_provider_failure(
                run_id=run_id, step=step, exc=exc, error_type="RateLimitError"
            )
            raise RateLimitError(
                f"Provider rate limit hit on run {run_id} step {step.id} (model={model}).",
                original_error=exc,
            ) from exc
        except anthropic.APIError as exc:
            await self._record_provider_failure(
                run_id=run_id, step=step, exc=exc, error_type=type(exc).__name__
            )
            raise CaptureError(
                f"Provider error on run {run_id} step {step.id} (model={model}): {exc}",
                original_error=exc,
            ) from exc

        latency_ms = (time.perf_counter() - start) * 1000
        completed_at = datetime.now(UTC)

        usage = getattr(response, "usage", None)
        input_tokens: int | None = getattr(usage, "input_tokens", None)
        output_tokens: int | None = getattr(usage, "output_tokens", None)

        cost: Decimal | None = None
        if input_tokens is not None and output_tokens is not None:
            cost = _compute_cost(model, input_tokens, output_tokens)

        try:
            outputs = response.model_dump()
        except AttributeError:
            outputs = {"raw": str(response)}

        completed_step = step.model_copy(
            update={
                "outputs": outputs,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "latency_ms": latency_ms,
                "completed_at": completed_at,
            }
        )
        await self._storage.record_step(completed_step)

        log.info(
            "llm_call_completed",
            run_id=str(run_id),
            step_id=str(step.id),
            model=model,
            seq=seq,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=str(cost) if cost else None,
            latency_ms=round(latency_ms, 1),
        )
        return response

    async def _record_provider_failure(
        self,
        run_id: UUID,
        step: Step,
        exc: Exception,
        error_type: str,
    ) -> None:
        error_dict = {"type": error_type, "message": str(exc)}
        error_step = step.model_copy(
            update={"error": error_dict, "completed_at": datetime.now(UTC)}
        )
        await self._storage.record_step(error_step)
        failure = FailureRecord(
            run_id=run_id,
            step_id=step.id,
            error_type=error_type,
            error_message=str(exc),
            provider=Provider.ANTHROPIC,
        )
        await self._storage.record_failure(failure)
        log.error(
            "llm_call_failed",
            run_id=str(run_id),
            step_id=str(step.id),
            error_type=error_type,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------


class _AsyncMessagesNamespace:
    """Mirrors anthropic.resources.Messages for the tethered async client."""

    def __init__(self, owner: AsyncTetheredAnthropic) -> None:
        self._owner = owner

    async def create(self, **kwargs: Any) -> Any:
        """Record and forward a messages.create request (async)."""
        return await self._owner._create_message(**kwargs)


class AsyncTetheredAnthropic:
    """Async drop-in Anthropic client wrapper that records every call to SQLite.

    Use inside async code with ``await tethered.messages.create(...)``.

    Args:
        client: An initialized ``anthropic.AsyncAnthropic`` instance.
        storage: An initialized ``SQLiteStorage`` instance.
        run_name: Human-readable label for this agent run.
        run_id: Optional existing run UUID to resume.
    """

    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        storage: SQLiteStorage,
        run_name: str,
        run_id: UUID | None = None,
    ) -> None:
        self._client = client
        self._storage = storage
        self._run_name = run_name
        self._run_id: UUID | None = run_id
        self._sequence_lock = Lock()
        self._sequence_counter: int = 0
        self.messages = _AsyncMessagesNamespace(self)

    @property
    def run_id(self) -> UUID | None:
        """The UUID of the current run, or None if no call has been made yet."""
        return self._run_id

    def _next_sequence_number(self) -> int:
        with self._sequence_lock:
            self._sequence_counter += 1
            return self._sequence_counter

    async def _ensure_run(self) -> UUID:
        if self._run_id is not None:
            return self._run_id
        run = Run.create(name=self._run_name)
        await self._storage.create_run(run)
        await self._storage.update_run_status(run.id, RunStatus.RUNNING)
        self._run_id = run.id
        log.info("run_created", run_id=str(run.id), name=self._run_name)
        return run.id

    async def _create_message(self, **kwargs: Any) -> Any:
        """Core async implementation — awaits the AsyncAnthropic client directly."""
        run_id = await self._ensure_run()
        seq = self._next_sequence_number()
        model: str = kwargs.get("model", "unknown")

        step = Step(
            run_id=run_id,
            sequence_number=seq,
            kind=StepKind.LLM_CALL,
            provider=Provider.ANTHROPIC,
            model=model,
            inputs=dict(kwargs),
        )
        await self._storage.record_step(step)
        log.info(
            "llm_call_started",
            run_id=str(run_id),
            step_id=str(step.id),
            model=model,
            seq=seq,
        )

        start = time.perf_counter()
        try:
            response = await self._client.messages.create(**kwargs)
        except anthropic.RateLimitError as exc:
            await self._record_provider_failure(
                run_id=run_id, step=step, exc=exc, error_type="RateLimitError"
            )
            raise RateLimitError(
                f"Provider rate limit hit on run {run_id} step {step.id} (model={model}).",
                original_error=exc,
            ) from exc
        except anthropic.APIError as exc:
            await self._record_provider_failure(
                run_id=run_id, step=step, exc=exc, error_type=type(exc).__name__
            )
            raise CaptureError(
                f"Provider error on run {run_id} step {step.id} (model={model}): {exc}",
                original_error=exc,
            ) from exc

        latency_ms = (time.perf_counter() - start) * 1000
        completed_at = datetime.now(UTC)

        usage = getattr(response, "usage", None)
        input_tokens: int | None = getattr(usage, "input_tokens", None)
        output_tokens: int | None = getattr(usage, "output_tokens", None)

        cost: Decimal | None = None
        if input_tokens is not None and output_tokens is not None:
            cost = _compute_cost(model, input_tokens, output_tokens)

        try:
            outputs = response.model_dump()
        except AttributeError:
            outputs = {"raw": str(response)}

        completed_step = step.model_copy(
            update={
                "outputs": outputs,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "latency_ms": latency_ms,
                "completed_at": completed_at,
            }
        )
        await self._storage.record_step(completed_step)

        log.info(
            "llm_call_completed",
            run_id=str(run_id),
            step_id=str(step.id),
            model=model,
            seq=seq,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=str(cost) if cost else None,
            latency_ms=round(latency_ms, 1),
        )
        return response

    async def _record_provider_failure(
        self,
        run_id: UUID,
        step: Step,
        exc: Exception,
        error_type: str,
    ) -> None:
        error_dict = {"type": error_type, "message": str(exc)}
        error_step = step.model_copy(
            update={"error": error_dict, "completed_at": datetime.now(UTC)}
        )
        await self._storage.record_step(error_step)
        failure = FailureRecord(
            run_id=run_id,
            step_id=step.id,
            error_type=error_type,
            error_message=str(exc),
            provider=Provider.ANTHROPIC,
        )
        await self._storage.record_failure(failure)
        log.error(
            "llm_call_failed",
            run_id=str(run_id),
            step_id=str(step.id),
            error_type=error_type,
            error=str(exc),
        )
