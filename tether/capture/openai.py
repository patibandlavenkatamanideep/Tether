"""TetheredOpenAI — drop-in OpenAI client wrapper with durable capture.

Every chat.completions.create call is intercepted, recorded to SQLite,
and then forwarded to the real OpenAI client. The return value is the
unchanged OpenAI response object, so existing code requires zero changes.

TODO(v0.2): Add async client support (AsyncTetheredOpenAI).
TODO(v0.2): Add streaming response capture (buffer chunks, store full response,
            return a re-yielding wrapper that is transparent to callers).

Usage:
    from openai import OpenAI
    from tether import TetheredOpenAI, SQLiteStorage

    storage = SQLiteStorage("agent.db")
    await storage.initialize()

    client = OpenAI(api_key="...")
    tethered = TetheredOpenAI(client=client, storage=storage, run_name="my_agent")

    response = tethered.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(response.choices[0].message.content)
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from decimal import Decimal
from threading import Lock
from typing import Any
from uuid import UUID

import openai

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
    "gpt-4o": (Decimal("0.0025"), Decimal("0.01")),
    "gpt-4o-mini": (Decimal("0.00015"), Decimal("0.0006")),
    "gpt-4-turbo": (Decimal("0.01"), Decimal("0.03")),
}
_DEFAULT_PRICING = (Decimal("0.01"), Decimal("0.03"))


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    """Calculate estimated cost in USD for a completion.

    Args:
        model: OpenAI model identifier string.
        input_tokens: Number of prompt tokens consumed.
        output_tokens: Number of completion tokens generated.

    Returns:
        Estimated cost as a Decimal (never float — avoids precision drift).
    """
    if model not in _PRICING:
        log.warning(
            "unknown_model_pricing",
            model=model,
            fallback_input=str(_DEFAULT_PRICING[0]),
            fallback_output=str(_DEFAULT_PRICING[1]),
            advice="Add this model to tether/capture/openai.py _PRICING table.",
        )
    input_rate, output_rate = _PRICING.get(model, _DEFAULT_PRICING)
    return (Decimal(input_tokens) / Decimal(1000)) * input_rate + (
        Decimal(output_tokens) / Decimal(1000)
    ) * output_rate


class _CompletionsNamespace:
    """Mirrors openai.resources.chat.completions for the tethered client.

    This is an internal class — users access it via ``TetheredOpenAI.chat.completions``.
    """

    def __init__(self, owner: TetheredOpenAI) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        """Record and forward a chat completion request.

        Accepts the same keyword arguments as ``openai.OpenAI().chat.completions.create``.
        The response is returned unchanged.

        Args:
            **kwargs: All arguments forwarded verbatim to the underlying OpenAI client.

        Returns:
            The raw ``ChatCompletion`` object from the OpenAI SDK.

        Raises:
            RateLimitError: If the provider returns a 429 response.
            CaptureError: If the provider returns any other error.
        """
        return self._owner._run_sync(self._owner._create_completion(**kwargs))


class _ChatNamespace:
    """Mirrors ``openai.OpenAI().chat`` for the tethered client."""

    def __init__(self, owner: TetheredOpenAI) -> None:
        self.completions = _CompletionsNamespace(owner)


class TetheredOpenAI:
    """Drop-in OpenAI client wrapper that records every LLM call to SQLite.

    Wrap your existing ``openai.OpenAI`` instance and use ``TetheredOpenAI``
    exactly as you would the original — the return values and exceptions are
    identical. Tether adds zero latency from the caller's perspective beyond
    the SQLite write (typically < 1 ms).

    Args:
        client: An initialized ``openai.OpenAI`` instance.
        storage: An initialized ``SQLiteStorage`` instance.
        run_name: Human-readable label for this agent run.
        run_id: Optional existing run UUID to resume. If None, a new Run is
                created on the first call.

    Example:
        storage = SQLiteStorage("agent.db")
        await storage.initialize()

        tethered = TetheredOpenAI(
            client=OpenAI(api_key="..."),
            storage=storage,
            run_name="research_agent_v1",
        )
        response = tethered.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Summarise quantum entanglement"}],
        )
    """

    def __init__(
        self,
        client: openai.OpenAI,
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
        self._run_initialized = False
        self.chat = _ChatNamespace(self)

    @property
    def run_id(self) -> UUID | None:
        """The UUID of the current run, or None if no call has been made yet."""
        return self._run_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_sequence_number(self) -> int:
        """Return the next monotonic sequence number, thread-safely."""
        with self._sequence_lock:
            self._sequence_counter += 1
            return self._sequence_counter

    def _run_sync(self, coro: Any) -> Any:
        """Execute a coroutine from synchronous context.

        Tries to use an existing running event loop (Jupyter / nested async),
        falling back to asyncio.run for plain scripts.
        """
        try:
            asyncio.get_running_loop()
            # We are inside an async context — run in a new thread to avoid
            # blocking the event loop. This is safe for the storage I/O pattern.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            return asyncio.run(coro)

    async def _ensure_run(self) -> UUID:
        """Create and persist the Run on the first call, then return its ID.

        Subsequent calls return the cached ID immediately.
        """
        if self._run_id is not None:
            return self._run_id

        run = Run.create(name=self._run_name)
        await self._storage.create_run(run)
        await self._storage.update_run_status(run.id, RunStatus.RUNNING)
        self._run_id = run.id
        log.info("run_created", run_id=str(run.id), name=self._run_name)
        return run.id

    async def _create_completion(self, **kwargs: Any) -> Any:
        """Core async implementation for chat.completions.create.

        Steps:
        1. Ensure the Run exists.
        2. Assign a monotonic sequence number.
        3. Record the Step (inputs captured, outputs=None initially).
        4. Call the real OpenAI client.
        5. Update the Step with outputs, tokens, cost, and latency.
        6. Return the raw response.

        If the provider call raises, the failure is recorded and the
        original exception is re-raised.

        Args:
            **kwargs: Forwarded verbatim to ``openai.OpenAI().chat.completions.create``.

        Returns:
            The raw ``ChatCompletion`` object from the OpenAI SDK.

        Raises:
            RateLimitError: Wrapping a 429 from the provider.
            CaptureError: Wrapping any other provider error.
        """
        run_id = await self._ensure_run()
        seq = self._next_sequence_number()
        model: str = kwargs.get("model", "unknown")

        step = Step(
            run_id=run_id,
            sequence_number=seq,
            kind=StepKind.LLM_CALL,
            provider=Provider.OPENAI,
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
            response = self._client.chat.completions.create(**kwargs)
        except openai.RateLimitError as exc:
            await self._record_provider_failure(
                run_id=run_id,
                step=step,
                exc=exc,
                error_type="RateLimitError",
            )
            raise RateLimitError(
                f"Provider rate limit hit on run {run_id} step {step.id} (model={model}). "
                "Enable recovery to retry automatically.",
                original_error=exc,
            ) from exc
        except openai.APIError as exc:
            await self._record_provider_failure(
                run_id=run_id,
                step=step,
                exc=exc,
                error_type=type(exc).__name__,
            )
            raise CaptureError(
                f"Provider error on run {run_id} step {step.id} (model={model}): {exc}",
                original_error=exc,
            ) from exc

        latency_ms = (time.perf_counter() - start) * 1000
        completed_at = datetime.now(UTC)

        # Extract token usage from the response.
        usage = getattr(response, "usage", None)
        input_tokens: int | None = getattr(usage, "prompt_tokens", None)
        output_tokens: int | None = getattr(usage, "completion_tokens", None)

        cost: Decimal | None = None
        if input_tokens is not None and output_tokens is not None:
            cost = _compute_cost(model, input_tokens, output_tokens)

        # Serialize the response to a plain dict for storage.
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
        """Persist a FailureRecord for a provider exception.

        Also updates the step with the error details so the run history
        reflects what went wrong.

        Args:
            run_id: The parent Run's UUID.
            step: The Step that was in-flight.
            exc: The exception raised by the provider.
            error_type: Human-readable exception class name.
        """
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
            provider=Provider.OPENAI,
        )
        await self._storage.record_failure(failure)

        log.error(
            "llm_call_failed",
            run_id=str(run_id),
            step_id=str(step.id),
            error_type=error_type,
            error=str(exc),
        )
