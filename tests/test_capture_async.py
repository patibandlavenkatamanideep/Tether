"""Tests for AsyncTetheredOpenAI and AsyncTetheredAnthropic."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import anthropic
import openai
import pytest

from tether.capture.anthropic import AsyncTetheredAnthropic
from tether.capture.openai import AsyncTetheredOpenAI
from tether.core.models import Provider, RunStatus, StepKind
from tether.core.storage import SQLiteStorage
from tether.exceptions import CaptureError, RateLimitError


# ---------------------------------------------------------------------------
# AsyncTetheredOpenAI
# ---------------------------------------------------------------------------


class TestAsyncTetheredOpenAICreatesRun:
    async def test_creates_run_on_first_call(
        self, storage: SQLiteStorage, mock_async_openai_client: MagicMock
    ):
        tethered = AsyncTetheredOpenAI(
            client=mock_async_openai_client,
            storage=storage,
            run_name="async_openai_agent",
        )
        await tethered.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert tethered._run_id is not None
        run = await storage.get_run(tethered._run_id)
        assert run is not None
        assert run.name == "async_openai_agent"
        assert run.status == RunStatus.RUNNING

    async def test_does_not_create_duplicate_runs(
        self, storage: SQLiteStorage, mock_async_openai_client: MagicMock
    ):
        tethered = AsyncTetheredOpenAI(
            client=mock_async_openai_client,
            storage=storage,
            run_name="no_dup",
        )
        await tethered.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "First"}],
        )
        first_run_id = tethered._run_id
        await tethered.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Second"}],
        )
        assert tethered._run_id == first_run_id

    async def test_run_id_none_before_first_call(
        self, storage: SQLiteStorage, mock_async_openai_client: MagicMock
    ):
        tethered = AsyncTetheredOpenAI(
            client=mock_async_openai_client,
            storage=storage,
            run_name="pre_call",
        )
        assert tethered.run_id is None


class TestAsyncOpenAICompletionCapture:
    async def test_records_provider_as_openai(
        self, storage: SQLiteStorage, mock_async_openai_client: MagicMock
    ):
        tethered = AsyncTetheredOpenAI(
            client=mock_async_openai_client,
            storage=storage,
            run_name="provider_test",
        )
        await tethered.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
        )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert steps[0].provider == Provider.OPENAI
        assert steps[0].kind == StepKind.LLM_CALL

    async def test_records_token_usage(
        self, storage: SQLiteStorage, mock_async_openai_client: MagicMock
    ):
        tethered = AsyncTetheredOpenAI(
            client=mock_async_openai_client,
            storage=storage,
            run_name="tokens_test",
        )
        await tethered.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
        )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert steps[0].input_tokens == 15
        assert steps[0].output_tokens == 10

    async def test_records_cost(
        self, storage: SQLiteStorage, mock_async_openai_client: MagicMock
    ):
        tethered = AsyncTetheredOpenAI(
            client=mock_async_openai_client,
            storage=storage,
            run_name="cost_test",
        )
        await tethered.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
        )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert steps[0].cost_usd is not None
        assert isinstance(steps[0].cost_usd, Decimal)

    async def test_returns_original_response(
        self, storage: SQLiteStorage, mock_async_openai_client: MagicMock
    ):
        tethered = AsyncTetheredOpenAI(
            client=mock_async_openai_client,
            storage=storage,
            run_name="passthru_test",
        )
        response = await tethered.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert response.choices[0].message.content == "Hello! I am a mock async LLM response."

    async def test_monotonic_sequence_numbers(
        self, storage: SQLiteStorage, mock_async_openai_client: MagicMock
    ):
        tethered = AsyncTetheredOpenAI(
            client=mock_async_openai_client,
            storage=storage,
            run_name="seq_test",
        )
        for _ in range(3):
            await tethered.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "ping"}],
            )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert [s.sequence_number for s in steps] == [1, 2, 3]


class TestAsyncOpenAIErrorHandling:
    async def test_rate_limit_error_recorded(
        self, storage: SQLiteStorage, mock_async_openai_client: MagicMock
    ):
        mock_async_openai_client.chat.completions.create = AsyncMock(
            side_effect=openai.RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429, headers={}),
                body={"error": {"message": "Rate limit exceeded"}},
            )
        )
        tethered = AsyncTetheredOpenAI(
            client=mock_async_openai_client,
            storage=storage,
            run_name="rate_limit_test",
        )
        with pytest.raises(RateLimitError):
            await tethered.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hi"}],
            )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert steps[0].error is not None
        assert steps[0].error["type"] == "RateLimitError"

    async def test_api_error_recorded_and_reraised(
        self, storage: SQLiteStorage, mock_async_openai_client: MagicMock
    ):
        mock_async_openai_client.chat.completions.create = AsyncMock(
            side_effect=openai.APIStatusError(
                message="Server error",
                response=MagicMock(status_code=500, headers={}),
                body={"error": {"message": "Server error"}},
            )
        )
        tethered = AsyncTetheredOpenAI(
            client=mock_async_openai_client,
            storage=storage,
            run_name="api_error_test",
        )
        with pytest.raises(CaptureError):
            await tethered.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hi"}],
            )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert steps[0].error is not None


# ---------------------------------------------------------------------------
# AsyncTetheredAnthropic
# ---------------------------------------------------------------------------


class TestAsyncTetheredAnthropicCreatesRun:
    async def test_creates_run_on_first_call(
        self, storage: SQLiteStorage, mock_async_anthropic_client: MagicMock
    ):
        tethered = AsyncTetheredAnthropic(
            client=mock_async_anthropic_client,
            storage=storage,
            run_name="async_claude_agent",
        )
        await tethered.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert tethered._run_id is not None
        run = await storage.get_run(tethered._run_id)
        assert run is not None
        assert run.name == "async_claude_agent"
        assert run.status == RunStatus.RUNNING

    async def test_does_not_create_duplicate_runs(
        self, storage: SQLiteStorage, mock_async_anthropic_client: MagicMock
    ):
        tethered = AsyncTetheredAnthropic(
            client=mock_async_anthropic_client,
            storage=storage,
            run_name="no_dup",
        )
        await tethered.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "First"}],
        )
        first_run_id = tethered._run_id
        await tethered.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Second"}],
        )
        assert tethered._run_id == first_run_id

    async def test_run_id_none_before_first_call(
        self, storage: SQLiteStorage, mock_async_anthropic_client: MagicMock
    ):
        tethered = AsyncTetheredAnthropic(
            client=mock_async_anthropic_client,
            storage=storage,
            run_name="pre_call",
        )
        assert tethered.run_id is None


class TestAsyncAnthropicMessageCapture:
    async def test_records_provider_as_anthropic(
        self, storage: SQLiteStorage, mock_async_anthropic_client: MagicMock
    ):
        tethered = AsyncTetheredAnthropic(
            client=mock_async_anthropic_client,
            storage=storage,
            run_name="provider_test",
        )
        await tethered.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
        )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert steps[0].provider == Provider.ANTHROPIC
        assert steps[0].kind == StepKind.LLM_CALL

    async def test_records_token_usage(
        self, storage: SQLiteStorage, mock_async_anthropic_client: MagicMock
    ):
        tethered = AsyncTetheredAnthropic(
            client=mock_async_anthropic_client,
            storage=storage,
            run_name="tokens_test",
        )
        await tethered.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
        )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert steps[0].input_tokens == 12
        assert steps[0].output_tokens == 8

    async def test_records_cost(
        self, storage: SQLiteStorage, mock_async_anthropic_client: MagicMock
    ):
        tethered = AsyncTetheredAnthropic(
            client=mock_async_anthropic_client,
            storage=storage,
            run_name="cost_test",
        )
        await tethered.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
        )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert steps[0].cost_usd is not None
        assert isinstance(steps[0].cost_usd, Decimal)

    async def test_returns_original_response(
        self, storage: SQLiteStorage, mock_async_anthropic_client: MagicMock
    ):
        tethered = AsyncTetheredAnthropic(
            client=mock_async_anthropic_client,
            storage=storage,
            run_name="passthru_test",
        )
        response = await tethered.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert response.content[0].text == "Hello! I am a mock async Claude response."

    async def test_monotonic_sequence_numbers(
        self, storage: SQLiteStorage, mock_async_anthropic_client: MagicMock
    ):
        tethered = AsyncTetheredAnthropic(
            client=mock_async_anthropic_client,
            storage=storage,
            run_name="seq_test",
        )
        for _ in range(3):
            await tethered.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert [s.sequence_number for s in steps] == [1, 2, 3]


class TestAsyncAnthropicErrorHandling:
    async def test_rate_limit_error_recorded(
        self, storage: SQLiteStorage, mock_async_anthropic_client: MagicMock
    ):
        mock_async_anthropic_client.messages.create = AsyncMock(
            side_effect=anthropic.RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429, headers={}),
                body={"error": {"message": "Rate limit exceeded"}},
            )
        )
        tethered = AsyncTetheredAnthropic(
            client=mock_async_anthropic_client,
            storage=storage,
            run_name="rate_limit_test",
        )
        with pytest.raises(RateLimitError):
            await tethered.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=100,
                messages=[{"role": "user", "content": "Hi"}],
            )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert steps[0].error is not None
        assert steps[0].error["type"] == "RateLimitError"

    async def test_api_error_recorded_and_reraised(
        self, storage: SQLiteStorage, mock_async_anthropic_client: MagicMock
    ):
        mock_async_anthropic_client.messages.create = AsyncMock(
            side_effect=anthropic.APIStatusError(
                message="Server error",
                response=MagicMock(status_code=500, headers={}),
                body={"error": {"message": "Server error"}},
            )
        )
        tethered = AsyncTetheredAnthropic(
            client=mock_async_anthropic_client,
            storage=storage,
            run_name="api_error_test",
        )
        with pytest.raises(CaptureError):
            await tethered.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=100,
                messages=[{"role": "user", "content": "Hi"}],
            )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert steps[0].error is not None
