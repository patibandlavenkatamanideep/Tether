"""Tests for tether.capture.anthropic.TetheredAnthropic."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from tether.capture.anthropic import TetheredAnthropic, _compute_cost
from tether.core.models import Provider, RunStatus, StepKind
from tether.core.storage import SQLiteStorage
from tether.exceptions import CaptureError, RateLimitError


class TestTetheredAnthropicCreatesRun:
    async def test_creates_run_on_first_call(
        self, storage: SQLiteStorage, mock_anthropic_client: MagicMock
    ):
        tethered = TetheredAnthropic(
            client=mock_anthropic_client,
            storage=storage,
            run_name="test_claude_agent",
        )
        tethered.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert tethered._run_id is not None
        run = await storage.get_run(tethered._run_id)
        assert run is not None
        assert run.name == "test_claude_agent"
        assert run.status == RunStatus.RUNNING

    async def test_does_not_create_duplicate_runs(
        self, storage: SQLiteStorage, mock_anthropic_client: MagicMock
    ):
        tethered = TetheredAnthropic(
            client=mock_anthropic_client,
            storage=storage,
            run_name="single_run",
        )
        tethered.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "First"}],
        )
        first_run_id = tethered._run_id
        tethered.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Second"}],
        )
        assert tethered._run_id == first_run_id

    async def test_run_id_property_none_before_first_call(
        self, storage: SQLiteStorage, mock_anthropic_client: MagicMock
    ):
        tethered = TetheredAnthropic(
            client=mock_anthropic_client,
            storage=storage,
            run_name="pre_call",
        )
        assert tethered.run_id is None


class TestMessageCapture:
    async def test_records_step_with_correct_provider(
        self, storage: SQLiteStorage, mock_anthropic_client: MagicMock
    ):
        tethered = TetheredAnthropic(
            client=mock_anthropic_client,
            storage=storage,
            run_name="provider_test",
        )
        tethered.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
        )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert len(steps) == 1
        step = steps[0]
        assert step.kind == StepKind.LLM_CALL
        assert step.provider == Provider.ANTHROPIC
        assert step.model == "claude-sonnet-4-6"

    async def test_records_inputs(
        self, storage: SQLiteStorage, mock_anthropic_client: MagicMock
    ):
        tethered = TetheredAnthropic(
            client=mock_anthropic_client,
            storage=storage,
            run_name="inputs_test",
        )
        messages = [{"role": "user", "content": "What is 2+2?"}]
        tethered.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=64,
            messages=messages,
        )
        steps = await storage.get_steps_for_run(tethered._run_id)
        step = steps[0]
        assert step.inputs["messages"] == messages
        assert step.inputs["max_tokens"] == 64

    async def test_records_token_usage(
        self, storage: SQLiteStorage, mock_anthropic_client: MagicMock
    ):
        tethered = TetheredAnthropic(
            client=mock_anthropic_client,
            storage=storage,
            run_name="token_test",
        )
        tethered.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
        )
        steps = await storage.get_steps_for_run(tethered._run_id)
        # mock_anthropic_client returns 12 input, 8 output tokens
        assert steps[0].input_tokens == 12
        assert steps[0].output_tokens == 8

    async def test_records_outputs(
        self, storage: SQLiteStorage, mock_anthropic_client: MagicMock
    ):
        tethered = TetheredAnthropic(
            client=mock_anthropic_client,
            storage=storage,
            run_name="outputs_test",
        )
        tethered.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
        )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert steps[0].outputs is not None
        assert "content" in steps[0].outputs

    async def test_records_latency(
        self, storage: SQLiteStorage, mock_anthropic_client: MagicMock
    ):
        tethered = TetheredAnthropic(
            client=mock_anthropic_client,
            storage=storage,
            run_name="latency_test",
        )
        tethered.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
        )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert steps[0].latency_ms is not None
        assert steps[0].latency_ms >= 0

    async def test_records_completed_at(
        self, storage: SQLiteStorage, mock_anthropic_client: MagicMock
    ):
        tethered = TetheredAnthropic(
            client=mock_anthropic_client,
            storage=storage,
            run_name="completed_at_test",
        )
        tethered.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
        )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert steps[0].completed_at is not None
        assert steps[0].completed_at.tzinfo is not None

    async def test_returns_original_response(
        self, storage: SQLiteStorage, mock_anthropic_client: MagicMock
    ):
        tethered = TetheredAnthropic(
            client=mock_anthropic_client,
            storage=storage,
            run_name="passthru_test",
        )
        response = tethered.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert response.content[0].text == "Hello! I am a mock Claude response."


class TestAnthropicCostCalculation:
    def test_sonnet_cost(self):
        cost = _compute_cost("claude-sonnet-4-6", input_tokens=1000, output_tokens=1000)
        expected = Decimal("0.003") + Decimal("0.015")
        assert cost == expected

    def test_opus_cost(self):
        cost = _compute_cost("claude-opus-4-6", input_tokens=1000, output_tokens=1000)
        expected = Decimal("0.015") + Decimal("0.075")
        assert cost == expected

    def test_haiku_cost(self):
        cost = _compute_cost("claude-haiku-4-5", input_tokens=1000, output_tokens=1000)
        expected = Decimal("0.00025") + Decimal("0.00125")
        assert cost == expected

    def test_haiku_dated_model_id(self):
        cost_a = _compute_cost("claude-haiku-4-5", 1000, 1000)
        cost_b = _compute_cost("claude-haiku-4-5-20251001", 1000, 1000)
        assert cost_a == cost_b

    def test_unknown_model_falls_back_to_sonnet_tier(self):
        cost = _compute_cost("claude-99-ultra", input_tokens=1000, output_tokens=1000)
        expected = Decimal("0.003") + Decimal("0.015")
        assert cost == expected

    def test_cost_is_decimal_not_float(self):
        cost = _compute_cost("claude-sonnet-4-6", input_tokens=7, output_tokens=3)
        assert isinstance(cost, Decimal)

    async def test_cost_stored_in_step(
        self, storage: SQLiteStorage, mock_anthropic_client: MagicMock
    ):
        tethered = TetheredAnthropic(
            client=mock_anthropic_client,
            storage=storage,
            run_name="cost_test",
        )
        tethered.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
        )
        steps = await storage.get_steps_for_run(tethered._run_id)
        step = steps[0]
        assert step.cost_usd is not None
        assert isinstance(step.cost_usd, Decimal)
        expected = _compute_cost("claude-sonnet-4-6", 12, 8)
        assert step.cost_usd == expected


class TestAnthropicErrorHandling:
    async def test_rate_limit_error_recorded(
        self, storage: SQLiteStorage, mock_anthropic_client: MagicMock
    ):
        mock_anthropic_client.messages.create.side_effect = anthropic.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429, headers={}),
            body={"error": {"message": "Rate limit exceeded"}},
        )
        tethered = TetheredAnthropic(
            client=mock_anthropic_client,
            storage=storage,
            run_name="rate_limit_test",
        )
        with pytest.raises(RateLimitError):
            tethered.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=100,
                messages=[{"role": "user", "content": "Hi"}],
            )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert len(steps) == 1
        assert steps[0].error is not None
        assert steps[0].error["type"] == "RateLimitError"

    async def test_api_error_recorded_and_reraised(
        self, storage: SQLiteStorage, mock_anthropic_client: MagicMock
    ):
        mock_anthropic_client.messages.create.side_effect = anthropic.APIStatusError(
            message="Internal server error",
            response=MagicMock(status_code=500, headers={}),
            body={"error": {"message": "Internal server error"}},
        )
        tethered = TetheredAnthropic(
            client=mock_anthropic_client,
            storage=storage,
            run_name="api_error_test",
        )
        with pytest.raises(CaptureError):
            tethered.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=100,
                messages=[{"role": "user", "content": "Hi"}],
            )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert steps[0].error is not None

    async def test_failure_record_provider_is_anthropic(
        self, storage: SQLiteStorage, mock_anthropic_client: MagicMock
    ):
        mock_anthropic_client.messages.create.side_effect = anthropic.RateLimitError(
            message="Rate limit",
            response=MagicMock(status_code=429, headers={}),
            body={},
        )
        tethered = TetheredAnthropic(
            client=mock_anthropic_client,
            storage=storage,
            run_name="failure_provider_test",
        )
        with pytest.raises(RateLimitError):
            tethered.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=100,
                messages=[{"role": "user", "content": "Hi"}],
            )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert steps[0].provider == Provider.ANTHROPIC


class TestAnthropicSequenceNumbers:
    async def test_monotonic_sequence_numbers(
        self, storage: SQLiteStorage, mock_anthropic_client: MagicMock
    ):
        tethered = TetheredAnthropic(
            client=mock_anthropic_client,
            storage=storage,
            run_name="seq_test",
        )
        for _ in range(4):
            tethered.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert len(steps) == 4
        assert [s.sequence_number for s in steps] == [1, 2, 3, 4]

    async def test_sequence_starts_at_one(
        self, storage: SQLiteStorage, mock_anthropic_client: MagicMock
    ):
        tethered = TetheredAnthropic(
            client=mock_anthropic_client,
            storage=storage,
            run_name="seq_start_test",
        )
        tethered.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": "hello"}],
        )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert steps[0].sequence_number == 1
