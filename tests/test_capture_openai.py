"""Tests for tether.capture.openai.TetheredOpenAI."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import openai
import pytest

from tether.capture.openai import TetheredOpenAI, _compute_cost
from tether.core.models import Provider, RunStatus, StepKind
from tether.core.storage import SQLiteStorage
from tether.exceptions import CaptureError, RateLimitError

from tests.conftest import make_completion_response


class TestTetheredOpenAICreatesRun:
    async def test_creates_run_on_first_call(
        self, storage: SQLiteStorage, mock_openai_client: MagicMock
    ):
        tethered = TetheredOpenAI(
            client=mock_openai_client,
            storage=storage,
            run_name="test_agent",
        )
        tethered.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert tethered._run_id is not None
        run = await storage.get_run(tethered._run_id)
        assert run is not None
        assert run.name == "test_agent"
        assert run.status == RunStatus.RUNNING

    async def test_does_not_create_duplicate_runs(
        self, storage: SQLiteStorage, mock_openai_client: MagicMock
    ):
        tethered = TetheredOpenAI(
            client=mock_openai_client,
            storage=storage,
            run_name="single_run_agent",
        )
        tethered.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "First"}],
        )
        first_run_id = tethered._run_id
        tethered.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Second"}],
        )
        assert tethered._run_id == first_run_id


class TestCompletionCapture:
    async def test_records_step_with_inputs(
        self, storage: SQLiteStorage, mock_openai_client: MagicMock
    ):
        tethered = TetheredOpenAI(
            client=mock_openai_client,
            storage=storage,
            run_name="capture_test",
        )
        messages = [{"role": "user", "content": "What is 2+2?"}]
        tethered.chat.completions.create(model="gpt-4o-mini", messages=messages)

        steps = await storage.get_steps_for_run(tethered._run_id)
        assert len(steps) == 1
        step = steps[0]
        assert step.kind == StepKind.LLM_CALL
        assert step.provider == Provider.OPENAI
        assert step.model == "gpt-4o-mini"
        assert step.inputs["messages"] == messages

    async def test_records_token_usage(
        self, storage: SQLiteStorage, mock_openai_client: MagicMock
    ):
        tethered = TetheredOpenAI(
            client=mock_openai_client,
            storage=storage,
            run_name="token_test",
        )
        tethered.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
        )
        steps = await storage.get_steps_for_run(tethered._run_id)
        step = steps[0]
        # mock_openai_client returns 15 input, 10 output tokens
        assert step.input_tokens == 15
        assert step.output_tokens == 10

    async def test_records_outputs(
        self, storage: SQLiteStorage, mock_openai_client: MagicMock
    ):
        tethered = TetheredOpenAI(
            client=mock_openai_client,
            storage=storage,
            run_name="outputs_test",
        )
        tethered.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
        )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert steps[0].outputs is not None
        assert "choices" in steps[0].outputs

    async def test_records_latency(
        self, storage: SQLiteStorage, mock_openai_client: MagicMock
    ):
        tethered = TetheredOpenAI(
            client=mock_openai_client,
            storage=storage,
            run_name="latency_test",
        )
        tethered.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
        )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert steps[0].latency_ms is not None
        assert steps[0].latency_ms >= 0

    async def test_records_completed_at(
        self, storage: SQLiteStorage, mock_openai_client: MagicMock
    ):
        tethered = TetheredOpenAI(
            client=mock_openai_client,
            storage=storage,
            run_name="completed_at_test",
        )
        tethered.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
        )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert steps[0].completed_at is not None
        assert steps[0].completed_at.tzinfo is not None

    async def test_returns_original_response(
        self, storage: SQLiteStorage, mock_openai_client: MagicMock
    ):
        tethered = TetheredOpenAI(
            client=mock_openai_client,
            storage=storage,
            run_name="passthru_test",
        )
        response = tethered.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert response.choices[0].message.content == "Hello! I am a mock LLM response."


class TestCostCalculation:
    def test_gpt4o_mini_cost(self):
        cost = _compute_cost("gpt-4o-mini", input_tokens=1000, output_tokens=1000)
        expected = Decimal("0.00015") + Decimal("0.0006")
        assert cost == expected

    def test_gpt4o_cost(self):
        cost = _compute_cost("gpt-4o", input_tokens=1000, output_tokens=1000)
        expected = Decimal("0.0025") + Decimal("0.01")
        assert cost == expected

    def test_gpt4_turbo_cost(self):
        cost = _compute_cost("gpt-4-turbo", input_tokens=1000, output_tokens=1000)
        expected = Decimal("0.01") + Decimal("0.03")
        assert cost == expected

    def test_unknown_model_uses_default(self):
        cost = _compute_cost("gpt-99-super", input_tokens=1000, output_tokens=1000)
        expected = Decimal("0.01") + Decimal("0.03")
        assert cost == expected

    def test_cost_decimal_precision(self):
        cost = _compute_cost("gpt-4o-mini", input_tokens=7, output_tokens=3)
        assert isinstance(cost, Decimal)

    async def test_cost_stored_in_step(
        self, storage: SQLiteStorage, mock_openai_client: MagicMock
    ):
        tethered = TetheredOpenAI(
            client=mock_openai_client,
            storage=storage,
            run_name="cost_test",
        )
        tethered.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
        )
        steps = await storage.get_steps_for_run(tethered._run_id)
        step = steps[0]
        assert step.cost_usd is not None
        assert isinstance(step.cost_usd, Decimal)
        # 15 input + 10 output tokens at gpt-4o-mini pricing
        expected = _compute_cost("gpt-4o-mini", 15, 10)
        assert step.cost_usd == expected


class TestErrorHandling:
    async def test_rate_limit_error_recorded(
        self, storage: SQLiteStorage, mock_openai_client: MagicMock
    ):
        mock_openai_client.chat.completions.create.side_effect = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429, headers={}),
            body={"error": {"message": "Rate limit exceeded"}},
        )
        tethered = TetheredOpenAI(
            client=mock_openai_client,
            storage=storage,
            run_name="rate_limit_test",
        )
        with pytest.raises(RateLimitError):
            tethered.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hi"}],
            )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert len(steps) == 1
        assert steps[0].error is not None
        assert steps[0].error["type"] == "RateLimitError"

    async def test_api_error_recorded_and_reraised(
        self, storage: SQLiteStorage, mock_openai_client: MagicMock
    ):
        mock_openai_client.chat.completions.create.side_effect = openai.APIStatusError(
            message="Internal server error",
            response=MagicMock(status_code=500, headers={}),
            body={"error": {"message": "Internal server error"}},
        )
        tethered = TetheredOpenAI(
            client=mock_openai_client,
            storage=storage,
            run_name="api_error_test",
        )
        with pytest.raises(CaptureError):
            tethered.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hi"}],
            )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert steps[0].error is not None


class TestSequenceNumbers:
    async def test_monotonic_sequence_numbers(
        self, storage: SQLiteStorage, mock_openai_client: MagicMock
    ):
        tethered = TetheredOpenAI(
            client=mock_openai_client,
            storage=storage,
            run_name="seq_test",
        )
        for _ in range(5):
            tethered.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "ping"}],
            )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert len(steps) == 5
        seq_numbers = [s.sequence_number for s in steps]
        assert seq_numbers == list(range(1, 6))

    async def test_sequence_starts_at_one(
        self, storage: SQLiteStorage, mock_openai_client: MagicMock
    ):
        tethered = TetheredOpenAI(
            client=mock_openai_client,
            storage=storage,
            run_name="seq_start_test",
        )
        tethered.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
        )
        steps = await storage.get_steps_for_run(tethered._run_id)
        assert steps[0].sequence_number == 1
