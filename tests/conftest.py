"""Shared pytest fixtures for the Tether test suite."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from tether.core.storage import SQLiteStorage


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Return a temporary path for a SQLite database, unique per test."""
    return tmp_path / "test_tether.db"


@pytest.fixture
async def storage(tmp_db_path: Path) -> SQLiteStorage:
    """Return an initialized SQLiteStorage, torn down after each test."""
    s = SQLiteStorage(tmp_db_path)
    await s.initialize()
    yield s  # type: ignore[misc]
    await s.close()


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Return a MagicMock that mimics openai.OpenAI with a realistic response."""
    client = MagicMock()

    # Build a realistic ChatCompletion-like response object.
    message = MagicMock()
    message.content = "Hello! I am a mock LLM response."
    message.role = "assistant"
    message.function_call = None
    message.tool_calls = None

    choice = MagicMock()
    choice.index = 0
    choice.message = message
    choice.finish_reason = "stop"

    usage = MagicMock()
    usage.prompt_tokens = 15
    usage.completion_tokens = 10
    usage.total_tokens = 25

    completion = MagicMock()
    completion.id = f"chatcmpl-{uuid4().hex[:12]}"
    completion.object = "chat.completion"
    completion.model = "gpt-4o-mini"
    completion.choices = [choice]
    completion.usage = usage
    completion.model_dump.return_value = {
        "id": completion.id,
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": message.content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 10,
            "total_tokens": 25,
        },
    }

    client.chat.completions.create.return_value = completion
    return client


@pytest.fixture
def mock_anthropic_client() -> MagicMock:
    """Return a MagicMock that mimics anthropic.Anthropic with a realistic response."""
    client = MagicMock()

    content_block = MagicMock()
    content_block.type = "text"
    content_block.text = "Hello! I am a mock Claude response."

    usage = MagicMock()
    usage.input_tokens = 12
    usage.output_tokens = 8

    message = MagicMock()
    message.id = f"msg_{uuid4().hex[:12]}"
    message.type = "message"
    message.role = "assistant"
    message.model = "claude-sonnet-4-6"
    message.content = [content_block]
    message.usage = usage
    message.stop_reason = "end_turn"
    message.model_dump.return_value = {
        "id": message.id,
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": content_block.text}],
        "usage": {"input_tokens": 12, "output_tokens": 8},
        "stop_reason": "end_turn",
    }

    client.messages.create.return_value = message
    return client


@pytest.fixture
def mock_async_openai_client() -> MagicMock:
    """Return a MagicMock that mimics openai.AsyncOpenAI."""
    client = MagicMock()

    message = MagicMock()
    message.content = "Hello! I am a mock async LLM response."
    message.role = "assistant"

    choice = MagicMock()
    choice.index = 0
    choice.message = message
    choice.finish_reason = "stop"

    usage = MagicMock()
    usage.prompt_tokens = 15
    usage.completion_tokens = 10
    usage.total_tokens = 25

    completion = MagicMock()
    completion.id = f"chatcmpl-{uuid4().hex[:12]}"
    completion.model = "gpt-4o-mini"
    completion.choices = [choice]
    completion.usage = usage
    completion.model_dump.return_value = {
        "id": completion.id,
        "model": "gpt-4o-mini",
        "choices": [{"message": {"role": "assistant", "content": message.content}}],
        "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
    }

    client.chat.completions.create = AsyncMock(return_value=completion)
    return client


@pytest.fixture
def mock_async_anthropic_client() -> MagicMock:
    """Return a MagicMock that mimics anthropic.AsyncAnthropic."""
    client = MagicMock()

    content_block = MagicMock()
    content_block.type = "text"
    content_block.text = "Hello! I am a mock async Claude response."

    usage = MagicMock()
    usage.input_tokens = 12
    usage.output_tokens = 8

    message = MagicMock()
    message.id = f"msg_{uuid4().hex[:12]}"
    message.type = "message"
    message.role = "assistant"
    message.model = "claude-sonnet-4-6"
    message.content = [content_block]
    message.usage = usage
    message.stop_reason = "end_turn"
    message.model_dump.return_value = {
        "id": message.id,
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": content_block.text}],
        "usage": {"input_tokens": 12, "output_tokens": 8},
        "stop_reason": "end_turn",
    }

    client.messages.create = AsyncMock(return_value=message)
    return client


def make_completion_response(
    content: str = "Hello!",
    model: str = "gpt-4o-mini",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> MagicMock:
    """Helper to build a configurable mock completion response."""
    message = MagicMock()
    message.content = content
    message.role = "assistant"

    choice = MagicMock()
    choice.index = 0
    choice.message = message
    choice.finish_reason = "stop"

    usage = MagicMock()
    usage.prompt_tokens = input_tokens
    usage.completion_tokens = output_tokens
    usage.total_tokens = input_tokens + output_tokens

    completion = MagicMock()
    completion.id = f"chatcmpl-{uuid4().hex[:12]}"
    completion.model = model
    completion.choices = [choice]
    completion.usage = usage
    completion.model_dump.return_value = {
        "id": completion.id,
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }
    return completion
