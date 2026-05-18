"""Tether — durable execution for long-running LLM agents.

Tether wraps your LLM client, records every call to SQLite, checkpoints
agent state, and recovers from provider failures automatically.

Quick start:
    from openai import OpenAI
    from tether import TetheredOpenAI, SQLiteStorage
    import asyncio

    async def main():
        storage = SQLiteStorage("agent.db")
        await storage.initialize()

        tethered = TetheredOpenAI(
            client=OpenAI(api_key="..."),
            storage=storage,
            run_name="my_agent",
        )
        response = tethered.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(response.choices[0].message.content)

    asyncio.run(main())
"""

from tether.capture.anthropic import AsyncTetheredAnthropic, TetheredAnthropic
from tether.capture.openai import AsyncTetheredOpenAI, TetheredOpenAI
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
from tether.exceptions import (
    BudgetExceededError,
    CaptureError,
    ConfigurationError,
    ProviderError,
    RateLimitError,
    RecoveryError,
    StorageError,
    TetherError,
)

__version__ = "0.2.0"
__author__ = "Venkata Manideep Patibandla"

__all__ = [
    # Main API
    "TetheredOpenAI",
    "AsyncTetheredOpenAI",
    "TetheredAnthropic",
    "AsyncTetheredAnthropic",
    "SQLiteStorage",
    # Models
    "Run",
    "Step",
    "Checkpoint",
    "FailureRecord",
    "RunStatus",
    "StepKind",
    "Provider",
    # Exceptions
    "TetherError",
    "StorageError",
    "CaptureError",
    "RecoveryError",
    "ConfigurationError",
    "RateLimitError",
    "ProviderError",
    "BudgetExceededError",
]
