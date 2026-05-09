# Tether

Durable execution for long-running LLM agents.

Tether wraps your LLM client (OpenAI, Anthropic — more coming), records every call to a local SQLite database, checkpoints agent state, and automatically recovers from provider failures — rate limits, 5xx errors, crashes, and network timeouts — without restarting your entire agent from scratch.

## Why it matters

Imagine an agent that has been running for 4 hours, made 200 LLM calls, and is three-quarters of the way through a complex research task. Then OpenAI returns a 429. Without Tether, you lose everything and start over. With Tether, the run resumes from the last checkpoint, swaps to Anthropic if OpenAI stays down, and keeps going — all transparently.

## Quickstart

```python
import asyncio
from openai import OpenAI
from tether import TetheredOpenAI, SQLiteStorage

async def main():
    # Initialize storage (creates tether.db in the current directory)
    storage = SQLiteStorage("tether.db")
    await storage.initialize()

    # Wrap your existing OpenAI client — zero changes to the rest of your code
    client = OpenAI(api_key="sk-...")
    tethered = TetheredOpenAI(
        client=client,
        storage=storage,
        run_name="research_agent_v1",
    )

    # Use exactly like the regular OpenAI client
    response = tethered.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Summarise the paper on attention mechanisms"}],
    )
    print(response.choices[0].message.content)

    # Every call is now recorded to SQLite
    steps = await storage.get_steps_for_run(tethered._run_id)
    print(f"Captured {len(steps)} step(s)")
    print(f"Cost: ${steps[0].cost_usd}")

asyncio.run(main())
```

## What's working in v0.1

- **`TetheredOpenAI`** — drop-in wrapper for `openai.OpenAI`
- **SQLite capture** — every `chat.completions.create` call is recorded with full inputs, outputs, token counts, cost, and latency
- **Cost tracking** — Decimal-precision cost calculation for `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`
- **Error recording** — provider errors (429, 5xx) are captured as `FailureRecord` rows before being re-raised
- **Pydantic v2 models** — `Run`, `Step`, `Checkpoint`, `FailureRecord` with strict typing and UTC-aware datetimes
- **WAL-mode SQLite** — safe for concurrent reads during writes

## Roadmap

- **Recovery engine** — automatic retry with exponential backoff, provider swap (OpenAI ↔ Anthropic)
- **Async client support** — `AsyncTetheredOpenAI` for async-first codebases
- **Streaming capture** — transparent buffering and recording of streamed responses
- **Anthropic wrapper** — `TetheredAnthropic` with identical API surface
- **Format adapter** — translate between OpenAI and Anthropic message formats mid-run
- **Checkpoint manager** — configurable checkpoint frequency, compression, S3 upload
- **LangChain integration** — `TetherCallbackHandler` for LangChain agents
- **Budget enforcement** — hard stop when `cost_usd` exceeds `monthly_budget`
- **Dashboard** — local web UI to inspect runs, steps, and costs

## Status

Early development. APIs will change. Not yet suitable for production use.

## Installation

```bash
pip install tether-py
```

Or from source:

```bash
git clone https://github.com/venkatamanideep/tether
cd tether
pip install -e ".[dev]"
```

## License

MIT — see [LICENSE](LICENSE).
