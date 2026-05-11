# Tether

Trace capture and replay layer for production LLM evaluation.

Tether wraps your LLM client (OpenAI today; Anthropic next) and persists
every call — full inputs, outputs, token counts, cost, latency, and
failures — to a local SQLite database. Captured traces become the input
for replay-based evaluation: re-running production traffic against
alternate models to measure cost-vs-quality tradeoffs with statistical
confidence intervals.

Durable execution (automatic recovery, provider failover, streaming)
is on the roadmap (see below) — v0.1 is capture-only.

## Why it matters

LLM teams ship prompt and model changes blindly. They swap GPT-4o for
GPT-4o-mini, change a system prompt, or add a tool, and discover quality
regressions from customer tickets. The missing input is real production
traffic in a form that can be replayed against alternate configurations.

Tether captures every LLM call your application makes — prompt, response,
tokens, cost, latency — in a queryable local SQLite database. Those
captured traces are then re-playable: feed them into an evaluation pipeline
(like CostGuard's /evaluate endpoint, backed by RealDataAgentBench scoring)
to answer "what would have happened if I'd used the cheaper model?" with a
bootstrap confidence interval, not a guess.

Durable execution — automatic checkpointing, provider failover, streaming
recovery — is the natural extension of the capture layer and is on the
roadmap.

## How This Fits With RDAB and CostGuard

Tether is the capture layer of a three-project evaluation stack:

```
Your app ──► Tether (capture) ──► SQLite trace store
                                          │
                                          ▼
                                 CostGuard /evaluate
                                 (RDAB-grounded scoring)
                                          │
                                          ▼
                                 Cost-vs-quality report
                                 with bootstrap CI
```

- **[RealDataAgentBench](https://github.com/patibandlavenkatamanideep/RealDataAgentBench)**
  is the benchmark methodology — 4-dimensional scoring (correctness, code
  quality, efficiency, statistical validity), 1,412 runs across 12 models,
  pre-registered experiments.
- **[CostGuard](https://github.com/patibandlavenkatamanideep/CostGuard)**
  is the runtime — RDAB-grounded validity scoring on every proxy call,
  circuit breakers, fallbacks, Prometheus observability.
- **Tether** (this repo) is the capture layer — durable SQLite persistence
  of every LLM call, enabling replay-based evaluation against alternate
  models.

The integration unlocks a capability no single repo provides alone:
statistically grounded cost-routing recommendations against real
production traffic, rather than synthetic benchmarks.

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

## What's working in v0.1 (capture layer)

- **`TetheredOpenAI`** — drop-in wrapper for `openai.OpenAI`. Zero code changes beyond construction.
- **SQLite capture** — every `chat.completions.create` call recorded with full inputs, outputs, token counts, cost, and latency. WAL mode for safe concurrent reads.
- **Cost tracking** — Decimal-precision cost calculation for `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`.
- **Error recording** — provider errors (429, 5xx) captured as `FailureRecord` rows before being re-raised. Failures are part of the trace, not invisible.
- **Pydantic v2 models** — `Run`, `Step`, `Checkpoint`, `FailureRecord` with strict typing and UTC-aware datetimes.

## Roadmap

**Capture layer (near-term):**
- Async client support — `AsyncTetheredOpenAI` for async-first codebases.
- Streaming capture — transparent buffering and recording of streamed responses.
- Anthropic wrapper — `TetheredAnthropic` with identical API surface.
- Format adapter — translate between OpenAI and Anthropic message formats for cross-provider replay.

**Replay and evaluation (the integration story):**
- CostGuard integration — feed captured traces into CostGuard's /evaluate endpoint for RDAB-grounded scoring.
- Replay engine — re-run captured prompts against alternate models with statistical comparison reports.
- Bootstrap CI on quality deltas — statistical significance testing for routing decisions.

**Durable execution (longer-term):**
- Recovery engine — automatic retry with exponential backoff, provider swap (OpenAI ↔ Anthropic) on persistent failures.
- Checkpoint manager — configurable checkpoint frequency, compression, optional S3 upload.
- Budget enforcement — hard stop when `cost_usd` exceeds `monthly_budget`.

**Observability:**
- LangChain integration — `TetherCallbackHandler` for LangChain agents.
- Dashboard — local web UI to inspect runs, steps, and costs.

## Status

Early development — v0.1 is the capture layer; replay and durable execution are on the roadmap. APIs will change. Not yet suitable for production use.

## Installation

```bash
pip install tether-py
```

Or from source:

```bash
git clone https://github.com/patibandlavenkatamanideep/Tether
cd tether
pip install -e ".[dev]"
```

## License

MIT — see [LICENSE](LICENSE).
