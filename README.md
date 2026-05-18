# Tether

[![CI](https://github.com/patibandlavenkatamanideep/Tether/actions/workflows/ci.yml/badge.svg)](https://github.com/patibandlavenkatamanideep/Tether/actions)
[![73 tests passing](https://img.shields.io/badge/tests-73%20passing-brightgreen)](https://github.com/patibandlavenkatamanideep/Tether/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Part of The Evaluation Stack](https://img.shields.io/badge/Evaluation%20Stack-RDAB%20%C2%B7%20CostGuard%20%C2%B7%20Tether-7c3aed)](https://github.com/patibandlavenkatamanideep/RealDataAgentBench)

Trace capture and replay layer for production LLM evaluation.

Tether wraps your LLM client (OpenAI today; Anthropic next) and persists
every call — full inputs, outputs, token counts, cost, latency, and
failures — to a local SQLite database. Captured traces become the input
for replay-based evaluation: re-running production traffic against
alternate models to measure cost-vs-quality tradeoffs with statistical
confidence intervals.

Durable execution (automatic recovery, provider failover, streaming)
is on the roadmap — v0.1 is capture-only.

## How This Fits With RDAB and CostGuard

Tether is the capture layer of The Evaluation Stack:

```
Your app ──► Tether (capture) ──► SQLite trace store
                                          │
                                          ▼
                                 CostGuard /replay
                                 (RDAB-grounded scoring)
                                          │
                                          ▼
                                 Cost-vs-quality report
                                 with 95% bootstrap CI
```

- **[RealDataAgentBench (RDAB)](https://github.com/patibandlavenkatamanideep/RealDataAgentBench)**
  is the benchmark methodology — 4-dimensional scoring (correctness, code
  quality, efficiency, statistical validity), 1,412+ runs across 12 models,
  pre-registered experiments.
- **[CostGuard](https://github.com/patibandlavenkatamanideep/CostGuard)**
  is the runtime — RDAB-grounded validity scoring on every proxy call,
  circuit breakers, fallbacks, Prometheus observability.
- **Tether** (this repo) is the capture layer — durable SQLite persistence
  of every LLM call, enabling replay-based evaluation against alternate
  models.

The integration is live: point CostGuard's `POST /replay` at a Tether
SQLite database and it replays every captured prompt against any
alternate model, returning a quality delta and 95% bootstrap CI in one
call. No synthetic benchmarks — your real production traffic.

## Why it matters

Building RDAB and CostGuard, I kept running into the same problem: teams
swap models based on benchmark scores from synthetic tasks, not their own
production traffic. They move to GPT-4o-mini because it's cheaper, find
out from customer tickets that something regressed, and have no data to
explain what changed or why. The missing input is always the same — real
calls, in a form that can be replayed.

Tether's integration with CostGuard `/replay` enables a concrete workflow:
capture 25 calls on `gpt-4o-mini`, replay against `gpt-4.1-mini`, get a
95% CI on the quality delta and the exact cost savings per call. That's
the output of [`scripts/demo_replay.py`](https://github.com/patibandlavenkatamanideep/CostGuard/blob/main/scripts/demo_replay.py)
in CostGuard — evidence, not a thought experiment.

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
    client = OpenAI()  # uses OPENAI_API_KEY from environment
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
    steps = await storage.get_steps_for_run(tethered.run_id)
    print(f"Captured {len(steps)} step(s)")
    print(f"Cost: ${steps[0].cost_usd}")

    # Close storage before passing tether.db to CostGuard /replay
    await storage.close()

asyncio.run(main())
```

Expected output:

```
Attention is a mechanism that allows a model to focus on...
Captured 1 step(s)
Cost: $0.0000215
```

Pass `tethered.run_id` and the path to `tether.db` directly to
`POST /replay` in CostGuard to compare any two models on this traffic.

## What's working in v0.1

- **`TetheredOpenAI`** — drop-in wrapper for `openai.OpenAI`. Zero code changes beyond construction.
- **SQLite capture** — every `chat.completions.create` call recorded with full inputs, outputs, token counts, cost, and latency. WAL mode for safe concurrent reads.
- **Cost tracking** — Decimal-precision cost calculation for `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`.
- **Error recording** — provider errors (429, 5xx) captured as `FailureRecord` rows before being re-raised. Failures are part of the trace, not invisible.
- **Pydantic v2 models** — `Run`, `Step`, `Checkpoint`, `FailureRecord` with strict typing and UTC-aware datetimes.
- **`run_id` property** — `tethered.run_id` exposes the UUID after the first call; pass it directly to CostGuard's `/replay` endpoint.

## Roadmap

**Live in v0.1:**
- ✅ CostGuard `/replay` integration — Tether SQLite → `POST /replay` → quality delta + 95% bootstrap CI + cost savings.
- ✅ Replay engine — prompts replayed against any alternate model in CostGuard's pricing catalogue.
- ✅ Bootstrap CI on quality deltas — `scipy.stats.bootstrap` with 1,000 resamples, percentile method.

**Coming next (capture layer):**
- Async client support — `AsyncTetheredOpenAI` for async-first codebases.
- Streaming capture — transparent buffering and recording of streamed responses.
- Anthropic wrapper — `TetheredAnthropic` with identical API surface.
- Format adapter — translate between OpenAI and Anthropic message formats for cross-provider replay.

**Long-term roadmap (durable execution):**
- Recovery engine — automatic retry with exponential backoff, provider swap (OpenAI ↔ Anthropic) on persistent failures.
- Checkpoint manager — configurable checkpoint frequency, compression, optional S3 upload.
- Budget enforcement — hard stop when `cost_usd` exceeds `monthly_budget`.
- LangChain integration — `TetherCallbackHandler` for LangChain agents.
- Dashboard — local web UI to inspect runs, steps, and costs.

## Status

v0.1 — capture layer complete and integrated with CostGuard `/replay`. Replay-based cost-vs-quality comparison is live. Durable execution (checkpointing, provider failover, streaming) is on the roadmap. APIs will change. Not yet suitable for production use.

## Installation

Install from source (recommended for v0.1):

```bash
git clone https://github.com/patibandlavenkatamanideep/Tether
cd Tether
pip install -e ".[dev]"
```

PyPI release planned for v0.2 alongside the Anthropic wrapper and async client.

## License

MIT — see [LICENSE](LICENSE).

---

Built by [Venkata Manideep Patibandla](https://venkatamanideep.com) · [LinkedIn](https://linkedin.com/in/manideep-analytics) · [GitHub](https://github.com/patibandlavenkatamanideep)

Part of The Evaluation Stack: [RDAB](https://github.com/patibandlavenkatamanideep/RealDataAgentBench) · [CostGuard](https://github.com/patibandlavenkatamanideep/CostGuard) · [Tether](https://github.com/patibandlavenkatamanideep/Tether)
