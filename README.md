# Trending Topic & Viral Signal Forecaster (Proof of Concept)

An agentic system that monitors live RSS + Reddit signals, compares them against
accumulated history, and forecasts which topics are likely to gain traction in
the next 24-48 hours. Built from the six capstone design checkpoints.

## Architecture (maps to the checkpoints)

| Component | File | Checkpoint |
|-----------|------|-----------|
| Collector Agent (live data + topic extraction) | `agents.py` | 1, 2, 5 |
| Retrieval Agent (semantic RAG over memory) | `agents.py`, `vector_store.py` | 3, 5 |
| Critic Agent (Tree-of-Thought + beam search) | `agents.py` | 4, 5 |
| Forecast Agent (briefing + write-back) | `agents.py` | 5 |
| Controller (sequential + feedback) | `agents.py` | 5 |
| Guardrails / human review | `guardrails.py` | 6 |
| External tools (fetch + scoring) | `tools.py` | 2 |
| LLM clients (Claude + Gemini embeddings) | `llm_clients.py` | 2, 3 |
| Tuning constants | `config.py` | all |

## Setup

```powershell
pip install -r requirements.txt
```

Set the two API keys as environment variables (the code reads them automatically):

* `ANTHROPIC_API_KEY` — Claude (reasoning: Critic + Forecast agents)
* `GEMINI_API_KEY` — Gemini (semantic embeddings for retrieval)

If `GEMINI_API_KEY` is missing the retrieval layer falls back to a local
hash-based embedding so the demo still runs.

## Run

Open **`Forecaster.ipynb`** and run the cells top to bottom. It seeds a little
demo history, runs one full forecast cycle, and prints the ranked briefings.

> This is a proof of concept, not production code — it favours readability and
> "it runs correctly" over robustness.
