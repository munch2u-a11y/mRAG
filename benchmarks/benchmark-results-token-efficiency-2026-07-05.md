# Micro-RAG Token Efficiency Over Time Benchmark

- Timestamp: `2026-07-05T13:50:33-04:00`
- Command: `python tests/run_token_efficiency_benchmark.py`
- Simulated turns: `200` (fact introduced every `5` turns)
- Context limit for compression: `8000` tokens

## Prompt Tokens Per Turn

| Turn | Baseline (full history) | Micro-RAG | Saving |
| ---: | ---: | ---: | ---: |
| 25 | 2,236 | 2,364 | -5.7% |
| 50 | 4,509 | 4,621 | -2.5% |
| 100 | 9,048 | 5,196 | 42.6% |
| 150 | 13,657 | 2,106 | 84.6% |
| 200 | 18,267 | 3,055 | 83.3% |

## Cumulative Prompt Tokens (200 turns)

- Baseline: `1,823,991`
- Micro-RAG: `631,080`
- **Cumulative saving: `65.4%`** (grows with session length)

## Long-Range Fact Recall

- Probes: `21` (facts at least 25 turns old at probe time)
- Micro-RAG recall (fact present in compressed context or injection): `0.857`
- Micro-RAG recall for facts >= 100 turns old: `0.875`
- Baseline recall: `1.0` by construction (it pays full-history token cost for it)

## Skill Retrieval (via `mrag.adapters` import)

- Catalog: `20` tools, natural-language task queries
- Correct skill surfaced in top-5 injection: `1.0`
- Missed: `none`
- Injection latency mean: `147.96 ms`

## Injection Latency During Simulation

- Mean: `226.51 ms`
- Median: `152.17 ms`
- Max (includes first-call embedding warmup): `3753.92 ms`

## Methodology Notes

- Retrieval quality is real: beliefs, skills, and queries are embedded with the
  Chroma default model (all-MiniLM-L6-v2); nothing retrieval-related is mocked.
- The compression summarizer and belief extractor are deterministic extractive
  mocks so the run is reproducible without API keys. They model an LLM that
  retains salient facts; with a production LLM, summary wording differs but the
  token accounting protocol is identical.
- Token counts use the same chars/4 estimator for both conditions, so the
  relative saving is estimator-independent to first order.
- No vendor comparison numbers are included: published numbers for other memory
  systems are not normalized to this protocol. The scripts are runnable as-is
  for anyone who wants to reproduce or adapt the protocol.
