# Micro-RAG Benchmark Summary

## Current Benchmark Layout

- `tests/run_100k_benchmark.py` is the tightened large-scale harness.
- `tests/run_associative_benchmark.py` now measures repeated steady-state runs with warmups and percentile stats.
- `tests/run_niah_benchmark.py` now measures repeated steady-state runs with warmups and percentile stats.
- Historical one-off benchmark reports have been moved under `benchmarks/legacy-*`.

## Current Interpretation Guidance

- Treat `associative` as the strongest behavior benchmark because it demonstrates graph-expanded recall on a linked fact chain.
- Treat `NIAH` as a focused retrieval sanity check, not a broad memory-quality benchmark.
- Treat the old `100k` report as historical only; use the rewritten `100k` harness for future runs.

## Current Caveats

- The associative and NIAH harnesses still use synthetic distractor vectors, so they are useful for controlled retrieval stress tests rather than full realism.
- The `100k` harness depends on Chroma's default embedding model being available locally; if it cannot initialize, the harness now fails fast instead of emitting misleading numbers.
- Repeated single-scenario success rates should not be described as general accuracy claims without a larger benchmark set.

## Real-Conversation QA Benchmarks (LoCoMo, LongMemEval)

Added 2026-07-06. Unlike the benchmarks above, `tests/run_locomo_benchmark.py`
and `tests/run_longmemeval_benchmark.py` run the full pipeline — fact
extraction, retrieval, answer generation, and grading — against a real LLM
(`gemini-3.1-flash-lite`), not deterministic mocks. This makes them the most
representative evidence currently available for real-world accuracy and
token cost, but also the noisiest: fact extraction runs at temperature > 0,
so identical code re-run against a freshly-ingested conversation can shift
individual question outcomes by a few points. Observed noise band this
session: roughly ±3 questions out of 60 between otherwise-identical LoCoMo
runs. Draw conclusions from aggregate trends and repeated runs, not a single
number.

**Results as of 2026-07-06** (see `benchmark-results-locomo-*.md` and
`benchmark-results-longmemeval-s-*.md` for full per-question logs):

| Dataset | Sample | Accuracy | Avg. injected tokens |
| :--- | :--- | ---: | ---: |
| LoCoMo | 3 conversations, 60 questions | 85.0% | 711 |
| LongMemEval_S | stratified, 20 questions (all 6 categories + abstention) | 85.0% | 719 |
| LongMemEval_S | single-session-user only, 10 questions | 90.0% | 707 |

Key finding: average injected token count stayed within a ~60-token band
(657-719) across both datasets and every question category, despite
LongMemEval's per-question haystack (38-62 independently-sampled sessions)
running roughly double LoCoMo's per-conversation session count. The token
budget in `PreGenerativeInjector` (`max_injected_tokens`, `token_budget_fraction`)
caps injected context size regardless of underlying belief-store size, so
retrieval cost does not grow with conversation length — this is the main
claim these two benchmarks support. Current benchmark code now records
tokenizer-backed counts when the tokenizer assets are locally available.

**Known failure modes observed** (see full logs for detail): a handful of
misses are generation-side (the answering LLM retrieved the correct facts but
didn't make an inferential leap the grader expected), not retrieval failures
— worth distinguishing when triaging a miss, since the fix differs (prompt
tuning vs. retrieval logic).

**Sample size**: 90 total graded questions across both datasets so far. Track
this figure honestly in any future write-up — it is not yet large enough to
support a strong statistical claim, only a directional one. No vendor
comparison numbers are included, consistent with the policy in the README.
