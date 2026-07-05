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
