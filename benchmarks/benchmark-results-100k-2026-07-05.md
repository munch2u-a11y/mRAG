# Micro-RAG 100k Benchmark Results

- Timestamp: `2026-07-05T14:19:10-04:00`
- Command: `python tests/run_100k_benchmark.py`
- Vector store: `ChromaDB PersistentClient`
- Query embedding model: `DefaultEmbeddingFunction / all-MiniLM-L6-v2`
- Corpus size: `100,000` beliefs
- Query runs: `20` after `5` warmups
- Index build time: `24.32 s`

## Steady-State Latency

| Stage | Mean | Median | P95 |
| :--- | ---: | ---: | ---: |
| Embed | 134.59 ms | 134.22 ms | 137.36 ms |
| Retrieve top-100 | 2.46 ms | 2.44 ms | 2.63 ms |
| Rerank | 0.16 ms | 0.16 ms | 0.17 ms |
| Format | 0.01 ms | 0.01 ms | 0.01 ms |
| Split total | 137.21 ms | 136.83 ms | 139.93 ms |
| Actual `inject()` total | 136.87 ms | 136.74 ms | 139.85 ms |

## Sanity Checks

- Target belief hit rate in top-5 after rerank: `20/20` (`1.000`)
- Sample top belief ids: `bel_55000, bel_55002, bel_54999, bel_54998, bel_55001`

## Methodology Notes

- Query latency measures the steady-state read path, not ingestion.
- The query embedding is real. The 100k stored corpus vectors are synthetic but shaped so the benchmark exercises realistic ANN retrieval and reranking cost at scale.
- This report intentionally does not include vendor comparison numbers, because those published results are usually end-to-end and not normalized to this exact benchmark protocol.
