# Micro-RAG 100k Benchmark Results

- Timestamp: `2026-07-05T13:50:32-04:00`
- Command: `python tests/run_100k_benchmark.py`
- Vector store: `ChromaDB PersistentClient`
- Query embedding model: `DefaultEmbeddingFunction / all-MiniLM-L6-v2`
- Corpus size: `100,000` beliefs
- Query runs: `20` after `5` warmups
- Index build time: `28.08 s`

## Steady-State Latency

| Stage | Mean | Median | P95 |
| :--- | ---: | ---: | ---: |
| Embed | 156.23 ms | 156.01 ms | 167.87 ms |
| Retrieve top-100 | 2.82 ms | 2.72 ms | 3.31 ms |
| Rerank | 0.18 ms | 0.17 ms | 0.21 ms |
| Format | 0.01 ms | 0.01 ms | 0.01 ms |
| Split total | 159.24 ms | 158.91 ms | 170.63 ms |
| Actual `inject()` total | 156.08 ms | 154.12 ms | 167.17 ms |

## Sanity Checks

- Target belief hit rate in top-5 after rerank: `20/20` (`1.000`)
- Sample top belief ids: `bel_55000, bel_55002, bel_54999, bel_54998, bel_55001`

## Methodology Notes

- Query latency measures the steady-state read path, not ingestion.
- The query embedding is real. The 100k stored corpus vectors are synthetic but shaped so the benchmark exercises realistic ANN retrieval and reranking cost at scale.
- This report intentionally does not include vendor comparison numbers, because those published results are usually end-to-end and not normalized to this exact benchmark protocol.
