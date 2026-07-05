# Micro-RAG Associative Multi-Hop Benchmark

- Timestamp: `2026-07-05T13:51:35-04:00`
- Query: `"What is Alice's favorite flower?"`
- Distractors: `5,000`
- Warmups per mode: `3`
- Measured runs per mode: `20`

## Recall

| Mode | Fact A hit rate | Fact B hit rate |
| :--- | ---: | ---: |
| Pure semantic | 1.000 | 1.000 |
| Graph-expanded | 1.000 | 1.000 |

## Latency

| Mode | Mean | Median | P95 |
| :--- | ---: | ---: | ---: |
| Pure semantic | 150.05 ms | 149.04 ms | 157.44 ms |
| Graph-expanded | 147.99 ms | 147.99 ms | 151.87 ms |

## Overhead

- Graph expansion overhead, mean: `-2.06 ms`
- Graph expansion overhead, p95: `-5.57 ms`

## Methodology Notes

- This is a repeated single-scenario benchmark, not a broad accuracy benchmark.
- Distractor vectors are synthetic; the two linked chain facts use real embeddings.
- Timings are steady-state query timings after warmup, not first-query cold-path timings.
