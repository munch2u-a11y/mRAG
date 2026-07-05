# Micro-RAG Associative Multi-Hop Benchmark

- Timestamp: `2026-07-05T14:18:45-04:00`
- Query: `"What is Alice's favorite flower?"`
- Distractors: `5,000`
- Warmups per mode: `3`
- Measured runs per mode: `20`

## Recall

| Mode | Fact A hit rate | Fact B hit rate |
| :--- | ---: | ---: |
| Pure semantic | 1.000 | 0.000 |
| Graph-expanded | 1.000 | 1.000 |

## Latency

| Mode | Mean | Median | P95 |
| :--- | ---: | ---: | ---: |
| Pure semantic | 141.01 ms | 141.04 ms | 142.95 ms |
| Graph-expanded | 142.40 ms | 141.80 ms | 145.00 ms |

## Overhead

- Graph expansion overhead, mean: `1.39 ms`
- Graph expansion overhead, p95: `2.05 ms`

## Methodology Notes

- This is a repeated single-scenario benchmark, not a broad accuracy benchmark.
- Distractor vectors are synthetic; the two linked chain facts use real embeddings.
- Timings are steady-state query timings after warmup, not first-query cold-path timings.
