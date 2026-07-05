# Needle in a Haystack (NIAH) Benchmark Results

- Timestamp: `2026-07-05T14:18:34-04:00`
- Query: `"What is the vault code?"`
- Haystack size: `5,000`
- Warmups: `3`
- Measured runs: `20`

## Retrieval

- Recall rate: `1.000`
- Average needle score: `0.8020`
- Average highest distractor score: `0.1835`
- Average precision margin: `0.6185`

## Latency

- Mean: `141.93 ms`
- Median: `141.95 ms`
- P95: `144.43 ms`

## Methodology Notes

- This is a repeated single-query retrieval benchmark, not a broad accuracy benchmark.
- Distractor vectors are synthetic; the needle uses a real embedding.
- Timings are steady-state query timings after warmup, not first-query cold-path timings.
