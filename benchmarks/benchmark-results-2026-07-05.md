# MicroRAG Benchmark Results

- Timestamp: `2026-07-05T12:52:47-04:00`
- Host: `Linux 6.17.0-35-generic #35~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue May 26 19:30:42 UTC 2 x86_64 GNU/Linux`
- Python: `3.12.3`
- Command: `python3 -m unittest tests.test_benchmarks -v`

## Results

- `ContextCompressor` overhead, 100 messages: `0.03 ms` per call
- `PreGenerativeInjector` latency, 1,000 beliefs: `69.29 ms` per call

## Test Status

- `test_compression_overhead`: `ok`
- `test_injector_latency`: `ok`
- Suite result: `Ran 2 tests in 5.562s`, `OK`

## Notes

- These benchmarks used `DummyVectorStore`, so they measure local framework overhead rather than real semantic embedding latency.
- The benchmark setup populated the belief store with `1,000` mock beliefs before measuring injection latency.
