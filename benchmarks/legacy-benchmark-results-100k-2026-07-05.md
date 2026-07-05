# Legacy 100k Benchmark Report

This file preserves the older root-level `100k` benchmark report that was generated before the benchmark harness was tightened.

It should be treated as a historical artifact, not the current source of truth, because:

- it used the older harness,
- it embedded unsupported cross-vendor comparison figures,
- and it did not clearly separate cold-path setup concerns from steady-state query timing.

## Preserved Contents

# Micro-RAG 100k Real Vector DB Benchmark Results

This benchmark evaluates the latency of the memory pipeline operating over a real **ChromaDB database containing 100,000 vector records**, utilizing the standard local **all-MiniLM-L6-v2 ONNX embedder** model.

## ⏱️ Execution Split Times (Averages over 20 runs)

- **Embed Time (ONNX Model Inference)**: `141.19 ms`
- **Retrieval Time (ChromaDB Query over 100k vectors)**: `2.67 ms`
- **Rerank/Relevance Time (Local CPU)**: `0.15 ms`
- **Total End-to-End Retrieval Latency**: `144.00 ms`
