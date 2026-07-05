import json
import os
import shutil
import sys
import time
from datetime import datetime
from statistics import mean, median

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mrag import BeliefStore, PreGenerativeInjector, create_vector_store


def _percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def _format_stats(values):
    return {
        "mean_ms": round(mean(values), 2),
        "median_ms": round(median(values), 2),
        "p95_ms": round(_percentile(values, 0.95), 2),
        "min_ms": round(min(values), 2),
        "max_ms": round(max(values), 2),
    }


def _benchmark_mode(injector, query_text, fact_a_content, fact_b_content, warmups, runs):
    for _ in range(warmups):
        injector.clear_blacklist()
        injector.inject(query_text)

    latencies = []
    fact_a_hits = 0
    fact_b_hits = 0
    sample_context = ""

    for _ in range(runs):
        injector.clear_blacklist()
        t0 = time.perf_counter()
        context = injector.inject(query_text)
        latencies.append((time.perf_counter() - t0) * 1000)

        if fact_a_content in context:
            fact_a_hits += 1
        if fact_b_content in context:
            fact_b_hits += 1
        if not sample_context:
            sample_context = context

    return {
        "latency_ms": _format_stats(latencies),
        "fact_a_hit_rate": round(fact_a_hits / runs, 3),
        "fact_b_hit_rate": round(fact_b_hits / runs, 3),
        "sample_context": sample_context,
    }


def run_associative_benchmark():
    num_distractors = 5000
    warmup_runs = 3
    measured_runs = 20
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    date_slug = timestamp[:10]

    db_dir = "./tests/benchmark_assoc_chroma_db"
    belief_dir = "./tests/benchmark_assoc_belief_data"
    output_dir = "./benchmarks"

    for path in [db_dir, belief_dir]:
        if os.path.exists(path):
            shutil.rmtree(path)
    os.makedirs(output_dir, exist_ok=True)

    print("====================================================")
    print("Associative & Multi-Hop Memory Benchmark")
    print("====================================================")

    print("1. Initializing stores...")
    belief_store = BeliefStore(data_dir=belief_dir)
    belief_store._cache_loaded = True
    vector_store = create_vector_store("chromadb", persist_dir=db_dir)

    print(f"2. Generating {num_distractors} distractor memories...")
    for i in range(num_distractors):
        bid = f"distractor_{i}"
        belief_store._beliefs_cache[bid] = belief_store._normalize_belief(
            {
                "id": bid,
                "content": f"The standard engine operating temperature is {70 + i} degrees.",
                "confidence": 0.5,
                "stability_index": 0.5,
                "relations": [],
                "verifications": 1.0,
            },
            category="premises",
        )

    distractor_ids = [f"distractor_{i}" for i in range(num_distractors)]
    distractor_embs = []
    for _ in range(num_distractors):
        vector = np.random.randn(384).astype(np.float32)
        norm = np.linalg.norm(vector)
        distractor_embs.append(vector / norm if norm > 0 else vector)

    for bid, emb in zip(distractor_ids, distractor_embs):
        belief_store._beliefs_cache[bid]["embedding"] = emb.tolist()

    batch_size = 2500
    for idx in range(0, num_distractors, batch_size):
        vector_store.add_vectors(
            distractor_ids[idx:idx + batch_size],
            distractor_embs[idx:idx + batch_size],
            [{"category": "premises"} for _ in range(len(distractor_ids[idx:idx + batch_size]))],
        )

    print("3. Inserting multi-hop associative chain...")
    fact_a_id = "fact_alice_flower"
    fact_b_id = "fact_bob_food"
    fact_a_content = "Alice's favorite flower is Bob's favorite food."
    fact_b_content = "Bob's favorite food is Sushi."

    emb_a = vector_store.embed_text(fact_a_content)
    emb_b = vector_store.embed_text(fact_b_content)

    belief_store._beliefs_cache[fact_a_id] = belief_store._normalize_belief(
        {
            "id": fact_a_id,
            "content": fact_a_content,
            "confidence": 0.9,
            "stability_index": 0.8,
            "relations": [fact_b_id],
            "verifications": 1.0,
            "embedding": emb_a.tolist(),
        },
        category="premises",
    )
    belief_store._beliefs_cache[fact_b_id] = belief_store._normalize_belief(
        {
            "id": fact_b_id,
            "content": fact_b_content,
            "confidence": 0.9,
            "stability_index": 0.8,
            "relations": [fact_a_id],
            "verifications": 1.0,
            "embedding": emb_b.tolist(),
        },
        category="premises",
    )

    vector_store.add_vectors(
        [fact_a_id, fact_b_id],
        [emb_a, emb_b],
        [{"category": "premises"}, {"category": "premises"}],
    )

    query_text = "What is Alice's favorite flower?"

    print("\n4. Benchmarking pure semantic search...")
    injector_semantic = PreGenerativeInjector(
        belief_store=belief_store,
        vector_store=vector_store,
        enable_graph_expansion=False,
    )
    injector_semantic._indexed_belief_ids.update(belief_store._beliefs_cache.keys())
    semantic_result = _benchmark_mode(
        injector_semantic,
        query_text,
        fact_a_content,
        fact_b_content,
        warmup_runs,
        measured_runs,
    )

    print("5. Benchmarking graph-expanded retrieval...")
    injector_graph = PreGenerativeInjector(
        belief_store=belief_store,
        vector_store=vector_store,
        enable_graph_expansion=True,
    )
    injector_graph._indexed_belief_ids.update(belief_store._beliefs_cache.keys())
    graph_result = _benchmark_mode(
        injector_graph,
        query_text,
        fact_a_content,
        fact_b_content,
        warmup_runs,
        measured_runs,
    )

    overhead_mean = graph_result["latency_ms"]["mean_ms"] - semantic_result["latency_ms"]["mean_ms"]
    overhead_p95 = graph_result["latency_ms"]["p95_ms"] - semantic_result["latency_ms"]["p95_ms"]

    result = {
        "timestamp": timestamp,
        "query_text": query_text,
        "num_distractors": num_distractors,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "semantic": semantic_result,
        "graph_expanded": graph_result,
        "graph_overhead_mean_ms": round(overhead_mean, 2),
        "graph_overhead_p95_ms": round(overhead_p95, 2),
        "methodology": {
            "distractor_embeddings": "synthetic random normalized vectors",
            "chain_facts_embeddings": "real embeddings from Chroma DefaultEmbeddingFunction",
            "query_latency_mode": "steady-state after warmup",
        },
    }

    json_path = os.path.join(output_dir, f"benchmark-results-associative-{date_slug}.json")
    md_path = os.path.join(output_dir, f"benchmark-results-associative-{date_slug}.md")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    report_content = f"""# Micro-RAG Associative Multi-Hop Benchmark

- Timestamp: `{timestamp}`
- Query: `"{query_text}"`
- Distractors: `{num_distractors:,}`
- Warmups per mode: `{warmup_runs}`
- Measured runs per mode: `{measured_runs}`

## Recall

| Mode | Fact A hit rate | Fact B hit rate |
| :--- | ---: | ---: |
| Pure semantic | {semantic_result["fact_a_hit_rate"]:.3f} | {semantic_result["fact_b_hit_rate"]:.3f} |
| Graph-expanded | {graph_result["fact_a_hit_rate"]:.3f} | {graph_result["fact_b_hit_rate"]:.3f} |

## Latency

| Mode | Mean | Median | P95 |
| :--- | ---: | ---: | ---: |
| Pure semantic | {semantic_result["latency_ms"]["mean_ms"]:.2f} ms | {semantic_result["latency_ms"]["median_ms"]:.2f} ms | {semantic_result["latency_ms"]["p95_ms"]:.2f} ms |
| Graph-expanded | {graph_result["latency_ms"]["mean_ms"]:.2f} ms | {graph_result["latency_ms"]["median_ms"]:.2f} ms | {graph_result["latency_ms"]["p95_ms"]:.2f} ms |

## Overhead

- Graph expansion overhead, mean: `{overhead_mean:.2f} ms`
- Graph expansion overhead, p95: `{overhead_p95:.2f} ms`

## Methodology Notes

- This is a repeated single-scenario benchmark, not a broad accuracy benchmark.
- Distractor vectors are synthetic; the two linked chain facts use real embeddings.
- Timings are steady-state query timings after warmup, not first-query cold-path timings.
"""

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(report_content)

    print(f"Saved JSON report to {json_path}")
    print(f"Saved Markdown report to {md_path}")

    shutil.rmtree(db_dir)
    shutil.rmtree(belief_dir)


if __name__ == "__main__":
    run_associative_benchmark()
