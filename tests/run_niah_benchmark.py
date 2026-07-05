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


def run_niah():
    haystack_size = 5000
    warmup_runs = 3
    measured_runs = 20
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    date_slug = timestamp[:10]

    db_dir = "./tests/benchmark_niah_chroma_db"
    belief_dir = "./tests/benchmark_niah_belief_data"
    output_dir = "./benchmarks"

    for path in [db_dir, belief_dir]:
        if os.path.exists(path):
            shutil.rmtree(path)
    os.makedirs(output_dir, exist_ok=True)

    print("====================================================")
    print("Needle in a Haystack (NIAH) Memory Benchmark")
    print("====================================================")

    print("1. Initializing stores...")
    belief_store = BeliefStore(data_dir=belief_dir)
    belief_store._cache_loaded = True
    vector_store = create_vector_store("chromadb", persist_dir=db_dir)

    needle_content = "The passcode for the private vault is 9987-Alpha."
    query_text = "What is the vault code?"

    print(f"2. Generating {haystack_size} distractor memories...")
    for i in range(haystack_size):
        belief_id = f"distractor_{i}"
        belief_store._beliefs_cache[belief_id] = belief_store._normalize_belief(
            {
                "id": belief_id,
                "content": f"User checked system logs at index {i}.",
                "confidence": 0.5,
                "stability_index": 0.5,
                "relations": [],
                "verifications": 1.0,
            },
            category="premises",
        )

    batch_ids = [f"distractor_{i}" for i in range(haystack_size)]
    batch_embs = []
    for _ in range(haystack_size):
        vector = np.random.randn(384).astype(np.float32)
        norm = np.linalg.norm(vector)
        batch_embs.append(vector / norm if norm > 0 else vector)

    for bid, emb in zip(batch_ids, batch_embs):
        belief_store._beliefs_cache[bid]["embedding"] = emb.tolist()

    batch_size = 2500
    for idx in range(0, haystack_size, batch_size):
        vector_store.add_vectors(
            batch_ids[idx:idx + batch_size],
            batch_embs[idx:idx + batch_size],
            [{"category": "premises"} for _ in range(len(batch_ids[idx:idx + batch_size]))],
        )

    print("3. Embedding the needle with the live model...")
    needle_emb = vector_store.embed_text(needle_content)
    needle_id = "needle_vault_code"

    belief_store._beliefs_cache[needle_id] = belief_store._normalize_belief(
        {
            "id": needle_id,
            "content": needle_content,
            "confidence": 0.9,
            "stability_index": 0.8,
            "relations": [],
            "verifications": 1.0,
            "embedding": needle_emb.tolist(),
        },
        category="premises",
    )
    vector_store.add_vectors([needle_id], [needle_emb], [{"category": "premises"}])

    injector = PreGenerativeInjector(belief_store=belief_store, vector_store=vector_store)
    injector._indexed_belief_ids.update(belief_store._beliefs_cache.keys())

    print("4. Warming query path...")
    for _ in range(warmup_runs):
        injector.clear_blacklist()
        injector.inject(trigger_text=query_text)

    print("5. Running repeated query benchmark...")
    latencies = []
    success_count = 0
    margins = []
    needle_scores = []
    distractor_scores = []

    for _ in range(measured_runs):
        injector.clear_blacklist()
        t0 = time.perf_counter()
        injected_context = injector.inject(trigger_text=query_text)
        latencies.append((time.perf_counter() - t0) * 1000)

        if needle_content in injected_context:
            success_count += 1

        query_emb = vector_store.embed_text(query_text)
        top_results = vector_store.query_top_k(query_emb, k=5)

        needle_score = 0.0
        next_best_score = 0.0
        for rank, (bid, score) in enumerate(top_results):
            if bid == needle_id:
                needle_score = score
            elif rank == 0 or (rank == 1 and top_results[0][0] == needle_id):
                next_best_score = score

        needle_scores.append(needle_score)
        distractor_scores.append(next_best_score)
        margins.append(needle_score - next_best_score)

    latency_stats = _format_stats(latencies)
    avg_needle_score = round(mean(needle_scores), 4)
    avg_next_best_score = round(mean(distractor_scores), 4)
    avg_margin = round(mean(margins), 4)

    result = {
        "timestamp": timestamp,
        "query_text": query_text,
        "needle_content": needle_content,
        "haystack_size": haystack_size,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "latency_ms": latency_stats,
        "recall_rate": round(success_count / measured_runs, 3),
        "avg_needle_score": avg_needle_score,
        "avg_next_best_score": avg_next_best_score,
        "avg_margin": avg_margin,
        "methodology": {
            "distractor_embeddings": "synthetic random normalized vectors",
            "needle_embedding": "real embedding from Chroma DefaultEmbeddingFunction",
            "query_latency_mode": "steady-state after warmup",
        },
    }

    json_path = os.path.join(output_dir, f"benchmark-results-niah-{date_slug}.json")
    md_path = os.path.join(output_dir, f"benchmark-results-niah-{date_slug}.md")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    report_content = f"""# Needle in a Haystack (NIAH) Benchmark Results

- Timestamp: `{timestamp}`
- Query: `"{query_text}"`
- Haystack size: `{haystack_size:,}`
- Warmups: `{warmup_runs}`
- Measured runs: `{measured_runs}`

## Retrieval

- Recall rate: `{result["recall_rate"]:.3f}`
- Average needle score: `{avg_needle_score:.4f}`
- Average highest distractor score: `{avg_next_best_score:.4f}`
- Average precision margin: `{avg_margin:.4f}`

## Latency

- Mean: `{latency_stats["mean_ms"]:.2f} ms`
- Median: `{latency_stats["median_ms"]:.2f} ms`
- P95: `{latency_stats["p95_ms"]:.2f} ms`

## Methodology Notes

- This is a repeated single-query retrieval benchmark, not a broad accuracy benchmark.
- Distractor vectors are synthetic; the needle uses a real embedding.
- Timings are steady-state query timings after warmup, not first-query cold-path timings.
"""

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(report_content)

    print(f"Saved JSON report to {json_path}")
    print(f"Saved Markdown report to {md_path}")

    shutil.rmtree(db_dir)
    shutil.rmtree(belief_dir)


if __name__ == "__main__":
    run_niah()
