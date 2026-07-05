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


def _unit_vector(values):
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        return values
    return values / norm


def _format_stats(values):
    return {
        "mean_ms": round(mean(values), 2),
        "median_ms": round(median(values), 2),
        "p95_ms": round(_percentile(values, 0.95), 2),
        "min_ms": round(min(values), 2),
        "max_ms": round(max(values), 2),
    }


def _run_split_query(injector, query_text, top_k):
    injector.clear_blacklist()

    t0 = time.perf_counter()
    query_embedding = injector._vector_store.embed_text(query_text)
    t1 = time.perf_counter()

    top_k_results = injector._vector_store.query_top_k(query_embedding, k=top_k)
    t2 = time.perf_counter()

    scored_beliefs = []
    for belief_id, sim in top_k_results:
        belief = injector._belief_store.get_belief(belief_id)
        if not belief:
            continue

        relevance = belief.get("relevance", injector._belief_store.compute_relevance(belief))
        score = sim * relevance
        if score > 0.05:
            scored_beliefs.append((score, belief))

    scored_beliefs.sort(key=lambda item: item[0], reverse=True)
    beliefs = [belief for _, belief in scored_beliefs[:5]]
    t3 = time.perf_counter()

    lines = ["--- Injected Context ---"]
    for belief in beliefs:
        lines.append(f"• {belief.get('content', '')} [{belief.get('confidence', 0.5):.2f}]")
    if beliefs:
        lines.append("------------------------")
        rendered = "\n".join(lines)
    else:
        rendered = ""
    t4 = time.perf_counter()

    return {
        "embed_ms": (t1 - t0) * 1000,
        "retrieve_ms": (t2 - t1) * 1000,
        "rerank_ms": (t3 - t2) * 1000,
        "format_ms": (t4 - t3) * 1000,
        "split_total_ms": (t4 - t0) * 1000,
        "top_ids": [belief.get("id") for belief in beliefs],
        "rendered": rendered,
    }


def run_benchmark():
    num_memories = 100000
    batch_size = 5000
    top_k = 100
    num_runs = 20
    warmup_runs = 5
    query_text = "What is the status of agent state number 55000?"
    target_belief_id = "bel_55000"
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    db_dir = "./tests/benchmark_chroma_db"
    belief_dir = "./tests/benchmark_belief_data"
    output_dir = "./benchmarks"

    for path in [db_dir, belief_dir]:
        if os.path.exists(path):
            shutil.rmtree(path)
    os.makedirs(output_dir, exist_ok=True)

    print("====================================================")
    print("1. Initializing stores...")
    belief_store = BeliefStore(data_dir=belief_dir)
    belief_store._cache_loaded = True
    vector_store = create_vector_store("chromadb", persist_dir=db_dir)

    print(f"2. Building {num_memories:,} benchmark beliefs in memory...")
    for i in range(num_memories):
        belief_id = f"bel_{i}"
        belief = {
            "id": belief_id,
            "content": f"This is mock belief number {i} containing agent state.",
            "confidence": 0.8 if i % 5 == 0 else 0.5,
            "stability_index": 0.6 if i % 3 == 0 else 0.4,
            "relations": [f"bel_{(i + 1) % num_memories}"] if i % 10 == 0 else [],
            "verifications": 1.0,
        }
        belief_store._beliefs_cache[belief_id] = belief_store._normalize_belief(
            belief,
            category="premises",
        )

    print("3. Preparing a retrieval-shaped vector corpus...")
    setup_query_embedding = _unit_vector(vector_store.embed_text(query_text).astype(np.float32))
    if float(np.linalg.norm(setup_query_embedding)) == 0.0:
        raise RuntimeError(
            "Query embedding failed before benchmark setup. "
            "Chroma DefaultEmbeddingFunction returned a zero vector, which usually means "
            "the ONNX model assets are unavailable in this environment. "
            "Pre-cache the embedding model or run the benchmark where the embedder can initialize."
        )
    rng = np.random.default_rng(7)

    index_started = time.perf_counter()
    for batch_start in range(0, num_memories, batch_size):
        batch_ids = [f"bel_{i}" for i in range(batch_start, min(batch_start + batch_size, num_memories))]
        batch_embs = []
        batch_metas = []

        for belief_id in batch_ids:
            belief_num = int(belief_id.split("_")[1])
            if belief_id == target_belief_id:
                emb = _unit_vector(setup_query_embedding + rng.normal(0.0, 0.001, 384).astype(np.float32))
            elif abs(belief_num - 55000) <= 2:
                emb = _unit_vector(setup_query_embedding + rng.normal(0.0, 0.01, 384).astype(np.float32))
            else:
                emb = _unit_vector(rng.normal(0.0, 1.0, 384).astype(np.float32))

            belief_store._beliefs_cache[belief_id]["embedding"] = emb.tolist()
            batch_embs.append(emb)
            batch_metas.append({"category": "premises"})

        vector_store.add_vectors(batch_ids, batch_embs, batch_metas)
        print(f"   Indexed {min(batch_start + batch_size, num_memories):,}/{num_memories:,}...")

    index_elapsed = time.perf_counter() - index_started
    print(f"   Index build complete in {index_elapsed:.2f} seconds.")

    injector = PreGenerativeInjector(belief_store=belief_store, vector_store=vector_store)
    injector._indexed_belief_ids.update(belief_store._beliefs_cache.keys())

    print("\n4. Warming query path...")
    for _ in range(warmup_runs):
        injector.clear_blacklist()
        injector.inject(query_text)

    print("5. Running steady-state benchmark...")
    inject_total_times = []
    embed_times = []
    retrieve_times = []
    rerank_times = []
    format_times = []
    split_total_times = []
    target_hit_count = 0
    sample_top_ids = []

    for _ in range(num_runs):
        injector.clear_blacklist()
        t0 = time.perf_counter()
        injection_text = injector.inject(query_text)
        inject_total_times.append((time.perf_counter() - t0) * 1000)

        split_result = _run_split_query(injector, query_text, top_k)
        embed_times.append(split_result["embed_ms"])
        retrieve_times.append(split_result["retrieve_ms"])
        rerank_times.append(split_result["rerank_ms"])
        format_times.append(split_result["format_ms"])
        split_total_times.append(split_result["split_total_ms"])

        if target_belief_id in split_result["top_ids"]:
            target_hit_count += 1
        if not sample_top_ids:
            sample_top_ids = split_result["top_ids"]
        if not injection_text:
            raise RuntimeError("Injector returned empty output during benchmark run.")

    stats = {
        "embed_ms": _format_stats(embed_times),
        "retrieve_ms": _format_stats(retrieve_times),
        "rerank_ms": _format_stats(rerank_times),
        "format_ms": _format_stats(format_times),
        "split_total_ms": _format_stats(split_total_times),
        "inject_total_ms": _format_stats(inject_total_times),
    }

    result = {
        "timestamp": timestamp,
        "num_memories": num_memories,
        "batch_size": batch_size,
        "num_runs": num_runs,
        "warmup_runs": warmup_runs,
        "top_k": top_k,
        "query_text": query_text,
        "target_belief_id": target_belief_id,
        "index_build_seconds": round(index_elapsed, 2),
        "methodology": {
            "vector_store": "ChromaDB PersistentClient with DefaultEmbeddingFunction",
            "query_embedding": "real all-MiniLM-L6-v2 embedding via Chroma DefaultEmbeddingFunction",
            "corpus_vectors": "synthetic 384d vectors; target and nearby beliefs are seeded near the real query embedding to make retrieval semantics testable at scale",
            "write_path_included_in_query_latency": False,
            "steady_state_path_measured": "PreGenerativeInjector.inject() after index warmup",
        },
        "stats": stats,
        "target_hit_rate": round(target_hit_count / num_runs, 3),
        "sample_top_ids": sample_top_ids,
    }

    date_slug = timestamp[:10]
    json_path = os.path.join(output_dir, f"benchmark-results-100k-{date_slug}.json")
    md_path = os.path.join(output_dir, f"benchmark-results-100k-{date_slug}.md")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    report_content = f"""# Micro-RAG 100k Benchmark Results

- Timestamp: `{timestamp}`
- Command: `python tests/run_100k_benchmark.py`
- Vector store: `ChromaDB PersistentClient`
- Query embedding model: `DefaultEmbeddingFunction / all-MiniLM-L6-v2`
- Corpus size: `{num_memories:,}` beliefs
- Query runs: `{num_runs}` after `{warmup_runs}` warmups
- Index build time: `{index_elapsed:.2f} s`

## Steady-State Latency

| Stage | Mean | Median | P95 |
| :--- | ---: | ---: | ---: |
| Embed | {stats["embed_ms"]["mean_ms"]:.2f} ms | {stats["embed_ms"]["median_ms"]:.2f} ms | {stats["embed_ms"]["p95_ms"]:.2f} ms |
| Retrieve top-{top_k} | {stats["retrieve_ms"]["mean_ms"]:.2f} ms | {stats["retrieve_ms"]["median_ms"]:.2f} ms | {stats["retrieve_ms"]["p95_ms"]:.2f} ms |
| Rerank | {stats["rerank_ms"]["mean_ms"]:.2f} ms | {stats["rerank_ms"]["median_ms"]:.2f} ms | {stats["rerank_ms"]["p95_ms"]:.2f} ms |
| Format | {stats["format_ms"]["mean_ms"]:.2f} ms | {stats["format_ms"]["median_ms"]:.2f} ms | {stats["format_ms"]["p95_ms"]:.2f} ms |
| Split total | {stats["split_total_ms"]["mean_ms"]:.2f} ms | {stats["split_total_ms"]["median_ms"]:.2f} ms | {stats["split_total_ms"]["p95_ms"]:.2f} ms |
| Actual `inject()` total | {stats["inject_total_ms"]["mean_ms"]:.2f} ms | {stats["inject_total_ms"]["median_ms"]:.2f} ms | {stats["inject_total_ms"]["p95_ms"]:.2f} ms |

## Sanity Checks

- Target belief hit rate in top-5 after rerank: `{target_hit_count}/{num_runs}` (`{result["target_hit_rate"]:.3f}`)
- Sample top belief ids: `{", ".join(sample_top_ids)}`

## Methodology Notes

- Query latency measures the steady-state read path, not ingestion.
- The query embedding is real. The 100k stored corpus vectors are synthetic but shaped so the benchmark exercises realistic ANN retrieval and reranking cost at scale.
- This report intentionally does not include vendor comparison numbers, because those published results are usually end-to-end and not normalized to this exact benchmark protocol.
"""

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(report_content)

    print("\n================ BENCHMARK RESULTS (100k memories) ================")
    print(f"Inject total mean: {stats['inject_total_ms']['mean_ms']:.2f} ms")
    print(f"Inject total p95:  {stats['inject_total_ms']['p95_ms']:.2f} ms")
    print(f"Retrieve mean:     {stats['retrieve_ms']['mean_ms']:.2f} ms")
    print(f"Target hit rate:   {target_hit_count}/{num_runs}")
    print("====================================================================")
    print(f"Saved JSON report to {json_path}")
    print(f"Saved Markdown report to {md_path}")

    shutil.rmtree(db_dir)
    shutil.rmtree(belief_dir)


if __name__ == "__main__":
    run_benchmark()
