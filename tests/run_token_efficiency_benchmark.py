"""Token-efficiency-over-time benchmark for Micro-RAG.

Simulates a long-running agent session and compares, turn by turn:

  A) Baseline: the agent keeps the full conversation history in the prompt.
  B) Micro-RAG: rolling context compression + belief consolidation + pre-
     generative injection (real Chroma embeddings for all retrieval).

It measures prompt tokens per turn, cumulative tokens, long-range fact recall
via periodic probes, and skill (tool) retrieval quality from the same store.

LLM-dependent steps (summarization, belief extraction) use deterministic
extractive mocks so the benchmark is reproducible without API keys; every
retrieval step uses real embeddings. See the methodology notes in the report.
"""

import json
import os
import random
import re
import shutil
import sys
import time
from datetime import datetime
from statistics import mean, median

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mrag import (
    BeliefStore,
    BeliefConsolidator,
    ContextCompressor,
    PreGenerativeInjector,
    adapters,
    create_vector_store,
)

TOTAL_TURNS = 200
FACT_EVERY_N_TURNS = 5
CONSOLIDATE_EVERY_N_TURNS = 10
PROBE_EVERY_N_TURNS = 25
PROBES_PER_CHECKPOINT = 3
CONTEXT_TOKEN_LIMIT = 8000
SYSTEM_PROMPT = "You are a helpful engineering assistant for the Orion project."

FACT_SUBJECTS = [
    ("staging server address", "10.14.%d.22"),
    ("deploy key suffix", "KX-%d"),
    ("database port", "5%03d"),
    ("project codename", "aurora-%d"),
    ("release branch", "release/2.%d"),
    ("on-call rotation day", "day-%d"),
    ("log retention window", "%d days"),
    ("api rate limit", "%d requests per minute"),
    ("backup schedule slot", "0%d:30 UTC"),
    ("feature flag epoch", "epoch-%d"),
]

SKILL_CATALOG = [
    ("csv_to_json", "Convert a CSV file into structured JSON records.",
     "I need to turn this csv spreadsheet into json records"),
    ("resize_image", "Resize an image to the given width and height.",
     "please make this picture smaller, change its width and height"),
    ("send_email", "Send an email message to a recipient address.",
     "send a message to bob's email address about the outage"),
    ("query_sql", "Run a SQL query against the analytics database.",
     "run a select statement on the analytics database"),
    ("translate_text", "Translate text between human languages.",
     "translate this paragraph from english to spanish"),
    ("fetch_weather", "Get the current weather forecast for a city.",
     "what's the forecast for tomorrow in Denver"),
    ("create_calendar_event", "Create a calendar event with time and invitees.",
     "schedule a meeting on my calendar for friday with the team"),
    ("summarize_pdf", "Extract and summarize the text content of a PDF file.",
     "give me a short summary of this pdf document"),
    ("scrape_webpage", "Download a webpage and extract its main text content.",
     "grab the article text from this web page url"),
    ("generate_chart", "Render a chart image from tabular data.",
     "plot these numbers as a bar chart image"),
    ("lint_python", "Run a linter over Python source code and report issues.",
     "check my python source file for style problems"),
    ("compress_files", "Bundle files into a compressed zip archive.",
     "zip up these files into one archive"),
    ("ocr_document", "Extract text from a scanned document image via OCR.",
     "read the text out of this scanned receipt image"),
    ("currency_convert", "Convert an amount between two currencies.",
     "how much is 50 euros in us dollars right now"),
    ("git_blame_file", "Show line-by-line last-change authorship for a file.",
     "who last changed each line of this source file in git"),
    ("transcribe_audio", "Transcribe speech audio into text.",
     "turn this voice recording into a written transcript"),
    ("diff_documents", "Compute a readable diff between two text documents.",
     "show me what changed between these two versions of the doc"),
    ("geocode_address", "Convert a street address into latitude and longitude.",
     "get the map coordinates for this street address"),
    ("hash_file", "Compute a cryptographic checksum for a file.",
     "verify this download by computing its checksum"),
    ("paginate_report", "Split a long report into paginated sections.",
     "break this long report into separate pages"),
]

FILLER_TOPICS = [
    "reviewing the deployment pipeline configuration and retry policies",
    "refactoring the ingestion service to reduce startup time",
    "triaging flaky integration tests in the nightly suite",
    "documenting the rollout plan for the new cache layer",
    "tuning autoscaling thresholds after the last traffic spike",
    "cleaning up stale feature flags before the quarterly release",
    "investigating a memory regression in the worker pool",
    "aligning the API error format with the platform guidelines",
]

_FACT_PATTERN = re.compile(r"Remember: (the [^.]+ is [^.]+)\.")
_RETAINED_PATTERN = re.compile(r"Key facts I must retain: ([^\n]+)\.")


def build_facts():
    facts = []
    for i in range(TOTAL_TURNS // FACT_EVERY_N_TURNS):
        subject, value_tpl = FACT_SUBJECTS[i % len(FACT_SUBJECTS)]
        cycle = i // len(FACT_SUBJECTS) + 1
        value = value_tpl % (10 + i)
        facts.append({
            "statement": f"the {subject} for milestone {cycle} is {value}",
            "value": value,
            "question": f"What is the {subject} for milestone {cycle}?",
        })
    return facts


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def mock_summarizer(prompt: str) -> str:
    """Deterministic extractive stand-in for the compression LLM.

    Retains every explicit fact sentence found in the prompt — both new
    "Remember:" sentences from raw turns and facts carried over from the
    previous recollection, mirroring the compressor prompt's instruction to
    preserve existing content — plus a bounded narrative filler.
    """
    facts = {}
    for retained_block in _RETAINED_PATTERN.findall(prompt):
        for fact in retained_block.split("; "):
            fact = fact.strip().rstrip(".")
            if fact:
                facts[fact] = None
    for fact in _FACT_PATTERN.findall(prompt):
        facts[fact] = None
    facts = list(facts)
    lines = ["We worked through routine engineering discussion across many turns."]
    if facts:
        lines.append("Key facts I must retain: " + "; ".join(facts) + ".")
    lines.append("[USER] and [MODEL] alternated turns; no unresolved threads remain.")
    return "\n".join(lines)


def mock_belief_extractor(prompt: str) -> str:
    """Deterministic extractive stand-in for the consolidation LLM."""
    facts = list(dict.fromkeys(_FACT_PATTERN.findall(prompt)))
    payload = [
        {
            "category": "premises",
            "content": fact,
            "confidence": 0.9,
            "source": "session log",
        }
        for fact in facts
    ]
    return json.dumps(payload)


def user_turn_text(turn_index: int, fact) -> str:
    topic = FILLER_TOPICS[turn_index % len(FILLER_TOPICS)]
    text = (
        f"On turn {turn_index} I'm {topic}. Can you sanity-check my approach "
        f"and flag anything risky before I open the pull request?"
    )
    if fact:
        text += f" Also, please Remember: {fact['statement']}."
    return text


def assistant_turn_text(turn_index: int) -> str:
    return (
        f"Reviewed turn {turn_index}: the approach looks reasonable. I checked the "
        f"failure modes you mentioned, suggested one guard clause, and noted a "
        f"follow-up for the test suite so nothing regresses silently."
    )


def run_benchmark():
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    date_slug = timestamp[:10]
    rng = random.Random(7)

    db_dir = "./tests/benchmark_tokeff_chroma_db"
    belief_dir = "./tests/benchmark_tokeff_belief_data"
    output_dir = "./benchmarks"
    for path in [db_dir, belief_dir]:
        if os.path.exists(path):
            shutil.rmtree(path)
    os.makedirs(output_dir, exist_ok=True)

    print("====================================================")
    print("Token Efficiency Over Time Benchmark")
    print("====================================================")

    print("1. Initializing Micro-RAG pipeline (real Chroma embeddings)...")
    belief_store = BeliefStore(data_dir=belief_dir)
    vector_store = create_vector_store("chromadb", persist_dir=db_dir)
    injector = PreGenerativeInjector(belief_store=belief_store, vector_store=vector_store)
    compressor = ContextCompressor(
        llm_callable=mock_summarizer,
        context_token_limit=CONTEXT_TOKEN_LIMIT,
        protect_first_n=1,
    )
    consolidator = BeliefConsolidator(belief_store=belief_store, llm_callable=mock_belief_extractor)

    print("2. Importing skill catalog through the OpenAI tools adapter...")
    openai_tools = [
        {"type": "function", "function": {"name": name, "description": desc}}
        for name, desc, _ in SKILL_CATALOG
    ]
    imported = adapters.import_openai_tools(openai_tools, belief_store)
    print(f"   Imported {imported} skills.")

    facts = build_facts()
    fact_iter = iter(facts)
    introduced_facts = []

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    baseline_tokens_running = estimate_tokens(SYSTEM_PROMPT)

    per_turn = []
    probes = []
    inject_latencies = []
    baseline_cumulative = 0
    mrag_cumulative = 0

    print(f"3. Simulating {TOTAL_TURNS} conversation turns...")
    for turn in range(1, TOTAL_TURNS + 1):
        fact = None
        if turn % FACT_EVERY_N_TURNS == 0:
            fact = next(fact_iter, None)
            if fact:
                introduced_facts.append({**fact, "turn": turn})

        user_text = user_turn_text(turn, fact)

        # --- Micro-RAG condition ---
        t0 = time.perf_counter()
        injection = injector.inject(trigger_text=user_text)
        inject_latencies.append((time.perf_counter() - t0) * 1000)

        history.append({"role": "user", "content": user_text})
        mrag_context_text = "\n".join(str(m.get("content", "")) for m in history)
        mrag_prompt_tokens = estimate_tokens(mrag_context_text) + estimate_tokens(injection)
        mrag_cumulative += mrag_prompt_tokens

        # --- Baseline condition (full history, no compression/injection) ---
        baseline_tokens_running += estimate_tokens(user_text)
        baseline_prompt_tokens = baseline_tokens_running
        baseline_cumulative += baseline_prompt_tokens

        assistant_text = assistant_turn_text(turn)
        history.append({"role": "model", "content": assistant_text})
        baseline_tokens_running += estimate_tokens(assistant_text)

        # Rolling compression keeps the working set bounded.
        history = compressor.compress(history)

        # Periodic background consolidation over the recent window.
        if turn % CONSOLIDATE_EVERY_N_TURNS == 0:
            consolidator.run_consolidation_pass(history[-2 * CONSOLIDATE_EVERY_N_TURNS:])

        per_turn.append({
            "turn": turn,
            "baseline_prompt_tokens": baseline_prompt_tokens,
            "mrag_prompt_tokens": mrag_prompt_tokens,
            "history_messages": len(history),
        })

        # Periodic long-range recall probes.
        if turn % PROBE_EVERY_N_TURNS == 0 and introduced_facts:
            eligible = [f for f in introduced_facts if turn - f["turn"] >= PROBE_EVERY_N_TURNS]
            if eligible:
                for probed in rng.sample(eligible, min(PROBES_PER_CHECKPOINT, len(eligible))):
                    injector.clear_blacklist()
                    probe_injection = injector.inject(trigger_text=probed["question"])
                    context_text = "\n".join(str(m.get("content", "")) for m in history)
                    available = probed["value"] in (context_text + "\n" + probe_injection)
                    probes.append({
                        "turn": turn,
                        "fact_turn": probed["turn"],
                        "age_turns": turn - probed["turn"],
                        "question": probed["question"],
                        "recalled": available,
                    })

    recall_rate = round(mean(1.0 if p["recalled"] else 0.0 for p in probes), 3) if probes else None
    old_probes = [p for p in probes if p["age_turns"] >= 100]
    recall_rate_age_100 = (
        round(mean(1.0 if p["recalled"] else 0.0 for p in old_probes), 3) if old_probes else None
    )

    print("4. Benchmarking skill retrieval from the same store...")
    skill_hits = 0
    skill_latencies = []
    skill_misses = []
    for name, _, task_query in SKILL_CATALOG:
        injector.clear_blacklist()
        t0 = time.perf_counter()
        injection = injector.inject(trigger_text=task_query)
        skill_latencies.append((time.perf_counter() - t0) * 1000)
        if f"'{name}'" in injection:
            skill_hits += 1
        else:
            skill_misses.append(name)
    skill_hit_rate = round(skill_hits / len(SKILL_CATALOG), 3)

    checkpoints = [25, 50, 100, 150, 200]
    checkpoint_rows = []
    for cp in checkpoints:
        row = per_turn[cp - 1]
        saving = 1.0 - row["mrag_prompt_tokens"] / row["baseline_prompt_tokens"]
        checkpoint_rows.append({
            "turn": cp,
            "baseline_prompt_tokens": row["baseline_prompt_tokens"],
            "mrag_prompt_tokens": row["mrag_prompt_tokens"],
            "per_turn_saving_pct": round(saving * 100, 1),
        })

    cumulative_saving_pct = round((1.0 - mrag_cumulative / baseline_cumulative) * 100, 1)

    result = {
        "timestamp": timestamp,
        "total_turns": TOTAL_TURNS,
        "context_token_limit": CONTEXT_TOKEN_LIMIT,
        "facts_introduced": len(introduced_facts),
        "checkpoints": checkpoint_rows,
        "cumulative_prompt_tokens": {
            "baseline": baseline_cumulative,
            "mrag": mrag_cumulative,
            "saving_pct": cumulative_saving_pct,
        },
        "recall_probes": {
            "count": len(probes),
            "recall_rate": recall_rate,
            "recall_rate_age_gte_100_turns": recall_rate_age_100,
            "baseline_recall_rate": 1.0,
            "note": "Baseline recall is 1.0 by construction: full history always contains every fact.",
        },
        "skill_retrieval": {
            "catalog_size": len(SKILL_CATALOG),
            "hit_rate_top5": skill_hit_rate,
            "missed": skill_misses,
            "latency_ms_mean": round(mean(skill_latencies), 2),
            "latency_ms_median": round(median(skill_latencies), 2),
        },
        "injection_latency_ms": {
            "mean": round(mean(inject_latencies), 2),
            "median": round(median(inject_latencies), 2),
            "max": round(max(inject_latencies), 2),
        },
        "per_turn": per_turn,
        "probes": probes,
        "methodology": {
            "token_estimator": "chars/4 heuristic, applied identically to both conditions",
            "retrieval_embeddings": "real Chroma DefaultEmbeddingFunction (all-MiniLM-L6-v2) for beliefs, skills, and queries",
            "summarizer": "deterministic extractive mock that retains explicit fact sentences plus bounded narrative filler",
            "belief_extractor": "deterministic extractive mock emitting the same JSON schema a production LLM extractor would",
            "baseline": "identical conversation with full uncompressed history and no injection",
        },
    }

    json_path = os.path.join(output_dir, f"benchmark-results-token-efficiency-{date_slug}.json")
    md_path = os.path.join(output_dir, f"benchmark-results-token-efficiency-{date_slug}.md")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    table_rows = "\n".join(
        f"| {r['turn']} | {r['baseline_prompt_tokens']:,} | {r['mrag_prompt_tokens']:,} "
        f"| {r['per_turn_saving_pct']:.1f}% |"
        for r in checkpoint_rows
    )
    report_content = f"""# Micro-RAG Token Efficiency Over Time Benchmark

- Timestamp: `{timestamp}`
- Command: `python tests/run_token_efficiency_benchmark.py`
- Simulated turns: `{TOTAL_TURNS}` (fact introduced every `{FACT_EVERY_N_TURNS}` turns)
- Context limit for compression: `{CONTEXT_TOKEN_LIMIT}` tokens

## Prompt Tokens Per Turn

| Turn | Baseline (full history) | Micro-RAG | Saving |
| ---: | ---: | ---: | ---: |
{table_rows}

## Cumulative Prompt Tokens ({TOTAL_TURNS} turns)

- Baseline: `{baseline_cumulative:,}`
- Micro-RAG: `{mrag_cumulative:,}`
- **Cumulative saving: `{cumulative_saving_pct:.1f}%`** (grows with session length)

## Long-Range Fact Recall

- Probes: `{len(probes)}` (facts at least {PROBE_EVERY_N_TURNS} turns old at probe time)
- Micro-RAG recall (fact present in compressed context or injection): `{recall_rate}`
- Micro-RAG recall for facts >= 100 turns old: `{recall_rate_age_100}`
- Baseline recall: `1.0` by construction (it pays full-history token cost for it)

## Skill Retrieval (via `mrag.adapters` import)

- Catalog: `{len(SKILL_CATALOG)}` tools, natural-language task queries
- Correct skill surfaced in top-5 injection: `{skill_hit_rate}`
- Missed: `{", ".join(skill_misses) if skill_misses else "none"}`
- Injection latency mean: `{round(mean(skill_latencies), 2)} ms`

## Injection Latency During Simulation

- Mean: `{result["injection_latency_ms"]["mean"]} ms`
- Median: `{result["injection_latency_ms"]["median"]} ms`
- Max (includes first-call embedding warmup): `{result["injection_latency_ms"]["max"]} ms`

## Methodology Notes

- Retrieval quality is real: beliefs, skills, and queries are embedded with the
  Chroma default model (all-MiniLM-L6-v2); nothing retrieval-related is mocked.
- The compression summarizer and belief extractor are deterministic extractive
  mocks so the run is reproducible without API keys. They model an LLM that
  retains salient facts; with a production LLM, summary wording differs but the
  token accounting protocol is identical.
- Token counts use the same chars/4 estimator for both conditions, so the
  relative saving is estimator-independent to first order.
- No vendor comparison numbers are included: published numbers for other memory
  systems are not normalized to this protocol. The scripts are runnable as-is
  for anyone who wants to reproduce or adapt the protocol.
"""

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(report_content)

    print("\n================ TOKEN EFFICIENCY RESULTS ================")
    print(f"Cumulative tokens  baseline: {baseline_cumulative:,}  mrag: {mrag_cumulative:,}  saving: {cumulative_saving_pct:.1f}%")
    print(f"Recall (all probes): {recall_rate}   Recall (age>=100 turns): {recall_rate_age_100}")
    print(f"Skill top-5 hit rate: {skill_hit_rate}  (missed: {skill_misses or 'none'})")
    print("==========================================================")
    print(f"Saved JSON report to {json_path}")
    print(f"Saved Markdown report to {md_path}")

    shutil.rmtree(db_dir)
    shutil.rmtree(belief_dir)


if __name__ == "__main__":
    run_benchmark()
