# Micro-RAG (mRAG)

Micro-RAG is a lightweight, framework-agnostic memory management system. It provides a clean decoupling of memory retrieval, belief storage, and context rolling for integration with any LLM harness or orchestrator.

## Core Philosophy

`MicroRAG` provides only the bare bones *memory components*. It replaces hard memory resets with rolling context compression and injects structured "beliefs" directly into the agent's prompt via a pre-generation pipeline.

### Components

- **BeliefStore**: A JSON-based storage engine for cognitive beliefs (premises, propositions, preferences). Evaluates importance via structural graph connections and confidence.
- **PreGenerativeInjector**: A text pre-filtering pipeline that screens incoming trigger text and weaves semantically and structurally relevant beliefs into the injected context of the agent.
- **ContextCompressor**: A rolling context window manager that summarizes middle turns while preserving recent history and specific structural turn identifiers.
- **BeliefConsolidator**: A cognitive processing pipeline that converts raw memory logs into formatted, categorized beliefs.

## Installation

To install the barebones memory system (zero external vector database dependencies):
```bash
pip install mrag
```

Or install with specific vector database extras:
```bash
pip install mrag[chromadb]  # For local ChromaDB support
pip install mrag[pinecone]  # For cloud Pinecone support
```

## Usage Example

```python
from mrag import BeliefStore, create_vector_store, PreGenerativeInjector

# 1. Setup Data Store
belief_store = BeliefStore(data_dir="./mrag_data")

# 2. Select Vector Database Backend (e.g. 'chromadb', 'pinecone', or 'dummy' for sandboxed testing)
vector_store = create_vector_store("chromadb")

# 3. Setup Pre-generative Injector Pipeline
injector = PreGenerativeInjector(belief_store=belief_store, vector_store=vector_store)

user_input = "Hello, what do you know about me?"

# 4. Inject Beliefs (Run this BEFORE calling your LLM)
injected_context = injector.inject(trigger_text=user_input)
print(injected_context)
# --- Injected Context ---
# • User prefers Python [0.95]
# ------------------------
```

See the `examples/` directory for a full simulation of LangGraph integration with compression and belief consolidation.

## Benchmarks

All numbers below were produced by the scripts in `tests/` on the code in this
repository; the full reports (JSON + Markdown, with methodology notes) live in
`benchmarks/`. Retrieval always uses real embeddings (Chroma's default
all-MiniLM-L6-v2); LLM-dependent steps in the token-efficiency protocol use
deterministic extractive mocks so every run is reproducible without API keys.

### Token efficiency over time (`run_token_efficiency_benchmark.py`)

A simulated 200-turn agent session (facts introduced every 5 turns, rolling
compression + belief consolidation + injection) versus the same session with
naive full-history prompting:

| Turn | Full history | Micro-RAG | Per-turn saving |
| ---: | ---: | ---: | ---: |
| 50   | 4,509 tokens  | 4,621 tokens | -2.5% |
| 100  | 9,048 tokens  | 5,196 tokens | 42.6% |
| 150  | 13,657 tokens | 2,106 tokens | 84.6% |
| 200  | 18,267 tokens | 3,055 tokens | 83.3% |

- **Cumulative prompt tokens over 200 turns: 65.4% fewer** (631k vs 1.82M) — and the saving keeps growing with session length.
- Long-range fact recall via compressed context + injection: **0.86** overall, **0.88** for facts more than 100 turns old (full-history baseline is 1.0 by construction, at full token cost).
- Skill retrieval: **20/20** natural-language task queries surfaced the correct imported tool in the top-5 injection.

### Retrieval quality and latency

| Benchmark | Result |
| :--- | :--- |
| Needle-in-a-haystack, 5k memories | recall **1.000**, precision margin 0.64 |
| Multi-hop associative recall, 5k distractors | both chain facts recalled **1.000**, graph expansion overhead ~0 ms |
| 100k memories, top-5 after rerank | target hit rate **20/20** |
| 100k retrieval + rerank overhead | **~3 ms** (2.8 ms ANN query + 0.2 ms rerank) |
| 100k end-to-end `inject()` | ~156 ms, dominated by CPU query embedding — swap in a faster/hosted embedder to reduce it |

Reproduce with:
```bash
python tests/run_token_efficiency_benchmark.py
python tests/run_niah_benchmark.py
python tests/run_associative_benchmark.py
python tests/run_100k_benchmark.py
```

No vendor comparison numbers are published here on purpose: other systems'
published results are not normalized to these protocols. The scripts are
self-contained, so you can run the same protocol against any system you want
to compare with.

## Testing

```bash
python -m unittest discover tests
```

The suite covers the belief store (decay, pruning, cache/index consistency),
the injector (retrieval, anti-repetition fallback, index sync), the context
compressor, the vector store factory, and all skill/soul adapters.

## Skills Adapters

If you already have existing agents with defined tools/skills, you can import them directly into Micro-RAG's `BeliefStore` as `skills` using the `mrag.adapters` module:

### OpenAI Tools Format
```python
from mrag import adapters, BeliefStore

belief_store = BeliefStore(data_dir="./mrag_data")

openai_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather info."
        }
    }
]
adapters.import_openai_tools(openai_tools, belief_store)
```

### Model Context Protocol (MCP)
```python
mcp_tools = {
    "tools": [
        {
            "name": "calculate_tax",
            "description": "Calculate tax rate based on zip code."
        }
    ]
}
adapters.import_mcp_tools(mcp_tools, belief_store)
```

### Custom YAML/JSON Directories (Hermes, OpenClaw, etc.)
You can import a directory of skill files:
```python
adapters.import_from_directory("./my_skills_dir", belief_store)
```
You can also pass a `custom_parser` to extract name/description mapping from any proprietary schema structure.

