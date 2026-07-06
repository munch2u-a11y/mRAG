# Developer Notes & Deployment Gotchas

This document tracks known issues, deployment hurdles, and best practices for developing and deploying the Micro-RAG package.

## 1. ChromaDB and NumPy C++ Compiler Dependencies

**The Issue**: `chromadb` and `numpy` (which is a dependency of `chromadb`) often require C++ build tools if a pre-compiled wheel is not available for the target architecture/OS (e.g., Alpine Linux in Docker, or specific ARM instances).

**Workarounds**:
- **Docker**: If you are deploying in a Docker container, prefer `debian:bullseye-slim` or `ubuntu` bases over `alpine` to ensure `manylinux` pre-compiled wheels are used.
- **Manual Vector Store**: The `VectorStore` in `mrag` is completely abstract. If `chromadb` is too heavy for your deployment environment, you can implement a lightweight NumPy-only, Faiss, or Pinecone backed VectorStore by inheriting from the `mrag.core.vector_store.VectorStore` abstract class.

## 2. LLM Callables for Context Compressor and Belief Consolidator

**The Issue**: The `ContextCompressor` and `BeliefConsolidator` components require an LLM to generate summaries and extract beliefs.

**Best Practice**:
- They expect a simple `Callable[[str], str]` (a function that takes a prompt string and returns a string).
- Do not pass the entire LangChain agent or LangGraph state directly to these classes. Wrap your LLM call in a simple adapter function.
- Example adapter:
  ```python
  def llm_adapter(prompt: str) -> str:
      response = chat_model.invoke([{"role": "user", "content": prompt}])
      return response.content
  ```

## 3. Integration Gotchas

- **`is_compressed_summary` metadata key**: the `ContextCompressor` tags its summary
  message with this extra key. Strict chat APIs reject unknown message fields, so strip
  it (and any other custom keys) when mapping messages to your provider's wire format.
- **Embedding cache**: the injector persists each belief's embedding under an
  `embedding` key inside the category JSON files so restarts don't re-embed everything.
  If you switch embedding backends (e.g. Chroma's 384-d default to Pinecone's 1024-d
  e5-large), delete the cached `embedding` fields (or the vector index) so beliefs are
  re-embedded at the new dimension.
- **Vector index lifecycle**: pruned/removed beliefs are skipped at query time but are
  not deleted from the vector index. For long-lived deployments, periodically rebuild
  the index (fresh collection + one `sync_index()` pass) to reclaim top-k slots.

## 4. Testing and Benchmarking

- Smoke tests and benchmarks are located in the `tests/` directory.
- We use the standard `unittest` library for smoke testing and `timeit` for latency benchmarking to minimize external dependencies.
- When running benchmarks, be aware that LLM API calls are naturally high-variance in latency. The benchmarks in `test_benchmarks.py` specifically measure the *overhead* of the `PreGenerativeInjector` query and the `ContextCompressor` string serialization, rather than network I/O. For full LLM benchmarks, ensure you set a valid API key and run against a fast model (e.g., Gemini 1.5 Flash or Claude 3 Haiku).
- `tests/run_locomo_benchmark.py` and `tests/run_longmemeval_benchmark.py` run
  the full pipeline against a real LLM and real datasets — see the README's
  "Real-conversation QA benchmarks" section. Set `GEMINI_API_KEY` via `.env`
  (copy `.env.example`) or `MRAG_CREDENTIALS_ENV=/path/to/file` if you keep
  credentials outside the repo tree.

## 5. Consolidation LLM Call Volume

`BeliefConsolidator.add_conversation_turn` issues one LLM call per turn for
fact extraction. Two additional call sites add to this, both toggleable:

- **Concept/relation-expansion tagging** rides on the same extraction call
  (no extra LLM call) — the extraction prompt asks for `category`/`instance`
  and `subject`/`relation`/`entities` fields alongside each fact.
- **Session synthesis** (`enable_session_synthesis`, default `True`) issues
  one additional LLM call per turn, but only when 3 or more facts were
  extracted from that turn. It reviews the turn's own facts for genuine
  parallel-list groups (e.g. three distinct hobbies for the same person) and
  adds one combined belief without removing the originals. Set
  `enable_session_synthesis=False` on `BeliefConsolidator` to skip this call
  entirely if the extra LLM cost isn't worth it for your use case — nothing
  downstream depends on it being present.
- **Structural cluster discovery** (`discover_and_consolidate_clusters`) is
  not automatic — call it periodically/on-demand (like a nightly job), not
  per-turn; it re-clusters a subject's entire history each time it runs, so
  calling it every turn is wasted work. It also has no benefit on a
  single-use, short-lived store (e.g. a benchmark harness that discards the
  store after one query) — skip it in that case.
