import json
import urllib.request
import urllib.error
from typing import Dict, Any, List

from mrag import (
    BeliefStore,
    ChromaVectorStore,
    PreGenerativeInjector,
    BeliefConsolidator,
    OllamaEmbeddingFunction
)

# 1. Define the LLM Callable signature matching Callable[[str], str] using local Ollama endpoint
def local_ollama_llm(prompt: str) -> str:
    """Wrapper calling local Ollama instance for text generation."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3", # default to llama3, phi3, mistral etc.
        "prompt": prompt,
        "stream": False
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            resp_data = json.loads(response.read().decode("utf-8"))
            return resp_data.get("response", "")
    except urllib.error.URLError as e:
        # If Ollama is not running locally, return a dummy JSON array so the script doesn't crash
        # and instead shows how it behaves under mock mode.
        print(f"\n[Ollama Offline Notice]: Could not connect to local Ollama server: {e}")
        print("Falling back to simulated consolidator extraction for dry-run...\n")
        return json.dumps([
            {
                "category": "premises",
                "content": "User prefers writing micro-services in Python.",
                "confidence": 0.95,
                "source": "Chat log turn 1"
            },
            {
                "category": "propositions",
                "content": "Ollama runs on port 11434 by default.",
                "confidence": 0.9,
                "source": "Chat log turn 2"
            }
        ])


def main():
    print("====================================================")
    print("Micro-RAG Native Ollama Integration Guide")
    print("====================================================")

    # 2. Initialize the Local Belief Store
    belief_store = BeliefStore(data_dir="./examples/ollama_belief_db")
    belief_store.load_into_cache()

    # 3. Setup Ollama Embedding Function & ChromaDB
    print("Initializing ChromaVectorStore utilizing local Ollama 'nomic-embed-text' model...")
    ollama_ef = OllamaEmbeddingFunction(
        host="http://localhost:11434",
        model_name="nomic-embed-text"
    )
    
    # We pass the Ollama embedding function to Chroma
    # (If Ollama is offline, it will output logs and fallback gracefully)
    vector_store = ChromaVectorStore(
        persist_dir="./examples/ollama_chroma_db",
        collection_name="ollama_mrag_beliefs",
        embedding_function=ollama_ef
    )

    # 4. Consolidate Beliefs using the local LLM
    print("Initializing BeliefConsolidator using local Ollama model...")
    consolidator = BeliefConsolidator(
        belief_store=belief_store,
        llm_callable=local_ollama_llm
    )
    
    # Simulate a conversational turn to analyze
    chat_log = [
        {"role": "user", "content": "I am working on setting up a fully local vector database. By the way, I prefer writing micro-services in Python rather than Go."},
        {"role": "assistant", "content": "Got it. Python is great for fast prototyping and micro-RAG setup. Ollama runs on port 11434 by default."}
    ]
    
    print("Running consolidation pass (extracting beliefs from chat history)...")
    consolidator.run_consolidation_pass(chat_log)

    # 5. Pre-Generative Context Injection
    print("\nSetting up Pre-Generative Injector...")
    injector = PreGenerativeInjector(
        belief_store=belief_store,
        vector_store=vector_store,
        enable_graph_expansion=True
    )
    
    # Simulate processing an incoming user query
    user_query = "Let's build a new service. What language should I write it in?"
    print(f"Processing query: '{user_query}'")
    
    injected_context = injector.inject(trigger_text=user_query)
    
    print("\n--- Resulting Injected Context for LLM prompt ---")
    if injected_context:
        print(injected_context)
    else:
        print("(No beliefs match the query threshold. If Ollama is offline, this is normal.)")
    print("-------------------------------------------------\n")

    # Cleanup test databases
    import shutil
    for d in ["./examples/ollama_belief_db", "./examples/ollama_chroma_db"]:
        if shutil.os.path.exists(d):
            shutil.rmtree(d)

if __name__ == "__main__":
    main()
