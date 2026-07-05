import sys
import os

# Add parent dir to path so we can import mrag
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mrag import BeliefStore, create_vector_store, PreGenerativeInjector, ContextCompressor, BeliefConsolidator, adapters

def mock_llm_for_compression(prompt: str) -> str:
    """Mock LLM response for context compression."""
    return "User and Model discussed building an agnostic memory system using Micro-RAG."

def mock_llm_for_consolidation(prompt: str) -> str:
    """Mock LLM response for belief extraction (must return JSON array)."""
    return '''[
  {
    "category": "premises",
    "content": "User wants a fast, agnostic memory system for LangGraph.",
    "confidence": 0.9,
    "source": "Turn 1"
  }
]'''

def main():
    print("=== Initializing MicroRAG Memory System ===")
    
    # 1. Setup Data Store
    data_dir = "./mrag_data"
    belief_store = BeliefStore(data_dir=data_dir)
    
    # Select vector store backend. We use 'dummy' for this example so it runs
    # zero-dependency and retrieves beliefs deterministically without a model download.
    # Swap to 'chromadb' or 'pinecone' for production.
    vector_store = create_vector_store("dummy")

    # 1.1 Optional: Pre-seed memory store with existing agent tools/skills
    print("\n--- Seeding existing agent skills/tools ---")
    mock_openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate_tax",
                "description": "Calculate tax rate based on zip code.",
                "parameters": {"type": "object", "properties": {"zip": {"type": "string"}}}
            }
        }
    ]
    imported = adapters.import_openai_tools(mock_openai_tools, belief_store)
    print(f"Imported {imported} tool(s) from OpenAI schemas into the Belief Store.")
    
    # 2. Setup Pre-generative Injector (injects beliefs into context)
    injector = PreGenerativeInjector(belief_store=belief_store, vector_store=vector_store)
    
    # 3. Setup Context Compressor (rolling window)
    # Using a tiny limit to force compression
    compressor = ContextCompressor(llm_callable=mock_llm_for_compression, context_token_limit=10, protect_first_n=1)
    
    # 4. Setup Consolidator (nightly/background processing)
    consolidator = BeliefConsolidator(belief_store=belief_store, llm_callable=mock_llm_for_consolidation)
    
    print("\n--- Simulating User Interaction (Turn 1) ---")
    user_input = "I really need a fast, agnostic memory system that integrates with LangGraph."
    print(f"User: {user_input}")
    
    # The agent's pre-generative injection pipeline triggers BEFORE hitting the LLM
    injected_context = injector.inject(trigger_text=user_input)
    print(f"\n[Pre-Generative Injection]:\n{injected_context or '(No relevant beliefs found yet)'}")
    
    # The LLM generates a response based on the injection...
    llm_response = "I can help you build an agnostic micro-RAG."
    
    # Add to conversation history
    history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input},
        {"role": "model", "content": llm_response},
        {"role": "user", "content": "Great!"},
        {"role": "model", "content": "Let's start."},
        {"role": "user", "content": "Okay."}
    ]
    
    print("\n--- Simulating Context Compression ---")
    # Simulate a context compression due to hitting limits
    compressed_history = compressor.compress(history)
    print(f"Original history length: {len(history)} -> Compressed: {len(compressed_history)}")
    for msg in compressed_history:
        role = msg.get("role")
        is_summary = msg.get("is_compressed_summary", False)
        print(f"  [{role.upper()}] (Is Summary: {is_summary})")
        
    print("\n--- Simulating Background Belief Consolidation ---")
    consolidator.run_consolidation_pass(compressed_history)
    
    print("\n--- Simulating User Interaction (Turn 2) ---")
    user_input_2 = "What do you know about my goals?"
    print(f"User: {user_input_2}")
    
    injected_context_2 = injector.inject(trigger_text=user_input_2)
    print(f"\n[Pre-Generative Injection (Turn 2)]:\n{injected_context_2}")
    
    # Cleanup data dir
    import shutil
    shutil.rmtree(data_dir)

if __name__ == "__main__":
    main()
