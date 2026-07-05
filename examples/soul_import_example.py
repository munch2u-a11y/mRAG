import os
import shutil
from pathlib import Path

from mrag import BeliefStore
from mrag.adapters import import_agent_soul

def main():
    print("====================================================")
    print("Micro-RAG Soul Preset Importer Example")
    print("====================================================")
    
    # 1. Define a temporary folder containing some agent profiles
    import_dir = "./examples/sample_agent_presets"
    os.makedirs(import_dir, exist_ok=True)
    
    # 2. Write a mock Hermes soul markdown file
    hermes_path = Path(import_dir) / "hermes_identity.md"
    hermes_path.write_text("""# Self-Identity
- I am an autonomous coding assistant optimized for high-performance agent workflows.
- I specialize in systems engineering, backend pipelines, and database optimization.

## Tone & Voice Preferences
- Always respond in a clear, technical, and concise manner.
- Prefer technical correctness over politeness.

## System Propositions
- Python 3.12 is the primary execution environment.
- Micro-RAG is the default agentic memory management system.
""", encoding="utf-8")

    # 3. Write a mock OpenClaw JSON config file
    openclaw_path = Path(import_dir) / "openclaw_profile.json"
    openclaw_path.write_text("""[
        {
            "category": "premises",
            "content": "Running locally on sandboxed Linux.",
            "confidence": 0.95
        },
        {
            "category": "propositions",
            "content": "ChromaDB port defaults to 8000.",
            "confidence": 0.85
        }
    ]""", encoding="utf-8")

    # 4. Initialize BeliefStore
    db_dir = "./examples/agent_belief_store"
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
        
    belief_store = BeliefStore(data_dir=db_dir)
    
    # 5. Import the files
    print(f"Importing preset profiles from '{import_dir}'...")
    imported_count = import_agent_soul(import_dir, belief_store)
    print(f"Successfully imported {imported_count} beliefs into the BeliefStore!\n")
    
    # 6. Retrieve and display the imported beliefs
    belief_store.load_into_cache()
    all_beliefs = belief_store.get_all_beliefs_flat()
    
    print("Imported Belief Index:")
    for b in all_beliefs:
        print(f"  - [{b['_category'].upper()}] (Source: {b['source']}) | {b['content']}")

    # Cleanup temporary folders
    shutil.rmtree(import_dir)
    shutil.rmtree(db_dir)

if __name__ == "__main__":
    main()
