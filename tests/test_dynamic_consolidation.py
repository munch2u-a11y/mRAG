import os
import shutil
import unittest
from typing import Any, Dict, List

from mrag import BeliefStore, DummyVectorStore
from mrag.core.context_compressor import resolve_context_limit
from mrag.core.belief_consolidator import BeliefConsolidator

class TestDynamicConsolidation(unittest.TestCase):
    def setUp(self):
        self.store_dir = "./tests/test_dyn_store_db"
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)
            
        self.belief_store = BeliefStore(data_dir=self.store_dir)
        self.vector_store = DummyVectorStore()
        
        # Mock LLM callable that extracts simple beliefs
        self.mock_extracted_beliefs = []
        def mock_llm(prompt: str) -> str:
            import json
            return json.dumps(self.mock_extracted_beliefs)
        self.mock_llm = mock_llm

    def tearDown(self):
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)

    def test_resolve_context_limit(self):
        # 1. Test standard model name resolution
        self.assertEqual(resolve_context_limit("gpt-4o"), 128000)
        self.assertEqual(resolve_context_limit("claude-3-5-sonnet"), 200000)
        self.assertEqual(resolve_context_limit("gemini-1.5-flash"), 1000000)
        self.assertEqual(resolve_context_limit("llama3"), 8192)

        # 2. Test environment variable override
        os.environ["MRAG_CONTEXT_LIMIT"] = "50000"
        self.assertEqual(resolve_context_limit("gpt-4o"), 50000)
        del os.environ["MRAG_CONTEXT_LIMIT"]

        # 3. Test unknown model error
        with self.assertRaises(ValueError):
            resolve_context_limit("unknown-model-name")

    def test_backlog_consolidation_trigger(self):
        # Setup consolidator with small context limit so backlog triggers quickly
        # 2000 tokens * 0.4 = 800 tokens threshold (approx 3200 characters)
        consolidator = BeliefConsolidator(
            belief_store=self.belief_store,
            llm_callable=self.mock_llm,
            context_limit=2000,
            ratio=0.40,
            vector_store=self.vector_store
        )
        self.assertEqual(consolidator.backlog_threshold_tokens, 800)
        
        # Inject standard mocked extraction output
        self.mock_extracted_beliefs = [
            {
                "category": "premises",
                "content": "User lives in Oregon.",
                "confidence": 0.9,
                "source": "Turn 1"
            }
        ]
        
        # Add turns that are small (under 800 tokens total)
        consolidator.add_conversation_turn({"role": "user", "content": "Hello there."})
        self.assertEqual(len(self.belief_store.get_all_beliefs_flat()), 0) # shouldn't trigger yet
        
        # Add a large turn (exceeding 800 tokens / 3200 chars)
        large_content = "A" * 3500
        consolidator.add_conversation_turn({"role": "user", "content": large_content})
        
        # Should have auto-triggered consolidation and seeded the mock belief
        self.assertEqual(len(self.belief_store.get_all_beliefs_flat()), 1)
        self.assertEqual(consolidator._backlog_tokens, 0) # backlog cleared

    def test_semantic_deduplication_and_merging(self):
        # We will use a MockVectorStore to control query_top_k return values
        class MockVectorStore(DummyVectorStore):
            def __init__(self):
                super().__init__()
                self.query_result = []
                
            def embed_text(self, text):
                return None
                
            def query_top_k(self, query_embedding, k=100):
                return self.query_result

        mock_vs = MockVectorStore()
        
        # Add a belief manually
        self.belief_store.add_belief(
            category="preferences",
            belief_id="bel_python_pref",
            content="User prefers Python for AI engineering.",
            confidence=0.6,
            source="Turn 1",
            relations=["relation_foo"]
        )
        
        # 1. Exact Duplicate Merging
        self.belief_store.merge_or_add_belief(
            category="preferences",
            belief_id="bel_python_pref", # exact ID
            content="User prefers Python for AI engineering.",
            confidence=0.7,
            source="Turn 2",
            relations=["relation_bar"],
            vector_store=mock_vs
        )
        
        # Verify it merged instead of creating a second entry
        beliefs = self.belief_store.get_all_beliefs_flat()
        self.assertEqual(len(beliefs), 1)
        b = beliefs[0]
        self.assertEqual(b["verifications"], 2.0)
        self.assertGreater(b["confidence"], 0.6) # raised asymptotically
        self.assertEqual(sorted(b["relations"]), ["relation_bar", "relation_foo"]) # merged relations

        # 2. Semantic Duplicate Merging
        # We tell the vector store to return our existing belief with similarity 0.95 (semantic match)
        mock_vs.query_result = [("bel_python_pref", 0.95)]
        
        self.belief_store.merge_or_add_belief(
            category="preferences",
            belief_id="bel_new_python_pref", # different ID
            content="The developer likes coding AI systems in Python.", # semantically similar
            confidence=0.8,
            source="Turn 3",
            relations=["relation_baz"],
            vector_store=mock_vs
        )
        
        # Verify it still merged into the matched belief and didn't create a new one
        beliefs = self.belief_store.get_all_beliefs_flat()
        self.assertEqual(len(beliefs), 1)
        b = beliefs[0]
        self.assertEqual(b["id"], "bel_python_pref") # merged into Python pref
        self.assertEqual(b["verifications"], 3.0)
        self.assertEqual(sorted(b["relations"]), ["relation_bar", "relation_baz", "relation_foo"])

if __name__ == "__main__":
    unittest.main()
