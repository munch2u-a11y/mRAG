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
        self.assertEqual(resolve_context_limit(model_name="gpt-4o"), 128000)
        self.assertEqual(resolve_context_limit(model_name="claude-3-5-sonnet"), 200000)
        self.assertEqual(resolve_context_limit(model_name="gemini-1.5-flash"), 1000000)
        self.assertEqual(resolve_context_limit(model_name="llama3"), 8192)

        # 2. Test environment variable override
        os.environ["MRAG_CONTEXT_LIMIT"] = "50000"
        self.assertEqual(resolve_context_limit(model_name="gpt-4o"), 50000)
        del os.environ["MRAG_CONTEXT_LIMIT"]

        # 3. Test unknown model error
        with self.assertRaises(ValueError):
            resolve_context_limit(model_name="unknown-model-name")

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

    def test_contradiction_fact_update_merging(self):
        class MockVectorStore(DummyVectorStore):
            def __init__(self):
                super().__init__()
                self.query_result = []
            def embed_text(self, text):
                return None
            def query_top_k(self, query_embedding, k=100):
                return self.query_result

        mock_vs = MockVectorStore()

        # Seed initial staging server IP address
        self.belief_store.add_belief(
            category="propositions",
            belief_id="bel_staging_ip",
            content="The staging server is 10.14.2.1",
            confidence=0.7,
            source="Turn 1"
        )

        # Mock a semantic match (high similarity, but different IP address/content)
        mock_vs.query_result = [("bel_staging_ip", 0.92)]

        # Consolidate newer turn with updated IP address
        self.belief_store.merge_or_add_belief(
            category="propositions",
            belief_id="bel_new_staging_ip",
            content="The staging server is 10.14.9.9",
            confidence=0.8,
            source="Turn 2",
            vector_store=mock_vs
        )

        # Verify that we merged but preferred the newer content to avoid the stale IP trap,
        # and that the contradiction was NOT counted as corroborating evidence.
        beliefs = self.belief_store.get_all_beliefs_flat()
        self.assertEqual(len(beliefs), 1)
        b = beliefs[0]
        self.assertEqual(b["id"], "bel_staging_ip")
        self.assertEqual(b["content"], "The staging server is 10.14.9.9") # Updated to newer!
        self.assertEqual(b["verifications"], 1.0)  # evidence trail restarted
        self.assertEqual(b["confidence"], 0.8)     # newer statement's own confidence, no boost
        self.assertEqual(b["previous_content"], "The staging server is 10.14.2.1")

    def test_opposite_preference_supersedes_instead_of_reinforcing(self):
        """'Adam prefers Python' -> 'Adam prefers Rust' must replace, not reinforce."""
        class MockVectorStore(DummyVectorStore):
            def __init__(self):
                super().__init__()
                self.query_result = []
            def embed_text(self, text):
                return None
            def query_top_k(self, query_embedding, k=100):
                return self.query_result

        mock_vs = MockVectorStore()
        self.belief_store.add_belief(
            category="preferences",
            belief_id="bel_adam_lang",
            content="Adam says he prefers to use Python.",
            confidence=0.9,
            stability_index=0.8,
        )
        mock_vs.query_result = [("bel_adam_lang", 0.94)]

        self.belief_store.merge_or_add_belief(
            category="preferences",
            belief_id="bel_adam_lang_new",
            content="Adam says he prefers to use Rust.",
            confidence=0.7,
            vector_store=mock_vs,
        )

        beliefs = self.belief_store.get_all_beliefs_flat()
        self.assertEqual(len(beliefs), 1)
        b = beliefs[0]
        self.assertIn("Rust", b["content"])
        self.assertEqual(b["confidence"], 0.7)          # not boosted above either statement
        self.assertEqual(b["verifications"], 1.0)       # reset, not incremented
        self.assertLess(b["stability_index"], 0.8)      # flip-flop destabilizes
        self.assertIn("Python", b["previous_content"])  # audit trail kept

    def test_paraphrase_merge_boosts_and_reindexes(self):
        """Same salient tokens -> corroboration; content update drops cached embedding."""
        class MockVectorStore(DummyVectorStore):
            def __init__(self):
                super().__init__()
                self.query_result = []
            def embed_text(self, text):
                return None
            def query_top_k(self, query_embedding, k=100):
                return self.query_result

        mock_vs = MockVectorStore()
        self.belief_store.add_belief(
            category="preferences",
            belief_id="bel_py_love",
            content="Adam loves Python.",
            confidence=0.6,
            embedding=[0.1] * 4,  # simulate a cached embedding of the old wording
        )
        mock_vs.query_result = [("bel_py_love", 0.95)]

        self.belief_store.merge_or_add_belief(
            category="preferences",
            belief_id="bel_py_love_new",
            content="Adam really enjoys coding in Python.",
            confidence=0.7,
            vector_store=mock_vs,
        )

        beliefs = self.belief_store.get_all_beliefs_flat()
        self.assertEqual(len(beliefs), 1)
        b = beliefs[0]
        self.assertEqual(b["verifications"], 2.0)       # counted as corroboration
        self.assertGreater(b["confidence"], 0.6)        # boosted
        self.assertNotIn("embedding", b)                # stale embedding invalidated

    def test_vaguer_restatement_keeps_specific_content(self):
        """A vaguer near-duplicate must not erase the specific value."""
        class MockVectorStore(DummyVectorStore):
            def __init__(self):
                super().__init__()
                self.query_result = []
            def embed_text(self, text):
                return None
            def query_top_k(self, query_embedding, k=100):
                return self.query_result

        mock_vs = MockVectorStore()
        self.belief_store.add_belief(
            category="propositions",
            belief_id="bel_ip_specific",
            content="The staging server is 10.14.9.9.",
            confidence=0.8,
        )
        mock_vs.query_result = [("bel_ip_specific", 0.93)]

        self.belief_store.merge_or_add_belief(
            category="propositions",
            belief_id="bel_ip_vague",
            content="There is a staging server for the team.",
            confidence=0.6,
            vector_store=mock_vs,
        )

        b = self.belief_store.get_belief("bel_ip_specific")
        self.assertIn("10.14.9.9", b["content"])   # specific value preserved
        self.assertEqual(b["verifications"], 2.0)  # still counted as corroboration

    def test_template_channel_catches_low_similarity_contradiction(self):
        """Opposite value statements embed far apart (measured ~0.50 for
        Python vs Rust with MiniLM), so the vector channel can't see them.
        The template index must catch the swap anyway."""
        class MockVectorStore(DummyVectorStore):
            def __init__(self):
                super().__init__()
            def embed_text(self, text):
                return None
            def query_top_k(self, query_embedding, k=100):
                return [("bel_adam_lang", 0.50)]  # realistic: below any sane threshold

        self.belief_store.add_belief(
            category="preferences",
            belief_id="bel_adam_lang",
            content="Adam says he prefers to use Python.",
            confidence=0.9,
        )

        self.belief_store.merge_or_add_belief(
            category="preferences",
            belief_id="bel_adam_lang_v2",
            content="Adam says he prefers to use Rust.",
            confidence=0.7,
            vector_store=MockVectorStore(),
        )

        beliefs = self.belief_store.get_all_beliefs_flat()
        self.assertEqual(len(beliefs), 1)  # no contradictory twin belief
        b = beliefs[0]
        self.assertIn("Rust", b["content"])
        self.assertEqual(b["verifications"], 1.0)
        self.assertIn("Python", b["previous_content"])

        # Different anchor (Eve, not Adam) must NOT merge into Adam's slot.
        self.belief_store.merge_or_add_belief(
            category="preferences",
            belief_id="bel_eve_lang",
            content="Eve says he prefers to use Go.",
            confidence=0.8,
            vector_store=MockVectorStore(),
        )
        self.assertIsNotNone(self.belief_store.get_belief("bel_eve_lang"))
        self.assertIn("Rust", self.belief_store.get_belief("bel_adam_lang")["content"])

    def test_semantic_merge_respects_category_boundary(self):
        """A preference must never merge into (and overwrite) a skills belief."""
        class MockVectorStore(DummyVectorStore):
            def __init__(self):
                super().__init__()
                self.query_result = []
            def embed_text(self, text):
                return None
            def query_top_k(self, query_embedding, k=100):
                return self.query_result

        mock_vs = MockVectorStore()
        self.belief_store.add_belief(
            category="skills",
            belief_id="tool_get_weather",
            content="Tool 'get_weather': fetches the weather forecast.",
            confidence=1.0,
        )
        mock_vs.query_result = [("tool_get_weather", 0.95)]

        self.belief_store.merge_or_add_belief(
            category="preferences",
            belief_id="bel_weather_pref",
            content="User likes checking the weather forecast daily.",
            confidence=0.7,
            vector_store=mock_vs,
        )

        tool = self.belief_store.get_belief("tool_get_weather")
        self.assertIn("Tool 'get_weather'", tool["content"])  # untouched
        self.assertIsNotNone(self.belief_store.get_belief("bel_weather_pref"))  # added separately

if __name__ == "__main__":
    unittest.main()
