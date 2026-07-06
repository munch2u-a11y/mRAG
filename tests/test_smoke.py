import unittest
import os
import shutil
from mrag import BeliefStore, create_vector_store, PreGenerativeInjector, ContextCompressor, BeliefConsolidator

def mock_llm_for_compression(prompt: str) -> str:
    return "User and Model discussed tests."

def mock_llm_for_consolidation(prompt: str) -> str:
    return '["User is writing smoke tests."]'

class TestMicroRAGSmoke(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "./test_mrag_data"
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
            
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_pipeline_smoke(self):
        # 1. Initialize Stores
        belief_store = BeliefStore(data_dir=self.test_dir)
        vector_store = create_vector_store("dummy")
            
        self.assertIsNotNone(belief_store)
        self.assertIsNotNone(vector_store)
        
        # 2. Add manual belief
        belief_store.add_belief("premises", "bel_1", "User loves testing.", 0.9, "Manual")
        
        # 3. Pre-generative Injection
        injector = PreGenerativeInjector(belief_store, vector_store)
        injection = injector.inject("Do you know what I love?")
        self.assertIn("User loves testing", injection)
        
        # 4. Compressor
        compressor = ContextCompressor(mock_llm_for_compression, context_token_limit=10, protect_first_n=1)
        history = [
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "model", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "model", "content": "I am fine!"},
            {"role": "user", "content": "That is good!"},
            {"role": "model", "content": "Indeed!"}
        ]
        compressed = compressor.compress(history)
        self.assertTrue(any(m.get("is_compressed_summary") for m in compressed))
        
        # 5. Consolidator: extraction runs immediately per conversation turn
        # (session-by-session), not via a separate batch pass.
        consolidator = BeliefConsolidator(belief_store, mock_llm_for_consolidation)
        combined_content = "\n".join(str(m.get("content", "")) for m in compressed)
        consolidator.add_conversation_turn({"content": combined_content})

        all_beliefs = belief_store.get_all_beliefs_flat()
        self.assertTrue(any(b.get("content") == "User is writing smoke tests." for b in all_beliefs))

    def test_beliefs_added_after_sync_are_retrievable(self):
        """The store-version fast path must not skip newly added beliefs."""
        sync_dir = self.test_dir + "_sync"
        if os.path.exists(sync_dir):
            # Defensive: a prior interrupted run (e.g. an assertion failure
            # before the cleanup line below) can leave stale cached
            # embeddings on disk that silently poison this run.
            shutil.rmtree(sync_dir)
        belief_store = BeliefStore(data_dir=sync_dir)
        vector_store = create_vector_store("dummy")
        belief_store.add_belief("premises", "b_first", "User owns a red bike.", 0.9)

        injector = PreGenerativeInjector(belief_store, vector_store)
        self.assertIn("red bike", injector.inject("What do I own?"))

        belief_store.add_belief("premises", "b_second", "User adopted a grey cat.", 0.9)
        injector.clear_blacklist()
        self.assertIn("grey cat", injector.inject("What did I adopt?"))
        shutil.rmtree(sync_dir)

    def test_repeated_query_still_returns_memories(self):
        """The anti-repetition blacklist must fall back instead of going silent."""
        repeat_dir = self.test_dir + "_repeat"
        if os.path.exists(repeat_dir):
            shutil.rmtree(repeat_dir)
        belief_store = BeliefStore(data_dir=repeat_dir)
        vector_store = create_vector_store("dummy")
        belief_store.add_belief("preferences", "p1", "User prefers Python.", 0.95)

        injector = PreGenerativeInjector(belief_store, vector_store)
        first = injector.inject("What language do I prefer?")
        second = injector.inject("What language do I prefer?")
        self.assertIn("User prefers Python", first)
        self.assertIn("User prefers Python", second)
        shutil.rmtree(repeat_dir)

    def test_compressor_skips_when_under_threshold(self):
        compressor = ContextCompressor(
            mock_llm_for_compression,
            context_token_limit=1000,
            protect_first_n=1
        )
        history = [
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "model", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "model", "content": "I am fine!"},
            {"role": "user", "content": "That is good!"},
        ]

        self.assertEqual(compressor.compress(history), history)

if __name__ == "__main__":
    unittest.main()
