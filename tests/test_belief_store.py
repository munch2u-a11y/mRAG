import unittest
import os
import shutil
from mrag import BeliefStore


class TestBeliefStoreDecay(unittest.TestCase):

    def setUp(self):
        self.test_dir = "./test_belief_store_data"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        self.store = BeliefStore(data_dir=self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_decay_is_gradual_for_fresh_high_confidence_beliefs(self):
        """One decay pass must not crater a freshly formed confident belief."""
        self.store.add_belief("skills", "tool_x", "Tool 'x': does x.", confidence=1.0)
        self.store.decay_all_beliefs()

        belief = self.store.get_belief("tool_x")
        self.assertIsNotNone(belief)
        self.assertGreater(belief["confidence"], 0.6)

    def test_decay_eventually_prunes_weak_beliefs(self):
        self.store.add_belief(
            "premises", "b_weak", "weak belief",
            confidence=0.3, stability_index=0.0, verifications=0.0
        )
        for _ in range(20):
            self.store.decay_all_beliefs()
        self.assertIsNone(self.store.get_belief("b_weak"))

    def test_decay_keeps_cache_consistent(self):
        """Pruned beliefs must disappear from cached lookups too."""
        self.store.add_belief(
            "premises", "b_doomed", "doomed belief",
            confidence=0.21, stability_index=0.0, verifications=0.0
        )
        self.store.add_belief("premises", "b_strong", "strong belief", confidence=0.95)
        self.store.load_into_cache()

        for _ in range(20):
            self.store.decay_all_beliefs()

        self.assertIsNone(self.store.get_belief("b_doomed"))
        cached = self.store.get_belief("b_strong")
        on_disk = next(
            b for b in self.store._read_category("premises") if b["id"] == "b_strong"
        )
        self.assertEqual(cached["confidence"], on_disk["confidence"])

    def test_get_related_uses_inbound_index_when_cached(self):
        self.store.add_belief("premises", "b_a", "belief A", relations=["b_b"])
        self.store.add_belief("premises", "b_b", "belief B")
        self.store.add_belief("premises", "b_c", "belief C", relations=["b_b"])
        self.store.load_into_cache()

        related_ids = {b["id"] for b in self.store.get_related("b_b")}
        self.assertEqual(related_ids, {"b_a", "b_c"})

        # Removing a belief must also drop its inbound edges
        self.store.remove_belief("premises", "b_c")
        related_ids = {b["id"] for b in self.store.get_related("b_b")}
        self.assertEqual(related_ids, {"b_a"})


if __name__ == "__main__":
    unittest.main()
