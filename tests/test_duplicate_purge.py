import os
import shutil
import unittest

from mrag import BeliefStore, DummyVectorStore
from mrag.core.pre_generative_injection import PreGenerativeInjector, _strip_timestamp_prefix


class ControlledVectorStore(DummyVectorStore):
    """Returns pre-assigned vectors for known texts instead of hash-derived
    ones, so duplicate-threshold behavior can be tested deterministically."""

    def __init__(self, vectors_by_text):
        super().__init__()
        self._by_text = vectors_by_text

    def embed_text(self, text):
        for key, vec in self._by_text.items():
            if key in text:
                return self.to_np(vec)
        return self.to_np([0.0] * self.dimension)

    def to_np(self, vec):
        from mrag._compat import np
        padded = vec + [0.0] * (self.dimension - len(vec))
        return np.array(padded[: self.dimension], dtype=float)


class TestDuplicatePurge(unittest.TestCase):
    def setUp(self):
        self.store_dir = "./tests/test_dup_purge_data"
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)
        self.belief_store = BeliefStore(data_dir=self.store_dir)

    def tearDown(self):
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)

    def test_timestamp_prefix_is_stripped_before_embedding(self):
        captured = []

        class SpyVectorStore(DummyVectorStore):
            def embed_text(self, text):
                captured.append(text)
                return super().embed_text(text)

        vs = SpyVectorStore()
        self.belief_store.add_belief(
            "premises", "b1", "[2023-05-25 13:14] Caroline researched adoption agencies.", 0.8
        )
        injector = PreGenerativeInjector(self.belief_store, vs)
        injector.inject("What did Caroline research?")

        # The belief's own embedding call (during sync_index) must never see
        # the leading timestamp bracket.
        belief_embed_calls = [t for t in captured if "Caroline researched" in t]
        self.assertTrue(belief_embed_calls)
        for t in belief_embed_calls:
            self.assertNotIn("[2023-05-25", t)
            self.assertEqual(t, _strip_timestamp_prefix(t))

    def test_near_identical_beliefs_collapse_to_one(self):
        vs = ControlledVectorStore({
            "query": [1.0, 0.0, 0.0],
            "Maria volunteers at a homeless shelter.": [1.0, 0.0, 0.0],
            "Maria has been volunteering at a homeless shelter.": [0.99, 0.01, 0.0],
            "Maria donated her old car to a homeless shelter.": [0.0, 1.0, 0.0],
        })
        self.belief_store.add_belief("premises", "b1", "[2023-08-03 18:20] Maria volunteers at a homeless shelter.", 0.8)
        self.belief_store.add_belief("premises", "b2", "[2023-02-25 20:55] Maria has been volunteering at a homeless shelter.", 0.8)
        self.belief_store.add_belief("premises", "b3", "[2022-12-22 18:10] Maria donated her old car to a homeless shelter.", 0.8)

        injector = PreGenerativeInjector(self.belief_store, vs)
        result = injector.inject("Does Maria volunteer at a shelter? query")

        # The car-donation fact (genuinely distinct) must survive...
        self.assertIn("donated her old car", result)
        # ...but the two near-identical volunteering restatements must
        # collapse to exactly one survivor, not both.
        volunteering_lines = [
            line for line in result.split("\n")
            if "volunteer" in line.lower() and "car" not in line.lower()
        ]
        self.assertEqual(len(volunteering_lines), 1)

    def test_recency_breaks_ties_between_equally_similar_duplicates(self):
        vs = ControlledVectorStore({
            "query": [1.0, 0.0, 0.0],
            "older statement": [1.0, 0.0, 0.0],
            "newer statement": [1.0, 0.0, 0.0],  # identical similarity to query -> tie
        })
        self.belief_store.add_belief("premises", "b_old", "[2023-01-01 00:00] older statement about Maria.", 0.8)
        self.belief_store.add_belief("premises", "b_new", "[2023-06-01 00:00] newer statement about Maria.", 0.8)

        injector = PreGenerativeInjector(self.belief_store, vs)
        result = injector.inject("query about Maria")

        self.assertIn("newer statement", result)
        self.assertNotIn("older statement", result)

    def test_distinct_episodic_facts_are_not_collapsed(self):
        """The core guardrail: genuinely different events about the same
        topic must NOT be merged away just because they share subject/verb
        structure (e.g. two different screenplays)."""
        vs = ControlledVectorStore({
            "query": [1.0, 0.0, 0.0],
            "finished her first screenplay": [1.0, 0.0, 0.0],
            "started her second screenplay": [0.0, 1.0, 0.0],
        })
        self.belief_store.add_belief("premises", "b1", "[2022-01-21] Joanna finished her first screenplay.", 0.8)
        self.belief_store.add_belief("premises", "b2", "[2022-02-25] Joanna started her second screenplay.", 0.8)

        injector = PreGenerativeInjector(self.belief_store, vs)
        result = injector.inject("query about Joanna's screenplay")

        self.assertIn("first screenplay", result)
        self.assertIn("second screenplay", result)


if __name__ == "__main__":
    unittest.main()
