import os
import shutil
import unittest

from mrag import BeliefStore, DummyVectorStore
from mrag.core.pre_generative_injection import (
    PreGenerativeInjector,
    DEFAULT_CONCEPT_EXPANSIONS,
    MAX_SEARCH_HEADS,
)


class TestSearchHeadGeneration(unittest.TestCase):
    """Covers the LoCoMo miss-pattern fixes:
    Pattern A (abstraction gap) and Pattern B (single-word deconstruction).
    """

    def setUp(self):
        self.store_dir = "./tests/test_retrieval_heads_data"
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)
        self.belief_store = BeliefStore(data_dir=self.store_dir)
        self.vector_store = DummyVectorStore()
        self.injector = PreGenerativeInjector(self.belief_store, self.vector_store)

    def tearDown(self):
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)

    def test_single_significant_words_become_heads(self):
        """Pattern B: a bigram-only extractor misses key nouns standing alone."""
        heads = self.injector._generate_search_heads(
            "When did Caroline give a speech at a school?"
        )
        self.assertIn("school", heads)
        self.assertIn("speech", heads)

    def test_single_word_heads_skip_stopwords_and_question_words(self):
        heads = self.injector._generate_search_heads("When did Maria visit the shelter?")
        self.assertNotIn("when", heads)
        self.assertNotIn("did", heads)
        self.assertIn("shelter", heads)
        self.assertIn("maria", heads)

    def test_concept_expansion_for_achievement(self):
        """Pattern A: abstract 'achievement' must reach concrete phrasing like 'finished'."""
        heads = self.injector._generate_search_heads(
            "What major achievement did Joanna accomplish in January 2022?"
        )
        self.assertTrue(
            {"finished", "completed", "won", "succeeded"} & set(heads),
            f"expected concrete achievement synonyms in heads, got {heads}",
        )

    def test_concept_expansion_for_relationship_status(self):
        """Pattern A: the exact case from the LoCoMo miss report."""
        heads = self.injector._generate_search_heads("What is Caroline's relationship status?")
        self.assertIn("single", heads)
        self.assertIn("married", heads)

    def test_concept_expansion_not_triggered_for_unrelated_query(self):
        heads = self.injector._generate_search_heads("Where did Caroline go camping?")
        for word in DEFAULT_CONCEPT_EXPANSIONS["relationship status"]:
            self.assertNotIn(word, heads)

    def test_custom_concept_expansions_extend_defaults(self):
        injector = PreGenerativeInjector(
            self.belief_store,
            self.vector_store,
            concept_expansions={"pet": ["dog", "cat", "adopted"]},
        )
        heads = injector._generate_search_heads("What pet does Melanie have?")
        self.assertIn("dog", heads)
        self.assertIn("cat", heads)
        # Built-in defaults must still be present alongside the custom entry.
        heads2 = injector._generate_search_heads("What is Caroline's relationship status?")
        self.assertIn("single", heads2)

    def test_head_count_is_capped(self):
        long_query = (
            "What did " + " ".join(f"word{i}" for i in range(40))
            + " achievement relationship status?"
        )
        heads = self.injector._generate_search_heads(long_query)
        self.assertLessEqual(len(heads), MAX_SEARCH_HEADS)

    def test_concept_expansion_survives_truncation_on_long_queries(self):
        """Expansion heads are the highest-value Pattern-A fix and must not be
        crowded out by single-word heads when a question is long."""
        long_query = (
            "What did " + " ".join(f"word{i}" for i in range(40))
            + " achievement relationship status?"
        )
        heads = self.injector._generate_search_heads(long_query)
        self.assertIn("single", heads)
        self.assertTrue({"finished", "completed", "won"} & set(heads))


if __name__ == "__main__":
    unittest.main()
