import os
import shutil
import unittest

from mrag import BeliefStore, DummyVectorStore, BeliefConsolidator, PreGenerativeInjector


class TestRelationLearning(unittest.TestCase):
    """Covers auto-appending (subject, relation) -> named-entity mappings
    (e.g. (John, friends) -> [Bob, Cindy]) from consolidated facts, so a
    query mentioning John's friends can reach beliefs about Bob and Cindy
    by name, without the query ever having to say their names itself."""

    def setUp(self):
        self.store_dir = "./tests/test_relation_learning_data"
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)
        self.belief_store = BeliefStore(data_dir=self.store_dir)

    def tearDown(self):
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)

    def test_learn_relation_expansion_creates_and_appends(self):
        self.assertTrue(self.belief_store.learn_relation_expansion("John", "friends", ["Bob", "Cindy"]))
        self.assertTrue(self.belief_store.learn_relation_expansion("john", "friends", ["Dave"]))
        # Re-learning an already-known entity is a no-op, not a duplicate.
        self.assertFalse(self.belief_store.learn_relation_expansion("John", "friends", ["bob"]))

        expansions = self.belief_store.get_all_relation_expansions()
        self.assertEqual(sorted(expansions["john"]["friends"]), ["Bob", "Cindy", "Dave"])

    def test_learn_relation_expansion_rejects_empty_input(self):
        self.assertFalse(self.belief_store.learn_relation_expansion("", "friends", ["Bob"]))
        self.assertFalse(self.belief_store.learn_relation_expansion("John", "", ["Bob"]))
        self.assertFalse(self.belief_store.learn_relation_expansion("John", "friends", []))

    def test_relation_expansions_do_not_leak_into_concept_expansions(self):
        self.belief_store.learn_relation_expansion("John", "friends", ["Bob", "Cindy"])
        self.belief_store.learn_concept_expansion("martial arts", "taekwondo")

        self.assertEqual(self.belief_store.get_all_concept_expansions(), {"martial arts": ["taekwondo"]})
        self.assertEqual(self.belief_store.get_all_relation_expansions(), {"john": {"friends": ["Bob", "Cindy"]}})

    def test_consolidator_extracts_facts_and_learns_relation_tags(self):
        def mock_llm(prompt: str) -> str:
            return """{
              "facts": [
                {"content": "John's friends are Bob and Cindy.",
                 "subject": "John", "relation": "friends", "entities": ["Bob", "Cindy"]}
              ]
            }"""

        consolidator = BeliefConsolidator(self.belief_store, mock_llm, vector_store=DummyVectorStore())
        consolidator.add_conversation_turn({"content": "John: My friends are Bob and Cindy."})

        self.assertEqual(self.belief_store.get_all_relation_expansions(), {"john": {"friends": ["Bob", "Cindy"]}})

    def test_injector_adds_relation_entities_as_search_heads(self):
        self.belief_store.learn_relation_expansion("John", "friends", ["Bob", "Cindy"])
        vs = DummyVectorStore()
        injector = PreGenerativeInjector(self.belief_store, vs)
        injector.sync_index()

        heads = injector._generate_search_heads("Who are John's friends?")
        self.assertIn("Bob", heads)
        self.assertIn("Cindy", heads)

    def test_injector_requires_both_subject_and_relation_word(self):
        """Mentioning John alone must not pull in every relation's entities —
        only when the query also uses the specific relation word."""
        self.belief_store.learn_relation_expansion("John", "friends", ["Bob", "Cindy"])
        self.belief_store.learn_relation_expansion("John", "kids", ["Jane", "Lucy"])
        vs = DummyVectorStore()
        injector = PreGenerativeInjector(self.belief_store, vs)
        injector.sync_index()

        heads = injector._generate_search_heads("What sports does John play?")
        self.assertNotIn("Bob", heads)
        self.assertNotIn("Jane", heads)

    def test_injector_handles_relation_word_pluralization_via_stemming(self):
        self.belief_store.learn_relation_expansion("John", "friend", ["Bob"])
        vs = DummyVectorStore()
        injector = PreGenerativeInjector(self.belief_store, vs)
        injector.sync_index()

        heads = injector._generate_search_heads("Who are John's friends?")
        self.assertIn("Bob", heads)

    def test_end_to_end_retrieval_reaches_named_entity_belief(self):
        """The actual point of the mechanism: a belief about Bob must be
        retrievable from a query that only says "John" and "friends"."""
        self.belief_store.learn_relation_expansion("John", "friends", ["Bob"])
        self.belief_store.add_belief("premises", "b1", "Bob works as a chef downtown.", 0.8)
        self.belief_store.add_belief("premises", "b2", "John enjoys hiking on weekends.", 0.8)

        vs = DummyVectorStore()
        injector = PreGenerativeInjector(self.belief_store, vs)
        result = injector.inject("Tell me about John's friends.")

        self.assertIn("Bob works as a chef", result)

    def test_concepts_category_still_excluded_from_indexing_with_relations_present(self):
        self.belief_store.learn_relation_expansion("John", "friends", ["Bob"])
        self.belief_store.add_belief("premises", "b1", "John enjoys hiking on weekends.", 0.8)

        vs = DummyVectorStore()
        injector = PreGenerativeInjector(self.belief_store, vs)
        result = injector.inject("What does John enjoy?")

        self.assertNotIn("relation:john:friends", result)


if __name__ == "__main__":
    unittest.main()
