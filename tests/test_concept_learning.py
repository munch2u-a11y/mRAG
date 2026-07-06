import os
import shutil
import unittest

from mrag import BeliefStore, DummyVectorStore, BeliefConsolidator, PreGenerativeInjector


class TestConceptLearning(unittest.TestCase):
    """Covers auto-appending category -> instance mappings (e.g.
    "martial arts" -> "taekwondo") from consolidated facts, so retrieval
    keeps up with abstract query wording without hand-curating every
    possible category upfront."""

    def setUp(self):
        self.store_dir = "./tests/test_concept_learning_data"
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)
        self.belief_store = BeliefStore(data_dir=self.store_dir)

    def tearDown(self):
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)

    def test_learn_concept_expansion_creates_and_appends(self):
        self.assertTrue(self.belief_store.learn_concept_expansion("Martial Arts", "Taekwondo"))
        self.assertTrue(self.belief_store.learn_concept_expansion("martial arts", "kickboxing"))
        # Re-learning the same pair is a no-op, not a duplicate.
        self.assertFalse(self.belief_store.learn_concept_expansion("martial arts", "taekwondo"))

        expansions = self.belief_store.get_all_concept_expansions()
        self.assertEqual(sorted(expansions["martial arts"]), ["kickboxing", "taekwondo"])

    def test_learn_concept_expansion_rejects_empty_input(self):
        self.assertFalse(self.belief_store.learn_concept_expansion("", "taekwondo"))
        self.assertFalse(self.belief_store.learn_concept_expansion("martial arts", ""))

    def test_consolidator_extracts_facts_and_learns_category_tags(self):
        def mock_llm(prompt: str) -> str:
            return """{
              "facts": [
                {"content": "John is participating in taekwondo on December 22, 2022.",
                 "category": "martial arts", "instance": "taekwondo", "subject": "John"}
              ]
            }"""

        consolidator = BeliefConsolidator(self.belief_store, mock_llm, vector_store=DummyVectorStore())
        consolidator.add_conversation_turn({"content": "Date: December 22, 2022\nJohn: I did taekwondo today!"})

        all_beliefs = self.belief_store.get_all_beliefs_flat()
        self.assertTrue(any("taekwondo" in b["content"] for b in all_beliefs))
        self.assertEqual(self.belief_store.get_all_concept_expansions(), {"martial arts": ["taekwondo"]})

    def test_consolidator_tolerates_bare_list_response(self):
        """Older/uncooperative LLM output (a bare fact list, no tags) must
        not break extraction — tags are a bonus signal, not a requirement."""
        def mock_llm(prompt: str) -> str:
            return '["Melanie painted a lake sunrise in 2022."]'

        consolidator = BeliefConsolidator(self.belief_store, mock_llm, vector_store=DummyVectorStore())
        consolidator.add_conversation_turn({"content": "Melanie: I painted a lake sunrise!"})

        all_beliefs = self.belief_store.get_all_beliefs_flat()
        self.assertTrue(any("lake sunrise" in b["content"] for b in all_beliefs))
        self.assertEqual(self.belief_store.get_all_concept_expansions(), {})

    def test_injector_merges_learned_expansions_additively(self):
        """Learned instances must extend, not replace, the built-in default
        list for the same category."""
        self.belief_store.learn_concept_expansion("achievement", "graduated")
        vs = DummyVectorStore()
        injector = PreGenerativeInjector(self.belief_store, vs)
        injector.sync_index()

        achievement_terms = injector._concept_expansions["achievement"]
        self.assertIn("graduated", achievement_terms)
        self.assertIn("finished", achievement_terms)  # built-in default preserved

    def test_user_override_replaces_category_even_if_learned(self):
        self.belief_store.learn_concept_expansion("achievement", "graduated")
        vs = DummyVectorStore()
        injector = PreGenerativeInjector(
            self.belief_store, vs, concept_expansions={"achievement": ["custom_term"]}
        )
        injector.sync_index()

        self.assertEqual(injector._concept_expansions["achievement"], ["custom_term"])

    def test_concepts_category_not_indexed_as_retrievable_content(self):
        """The 'concepts' bookkeeping beliefs must never surface in the
        injected context as if they were real facts."""
        self.belief_store.learn_concept_expansion("martial arts", "taekwondo")
        self.belief_store.add_belief("premises", "b1", "John enjoys taekwondo practice.", 0.8)

        vs = DummyVectorStore()
        injector = PreGenerativeInjector(self.belief_store, vs)
        result = injector.inject("What martial arts has John done?")

        self.assertNotIn("martial arts]", result)  # the bare category-name belief
        self.assertIn("taekwondo practice", result)


if __name__ == "__main__":
    unittest.main()
