import json
import os
import shutil
import unittest

from mrag import BeliefStore, DummyVectorStore, BeliefConsolidator, PreGenerativeInjector


class TestSessionSynthesis(unittest.TestCase):
    """Covers the additive per-session merge pass: after extracting facts
    from a session, a second LLM call reviews them and may ADD (never
    replace) a combined belief for a genuine same-subject-same-relation
    group, e.g. several "John loves X" facts -> one "John loves X, Y, and Z"
    belief, so a query about John's activities in general has a shot at one
    compact statement instead of needing to individually retrieve every
    scattered mention.
    """

    def setUp(self):
        self.store_dir = "./tests/test_session_synthesis_data"
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)
        self.belief_store = BeliefStore(data_dir=self.store_dir)

    def tearDown(self):
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)

    def _make_consolidator(self, extraction_response, synthesis_response):
        calls = {"n": 0}

        def mock_llm(prompt):
            calls["n"] += 1
            if "facts" in prompt.lower() and "Turns:" in prompt:
                return extraction_response
            return synthesis_response

        consolidator = BeliefConsolidator(self.belief_store, mock_llm, vector_store=DummyVectorStore())
        return consolidator, calls

    def test_merged_belief_added_without_removing_originals(self):
        extraction = json.dumps({"facts": [
            {"content": "John loves hiking."},
            {"content": "John loves cooking Italian food."},
            {"content": "John loves reading science fiction novels."},
        ]})
        synthesis = json.dumps({"merged": [
            {"content": "John loves hiking, cooking Italian food, and reading science fiction novels.",
             "source_indices": [1, 2, 3]}
        ]})
        consolidator, calls = self._make_consolidator(extraction, synthesis)
        consolidator.add_conversation_turn({"content": "Date: June 1, 2023\nJohn: I love hiking, cooking, and reading."})

        all_beliefs = self.belief_store.get_all_beliefs_flat()
        contents = [b["content"] for b in all_beliefs]

        # Originals must survive untouched.
        self.assertTrue(any("John loves hiking." in c for c in contents))
        self.assertTrue(any("cooking Italian food" in c for c in contents))
        self.assertTrue(any("reading science fiction novels" in c for c in contents))
        # The merged belief must also be present, additively.
        merged = [b for b in all_beliefs if b.get("source") == "turn_synthesis"]
        self.assertEqual(len(merged), 1)
        self.assertIn("hiking, cooking Italian food, and reading", merged[0]["content"])
        # Linked back to its constituents for traceability.
        self.assertEqual(len(merged[0].get("relations", [])), 3)

    def test_no_merge_pass_triggered_with_fewer_than_three_facts(self):
        extraction = json.dumps({"facts": [
            {"content": "John loves hiking."},
            {"content": "John loves cooking."},
        ]})
        consolidator, calls = self._make_consolidator(extraction, json.dumps({"merged": []}))
        consolidator.add_conversation_turn({"content": "John: I love hiking and cooking."})

        self.assertEqual(calls["n"], 1)  # only the extraction call, no synthesis call
        merged = [b for b in self.belief_store.get_all_beliefs_flat() if b.get("source") == "turn_synthesis"]
        self.assertEqual(merged, [])

    def test_empty_merge_response_adds_nothing(self):
        extraction = json.dumps({"facts": [
            {"content": "John loves hiking."},
            {"content": "Maria went to the store."},
            {"content": "John loves cooking."},
        ]})
        consolidator, calls = self._make_consolidator(extraction, json.dumps({"merged": []}))
        consolidator.add_conversation_turn({"content": "John: I love hiking and cooking. Maria: I went to the store."})

        merged = [b for b in self.belief_store.get_all_beliefs_flat() if b.get("source") == "turn_synthesis"]
        self.assertEqual(merged, [])

    def test_malformed_synthesis_response_does_not_crash(self):
        extraction = json.dumps({"facts": [
            {"content": "John loves hiking."},
            {"content": "John loves cooking."},
            {"content": "John loves reading."},
        ]})
        consolidator, calls = self._make_consolidator(extraction, "not valid json at all")
        # Should not raise.
        consolidator.add_conversation_turn({"content": "John: I love hiking, cooking, and reading."})

        merged = [b for b in self.belief_store.get_all_beliefs_flat() if b.get("source") == "turn_synthesis"]
        self.assertEqual(merged, [])

    def test_merge_with_invalid_indices_is_skipped(self):
        extraction = json.dumps({"facts": [
            {"content": "John loves hiking."},
            {"content": "John loves cooking."},
            {"content": "John loves reading."},
        ]})
        synthesis = json.dumps({"merged": [
            {"content": "Bogus merge", "source_indices": [1, 2, 99]},
        ]})
        consolidator, calls = self._make_consolidator(extraction, synthesis)
        consolidator.add_conversation_turn({"content": "John: I love hiking, cooking, and reading."})

        merged = [b for b in self.belief_store.get_all_beliefs_flat() if b.get("source") == "turn_synthesis"]
        self.assertEqual(merged, [])

    def test_merge_with_only_two_facts_is_skipped_even_if_llm_suggests_it(self):
        """The 3-item floor is enforced client-side too, defensive against
        the LLM ignoring the "3 or more" instruction."""
        extraction = json.dumps({"facts": [
            {"content": "John loves hiking."},
            {"content": "John loves cooking."},
            {"content": "Maria went to the store."},
        ]})
        synthesis = json.dumps({"merged": [
            {"content": "John loves hiking and cooking.", "source_indices": [1, 2]},
        ]})
        consolidator, calls = self._make_consolidator(extraction, synthesis)
        consolidator.add_conversation_turn({"content": "John: I love hiking and cooking. Maria: I went to the store."})

        merged = [b for b in self.belief_store.get_all_beliefs_flat() if b.get("source") == "turn_synthesis"]
        self.assertEqual(merged, [])

    def test_retrieval_suppresses_constituents_when_merge_is_also_a_candidate(self):
        """The whole point of the merge: when both the merge and its
        constituents would otherwise be injected together, only the merge
        (which is guaranteed to name every specific detail) should survive —
        keeping both just spends extra token budget restating the same
        facts."""
        extraction = json.dumps({"facts": [
            {"content": "John loves hiking."},
            {"content": "John loves cooking."},
            {"content": "John loves reading."},
        ]})
        synthesis = json.dumps({"merged": [
            {"content": "John loves hiking, cooking, and reading.", "source_indices": [1, 2, 3]}
        ]})
        consolidator, calls = self._make_consolidator(extraction, synthesis)
        consolidator.add_conversation_turn({"content": "John: I love hiking, cooking, and reading."})

        vs = DummyVectorStore()
        injector = PreGenerativeInjector(self.belief_store, vs, max_injected_tokens=100000, token_budget_fraction=1.0)
        result = injector.inject("What does John love?")

        self.assertIn("John loves hiking, cooking, and reading.", result)
        self.assertNotIn("John loves hiking.\n", result)
        self.assertNotIn("John loves cooking.\n", result)
        self.assertNotIn("John loves reading.\n", result)


if __name__ == "__main__":
    unittest.main()
