import os
import shutil
import unittest

from mrag import BeliefStore, DummyVectorStore, BeliefConsolidator


class TestClusterConsolidation(unittest.TestCase):
    """Covers the immediate, threshold-triggered rollup of a (subject,
    category) cluster once it grows large enough to dilute retrieval —
    modeled on ContextCompressor's synchronous trigger rather than a
    deferred/batched pass, so a diluting cluster never sits unaddressed."""

    def setUp(self):
        self.store_dir = "./tests/test_cluster_consolidation_data"
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)
        self.belief_store = BeliefStore(data_dir=self.store_dir)

    def tearDown(self):
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)

    def _add_fact(self, consolidator, content, category=None, instance=None, subject=None):
        fact = {"content": content}
        if category:
            fact.update(category=category, instance=instance, subject=subject)

        def mock_llm(prompt):
            import json
            return json.dumps({"facts": [fact]})

        consolidator.llm = mock_llm
        consolidator.add_conversation_turn({"content": content})

    def test_cluster_membership_tracked_and_counted(self):
        self.assertEqual(self.belief_store.tag_cluster_membership("Maria", "volunteering", "bel_1"), 1)
        self.assertEqual(self.belief_store.tag_cluster_membership("Maria", "volunteering", "bel_2"), 2)
        # Re-tagging the same belief must not double-count.
        self.assertEqual(self.belief_store.tag_cluster_membership("Maria", "volunteering", "bel_1"), 2)
        self.assertEqual(
            sorted(self.belief_store.get_cluster_members("Maria", "volunteering")),
            ["bel_1", "bel_2"],
        )
        self.assertFalse(self.belief_store.is_cluster_consolidated("Maria", "volunteering"))

    def test_mark_consolidated_persists(self):
        self.belief_store.tag_cluster_membership("Maria", "volunteering", "bel_1")
        self.assertFalse(self.belief_store.is_cluster_consolidated("Maria", "volunteering"))
        self.belief_store.mark_cluster_consolidated("Maria", "volunteering")
        self.assertTrue(self.belief_store.is_cluster_consolidated("Maria", "volunteering"))

    def test_cluster_below_threshold_does_not_trigger_rollup(self):
        consolidator = BeliefConsolidator(
            self.belief_store, lambda p: "{}", vector_store=DummyVectorStore(),
        )
        for i in range(5):  # below CLUSTER_SIZE_THRESHOLD (10)
            self._add_fact(
                consolidator, f"Maria volunteered at the shelter event #{i}.",
                category="volunteering", instance="shelter", subject="Maria",
            )

        rollups = [b for b in self.belief_store.get_all_beliefs_flat() if "rollup" in b["id"]]
        self.assertEqual(rollups, [])

    # Genuinely distinct facts (mirroring the real Maria/shelter cluster found
    # in LoCoMo) rather than a templated "#i" counter — a counter-only diff
    # is a same-fact-slot value swap to the contradiction detector (like
    # "prefers Python" vs "prefers Rust"), and gets merged as designed rather
    # than kept as separate episodic events.
    _DISTINCT_SHELTER_EVENTS = [
        "Maria donated her old car to a homeless shelter.",
        "Maria organized a meal for shelter residents.",
        "Maria comforted a lonely eight-year-old girl at the shelter.",
        "Maria gave a talk at the homeless shelter about her experiences.",
        "Maria received a medal for her volunteer work at the shelter.",
        "Maria helped shelter residents apply for jobs.",
        "Maria dropped off baked goods at the shelter.",
        "Maria played games with kids at the shelter.",
        "Maria collected clothing donations for the shelter.",
        "Maria led a fundraiser walk for the shelter.",
    ]

    def test_cluster_at_size_threshold_triggers_rollup_without_deleting_originals(self):
        rollup_calls = []

        def mock_llm_factory(sentence):
            def fn(prompt):
                import json
                if "consolidating a cluster" in prompt.lower():
                    rollup_calls.append(prompt)
                    return "Maria has volunteered extensively at a homeless shelter, including donating her car and organizing meals."
                return json.dumps({"facts": [{
                    "content": sentence,
                    "category": "volunteering", "instance": "shelter", "subject": "Maria",
                }]})
            return fn

        consolidator = BeliefConsolidator(self.belief_store, lambda p: "{}", vector_store=DummyVectorStore())
        for sentence in self._DISTINCT_SHELTER_EVENTS:  # exactly CLUSTER_SIZE_THRESHOLD (10)
            consolidator.llm = mock_llm_factory(sentence)
            consolidator.add_conversation_turn({"content": sentence})

        all_beliefs = self.belief_store.get_all_beliefs_flat()
        originals = [b for b in all_beliefs if b["content"] in self._DISTINCT_SHELTER_EVENTS]
        rollups = [b for b in all_beliefs if "rollup" in b["id"]]

        # Originals must survive untouched (no lost episodic detail).
        self.assertEqual(len(originals), len(self._DISTINCT_SHELTER_EVENTS))
        # Exactly one rollup belief was created.
        self.assertEqual(len(rollups), 1)
        self.assertIn("donating her car", rollups[0]["content"])
        # The rollup links back to every constituent for graph expansion.
        self.assertEqual(len(rollups[0]["relations"]), len(self._DISTINCT_SHELTER_EVENTS))
        self.assertTrue(self.belief_store.is_cluster_consolidated("maria", "volunteering"))

    def test_cluster_does_not_retrigger_after_consolidation(self):
        rollup_call_count = [0]
        extra_events = [
            "Maria attended a shelter board meeting.",
            "Maria mentored a new shelter volunteer.",
            "Maria organized a holiday drive for the shelter.",
        ]

        def make_llm(sentence):
            def fn(prompt):
                import json
                if "consolidating a cluster" in prompt.lower():
                    rollup_call_count[0] += 1
                    return "Rollup summary."
                return json.dumps({"facts": [{
                    "content": sentence,
                    "category": "volunteering", "instance": "shelter", "subject": "Maria",
                }]})
            return fn

        consolidator = BeliefConsolidator(self.belief_store, lambda p: "{}", vector_store=DummyVectorStore())
        for sentence in self._DISTINCT_SHELTER_EVENTS + extra_events:  # crosses threshold at 10, then keeps going
            consolidator.llm = make_llm(sentence)
            consolidator.add_conversation_turn({"content": sentence})

        # Only one rollup call across the whole run, not one per new member
        # after the cluster was already consolidated.
        self.assertEqual(rollup_call_count[0], 1)

    def test_token_threshold_triggers_rollup_below_count_threshold(self):
        """A handful of very long beliefs can crowd the injection budget just
        as effectively as 10 short ones — the token-fraction leg must catch
        that case even when CLUSTER_SIZE_THRESHOLD hasn't been reached."""
        long_sentence = "Maria volunteered at the shelter and " + ("helped organize donations and prepare meals " * 8)

        def make_llm(i):
            def fn(prompt):
                import json
                if "consolidating a cluster" in prompt.lower():
                    return "Rollup summary of Maria's extensive shelter volunteering."
                return json.dumps({"facts": [{
                    "content": f"{long_sentence} (event #{i})",
                    "category": "volunteering", "instance": "shelter", "subject": "Maria",
                }]})
            return fn

        consolidator = BeliefConsolidator(
            self.belief_store, lambda p: "{}", vector_store=DummyVectorStore(),
            max_injected_tokens=100,  # small budget so a few long facts trip the 25% threshold fast
        )
        for i in range(3):  # well below CLUSTER_SIZE_THRESHOLD (10)
            consolidator.llm = make_llm(i)
            consolidator.add_conversation_turn({"content": f"turn {i}"})

        rollups = [b for b in self.belief_store.get_all_beliefs_flat() if "rollup" in b["id"]]
        self.assertEqual(len(rollups), 1)

    def test_different_subjects_have_independent_clusters(self):
        def make_llm(subject, i):
            def fn(prompt):
                import json
                return json.dumps({"facts": [{
                    "content": f"{subject} volunteered at the shelter event #{i}.",
                    "category": "volunteering", "instance": "shelter", "subject": subject,
                }]})
            return fn

        consolidator = BeliefConsolidator(self.belief_store, lambda p: "{}", vector_store=DummyVectorStore())
        for i in range(4):
            consolidator.llm = make_llm("Maria", i)
            consolidator.add_conversation_turn({"content": f"maria turn {i}"})
        for i in range(3):
            consolidator.llm = make_llm("John", i)
            consolidator.add_conversation_turn({"content": f"john turn {i}"})

        self.assertEqual(len(self.belief_store.get_cluster_members("Maria", "volunteering")), 4)
        self.assertEqual(len(self.belief_store.get_cluster_members("John", "volunteering")), 3)


if __name__ == "__main__":
    unittest.main()
