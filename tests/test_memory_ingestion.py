import json
import os
import shutil
import unittest

from mrag import BeliefStore, DummyVectorStore, BeliefConsolidator, PreGenerativeInjector, MemoryIngestor
from mrag.core.memory_ingestor import chunk_text
from mrag.core.token_counting import count_text_tokens


def _sentence(i: int) -> str:
    return f"This is filler sentence number {i} about a completely mundane topic."


class TestChunking(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        text = "I adopted a corgi named Biscuit last week."
        self.assertEqual(chunk_text(text, 100), [text])

    def test_long_text_splits_near_target(self):
        text = " ".join(_sentence(i) for i in range(40))
        chunks = chunk_text(text, 100)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            # Sentence packing may finish a sentence past the target but
            # never runs away from it.
            self.assertLessEqual(count_text_tokens(chunk), 150)
        # Nothing lost: every sentence survives verbatim in some chunk.
        joined = " ".join(chunks)
        for i in range(40):
            self.assertIn(_sentence(i), joined)

    def test_giant_unpunctuated_text_hard_splits(self):
        text = "word " * 600  # no sentence boundaries at all
        chunks = chunk_text(text.strip(), 100)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(count_text_tokens(chunk), 150)

    def test_empty_text_is_no_chunks(self):
        self.assertEqual(chunk_text("", 100), [])
        self.assertEqual(chunk_text("   \n  ", 100), [])


class TestMemoryIngestor(unittest.TestCase):
    def setUp(self):
        self.store_dir = "./tests/test_memory_ingestor_data"
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)
        self.store = BeliefStore(data_dir=self.store_dir)
        self.ingestor = MemoryIngestor(self.store)

    def tearDown(self):
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)

    def test_add_event_stores_raw_text_with_metadata(self):
        ids = self.ingestor.add_event(
            "I finally finished my screenplay about lighthouse keepers!",
            source="user_input",
            timestamp="May 8, 2023",
            session_id="s1",
            turn_id="t3",
            speaker="Caroline",
        )
        self.assertEqual(len(ids), 1)
        belief = self.store.get_belief(ids[0])
        self.assertEqual(
            belief["content"],
            "[2023-05-08] Caroline: I finally finished my screenplay about lighthouse keepers!",
        )
        self.assertEqual(belief["_category"], "memory")
        self.assertEqual(belief["source"], "user_input")
        self.assertEqual(belief["session_id"], "s1")
        self.assertEqual(belief["turn_id"], "t3")
        self.assertIsNone(belief["reviewed_at"])

    def test_add_event_chunks_long_text_and_links_event(self):
        text = " ".join(_sentence(i) for i in range(40))
        ids = self.ingestor.add_event(text, source="tool_return", timestamp="May 8, 2023")
        self.assertGreater(len(ids), 1)
        beliefs = [self.store.get_belief(bid) for bid in ids]
        event_ids = {b["event_id"] for b in beliefs}
        self.assertEqual(len(event_ids), 1)  # all chunks share the event
        self.assertEqual([b["chunk_index"] for b in beliefs], list(range(len(ids))))
        self.assertTrue(all(b["chunk_count"] == len(ids) for b in beliefs))

    def test_replaying_same_event_is_noop(self):
        first = self.ingestor.add_event("Hello there.", timestamp="May 8, 2023", speaker="Bob")
        second = self.ingestor.add_event("Hello there.", timestamp="May 8, 2023", speaker="Bob")
        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])

    def test_add_turn_peels_date_and_speaker(self):
        ids = self.ingestor.add_turn(
            {"content": "Date: 8:56 pm on 20 July, 2023\nMelanie: We visited the tide pools with the kids."},
            session_id="conv1",
        )
        belief = self.store.get_belief(ids[0])
        self.assertEqual(
            belief["content"],
            "[2023-07-20 20:56] Melanie: We visited the tide pools with the kids.",
        )
        self.assertEqual(belief["speaker"], "Melanie")

    def test_memory_chunks_exempt_from_decay_pruning(self):
        self.ingestor.add_event("A permanent record line.", timestamp="May 8, 2023")
        before = [b for b in self.store.get_all_beliefs_flat() if b["_category"] == "memory"]
        for _ in range(25):  # far more passes than it takes to prune a normal belief
            self.store.decay_all_beliefs()
        after = [b for b in self.store.get_all_beliefs_flat() if b["_category"] == "memory"]
        self.assertEqual(len(after), len(before))
        self.assertEqual(after[0]["confidence"], 1.0)


class TestNightlyReview(unittest.TestCase):
    def setUp(self):
        self.store_dir = "./tests/test_nightly_review_data"
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)
        self.store = BeliefStore(data_dir=self.store_dir)
        self.ingestor = MemoryIngestor(self.store)
        self.llm_calls = []

    def tearDown(self):
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)

    def _ingest_sample(self):
        self.ingestor.add_event(
            "I started taekwondo classes yesterday, it was exhausting but great!",
            source="user_input", timestamp="May 8, 2023", speaker="John",
        )
        self.ingestor.add_event(
            "That sounds like a great new hobby, John.",
            source="assistant_output", timestamp="May 8, 2023",
        )

    def test_review_forms_layer2_beliefs_with_provenance(self):
        def mock_llm(prompt: str) -> str:
            self.llm_calls.append(prompt)
            return json.dumps({
                "beliefs": [{
                    "content": "John started taekwondo classes on May 7, 2023.",
                    "term": "taekwondo",
                    "aliases": ["martial arts"],
                    "source_indices": [1],
                    "category": "martial arts", "instance": "taekwondo", "subject": "John",
                }]
            })

        self._ingest_sample()
        consolidator = BeliefConsolidator(self.store, mock_llm, vector_store=DummyVectorStore())
        stats = consolidator.run_nightly_review()

        self.assertEqual(stats["chunks_pending"], 2)
        self.assertEqual(stats["chunks_reviewed"], 2)
        self.assertEqual(stats["beliefs_formed"], 1)
        self.assertEqual(len(self.llm_calls), 1)
        # Both chunks appear numbered in the review prompt.
        self.assertIn("1. [2023-05-08] John:", self.llm_calls[0])
        self.assertIn("2. [2023-05-08]", self.llm_calls[0])

        formed = [b for b in self.store.get_all_beliefs_flat() if b.get("source") == "nightly_review"]
        self.assertEqual(len(formed), 1)
        belief = formed[0]
        self.assertEqual(belief["layer"], 2)
        self.assertEqual(belief["term"], "taekwondo")
        self.assertEqual(belief["aliases"], ["martial arts"])
        # Timestamp prefix comes from the most recent source chunk.
        self.assertTrue(belief["content"].startswith("[2023-05-08"))
        # Provenance points at the actual memory chunk.
        refs = belief["memory_refs"]
        self.assertEqual(len(refs), 1)
        source_chunk = self.store.get_belief(refs[0])
        self.assertIn("taekwondo classes", source_chunk["content"])
        # Expansion vocabulary was learned through the existing plumbing.
        self.assertEqual(self.store.get_all_concept_expansions(), {"martial arts": ["taekwondo"]})

    def test_reviewed_chunks_not_reprocessed(self):
        def mock_llm(prompt: str) -> str:
            self.llm_calls.append(prompt)
            return '{"beliefs": []}'

        self._ingest_sample()
        consolidator = BeliefConsolidator(self.store, mock_llm, vector_store=DummyVectorStore())
        first = consolidator.run_nightly_review()
        second = consolidator.run_nightly_review()
        self.assertEqual(first["chunks_reviewed"], 2)
        self.assertEqual(second["chunks_pending"], 0)
        self.assertEqual(second["batches"], 0)
        self.assertEqual(len(self.llm_calls), 1)

    def test_failed_llm_call_leaves_chunks_unreviewed(self):
        def mock_llm(prompt: str) -> str:
            raise RuntimeError("simulated outage")

        self._ingest_sample()
        consolidator = BeliefConsolidator(self.store, mock_llm, vector_store=DummyVectorStore())
        stats = consolidator.run_nightly_review()
        self.assertEqual(stats["chunks_reviewed"], 0)
        still_pending = [
            b for b in self.store.get_all_beliefs_flat()
            if b.get("_category") == "memory" and not b.get("reviewed_at")
        ]
        self.assertEqual(len(still_pending), 2)

    def test_large_backlog_splits_into_batches(self):
        def mock_llm(prompt: str) -> str:
            self.llm_calls.append(prompt)
            return '{"beliefs": []}'

        for i in range(30):
            self.ingestor.add_event(
                " ".join(_sentence(i * 100 + j) for j in range(12)),
                timestamp="May 8, 2023", speaker="John",
            )
        consolidator = BeliefConsolidator(self.store, mock_llm, vector_store=DummyVectorStore())
        stats = consolidator.run_nightly_review(max_batch_tokens=500)
        self.assertGreater(stats["batches"], 1)
        self.assertEqual(stats["chunks_reviewed"], stats["chunks_pending"])


class TestLexiconRetrieval(unittest.TestCase):
    def setUp(self):
        self.store_dir = "./tests/test_lexicon_retrieval_data"
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)
        self.store = BeliefStore(data_dir=self.store_dir)
        self.vs = DummyVectorStore()

    def tearDown(self):
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)

    def _add_layer2(self, belief_id: str, content: str, term: str, aliases=None):
        self.store.add_belief(
            category="premises",
            belief_id=belief_id,
            content=content,
            confidence=1.0,
            source="nightly_review",
            layer=2,
            term=term,
            aliases=aliases or [],
        )

    def test_term_match_injects_layer2_belief_first(self):
        ingestor = MemoryIngestor(self.store)
        for i in range(20):
            ingestor.add_event(_sentence(i), timestamp="May 8, 2023", speaker="Filler")
        self._add_layer2(
            "bel_nr_tkd",
            "[2023-05-08] John practices taekwondo and attends classes twice a week.",
            term="taekwondo",
            aliases=["martial arts"],
        )
        injector = PreGenerativeInjector(self.store, self.vs)
        result = injector.inject("What martial arts does John do?")
        self.assertIn("taekwondo", result)
        first_bullet = [l for l in result.split("\n") if l.startswith("•")][0]
        self.assertIn("taekwondo", first_bullet)

    def test_lexicon_vocabulary_feeds_concept_expansions(self):
        self._add_layer2(
            "bel_nr_tkd",
            "[2023-05-08] John practices taekwondo and attends classes twice a week.",
            term="taekwondo",
            aliases=["martial arts"],
        )
        injector = PreGenerativeInjector(self.store, self.vs)
        injector.sync_index()
        self.assertIn("taekwondo", injector._concept_expansions.get("martial arts", []))

    def test_no_term_beliefs_means_no_lexicon(self):
        self.store.add_belief(
            category="premises",
            belief_id="bel_plain",
            content="A plain extracted belief with no term.",
            confidence=1.0,
        )
        injector = PreGenerativeInjector(self.store, self.vs)
        injector.sync_index()
        self.assertEqual(injector._lexicon_terms, {})
        self.assertEqual(injector._match_lexicon_terms("anything at all"), [])


class TestAdjacencyExpansion(unittest.TestCase):
    def setUp(self):
        self.store_dir = "./tests/test_adjacency_data"
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)
        self.store = BeliefStore(data_dir=self.store_dir)
        self.vs = DummyVectorStore()
        self.ingestor = MemoryIngestor(self.store)

    def tearDown(self):
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)

    def _ingest_session(self):
        turns = [
            ("Melanie", "The weather has been lovely this week."),
            ("Caroline", "It really has, perfect for being outside."),
            ("Melanie", "My kids are so excited about summer break! Any fun plans for the summer?"),
            ("Caroline", "Researching adoption agencies - it's been a dream to have a family."),
            ("Melanie", "That is wonderful news, I am so happy for you."),
            ("Caroline", "Thanks! It means a lot to have your support."),
        ]
        for t_idx, (speaker, text) in enumerate(turns):
            self.ingestor.add_event(
                text, source="conversation", timestamp="May 25, 2023",
                session_id="s1", turn_id=f"t{t_idx}", speaker=speaker,
            )

    def test_neighbor_turn_is_pulled_in(self):
        self._ingest_session()
        # Distractors so the reply turn isn't trivially retrieved on its own.
        for i in range(30):
            self.ingestor.add_event(_sentence(i), timestamp="May 1, 2023",
                                    session_id="sx", turn_id=f"t{i}", speaker="Filler")
        injector = PreGenerativeInjector(self.store, self.vs)
        result = injector.inject("Any fun plans for the summer?")
        self.assertIn("Any fun plans for the summer?", result)
        # The payload reply one turn away must ride in via adjacency even
        # though the query shares no vocabulary with it.
        self.assertIn("Researching adoption agencies", result)

    def test_neighbors_are_woven_in_turn_order(self):
        self._ingest_session()
        # Distractors force the neighbor turns out of the base selection so
        # they enter via adjacency and get woven around the anchor.
        for i in range(30):
            self.ingestor.add_event(_sentence(i), timestamp="May 1, 2023",
                                    session_id="sx", turn_id=f"t{i}", speaker="Filler")
        injector = PreGenerativeInjector(self.store, self.vs)
        result = injector.inject("Any fun plans for the summer?")
        bullets = [l for l in result.split("\n") if l.startswith("•")]
        anchor_pos = next(i for i, l in enumerate(bullets) if "Any fun plans" in l)
        reply_pos = next(i for i, l in enumerate(bullets) if "Researching adoption agencies" in l)
        prev_pos = next(i for i, l in enumerate(bullets) if "perfect for being outside" in l)
        self.assertEqual(reply_pos, anchor_pos + 1)
        self.assertEqual(prev_pos, anchor_pos - 1)

    def test_budget_cap_still_respected(self):
        self._ingest_session()
        for i in range(60):
            self.ingestor.add_event(_sentence(i), timestamp="May 1, 2023",
                                    session_id="sx", turn_id=f"t{i}", speaker="Filler")
        injector = PreGenerativeInjector(self.store, self.vs)
        beliefs = injector._pull_relevant_beliefs("Any fun plans for the summer?")
        total = sum(count_text_tokens(b.get("content", "")) for b in beliefs)
        self.assertLessEqual(total, injector.max_injected_tokens)

    def test_no_metadata_store_unchanged(self):
        # Beliefs without session/turn metadata: adjacency must stay inert
        # and the full budget must go to the ranked candidates.
        for i in range(10):
            self.store.add_belief(
                category="premises", belief_id=f"bel_{i}",
                content=f"Plain extracted fact number {i} about topic {i}.",
                confidence=1.0,
            )
        injector = PreGenerativeInjector(self.store, self.vs)
        injector.sync_index()
        self.assertEqual(injector._memory_turn_index, {})
        self.assertEqual(injector._memory_event_index, {})
        beliefs = injector._pull_relevant_beliefs("topic")
        self.assertEqual(len(beliefs), 10)

    def test_multichunk_event_siblings_adjacent(self):
        long_text = "The trip started badly. " + " ".join(_sentence(i) for i in range(20)) + \
                    " In the end we camped in the forest by the lake."
        ids = self.ingestor.add_event(long_text, timestamp="May 25, 2023",
                                      session_id="s1", turn_id="t0", speaker="Melanie")
        self.assertGreater(len(ids), 1)
        injector = PreGenerativeInjector(self.store, self.vs)
        injector.sync_index()
        first = self.store.get_belief(ids[0])
        neighbors = injector._adjacent_memory_chunks(first, turn_window=1)
        self.assertIn(ids[1], [n.get("id") for n in neighbors])


if __name__ == "__main__":
    unittest.main()
