import os
import shutil
import unittest

from mrag import BeliefStore, BeliefConsolidator, create_vector_store

try:
    import sklearn  # noqa: F401
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def _rollup_beliefs(belief_store):
    """Real rollups live in 'premises'; the bookkeeping record written by
    record_structural_cluster lives in 'concepts' under the same source
    string, so filtering on source alone double-counts."""
    return [
        b for b in belief_store.get_all_beliefs_flat()
        if b.get("source") == "structural_cluster_discovery" and b.get("_category") == "premises"
    ]


class TestStructuralClusteringLightweight(unittest.TestCase):
    """Fast tests that don't need sklearn or real embeddings."""

    def setUp(self):
        self.store_dir = "./tests/test_structural_clustering_data"
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)
        self.belief_store = BeliefStore(data_dir=self.store_dir)
        self.consolidator = BeliefConsolidator(self.belief_store, lambda p: "Rollup summary.")

    def tearDown(self):
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)

    def test_requires_sklearn_with_clear_error_when_missing(self):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "sklearn.cluster":
                raise ImportError("simulated missing sklearn")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = fake_import
        try:
            with self.assertRaises(ImportError) as ctx:
                self.consolidator.discover_and_consolidate_clusters()
            self.assertIn("mrag[clustering]", str(ctx.exception))
        finally:
            builtins.__import__ = real_import

    def test_infer_subject_from_content(self):
        self.assertEqual(
            BeliefConsolidator._infer_subject("[2023-05-01 10:00] Evan struggled with his health."),
            "Evan",
        )
        self.assertEqual(BeliefConsolidator._infer_subject("Maria volunteers at a shelter."), "Maria")
        self.assertIsNone(BeliefConsolidator._infer_subject("a lowercase sentence with no name"))


@unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn not installed (pip install mrag[clustering])")
class TestStructuralClusteringWithRealEmbeddings(unittest.TestCase):
    """Real end-to-end coverage using actual Chroma embeddings -- synthetic
    low-dimensional hand-crafted vectors turn out to be a degenerate case for
    cosine-metric HDBSCAN (all-noise even on well-separated toy clusters),
    so real semantic embeddings are both more representative and more
    reliable for validating this feature."""

    # Real belief texts from Evan's actual store, spanning distinct topics,
    # none of them explicitly LLM-tagged with a category -- exactly the case
    # the tag-based system misses. Matches the validated simulation.
    HEALTH_FACTS = [
        "Evan struggled with Evan's health a few years ago.",
        "Evan is dealing with health issues.",
        "Evan focused on Evan's well-being instead of quick results to change Evan's health.",
        "Evan maintains a fitness routine to feel healthy and strong.",
        "Evan is currently attending physical therapy for Evan's knee injury.",
    ]
    CAR_FACTS = [
        "Evan went on a trip with Evan's family in a new Prius.",
        "Evan's old Prius broke down and Evan decided to have it repaired and sell it.",
        "Evan's new Prius broke down on December 5, 2023.",
        "Evan relies on the Prius for an active lifestyle and road trips.",
        "Evan was involved in a minor car accident on December 24, 2023.",
    ]
    NOISE_FACTS = [
        "Evan consumed ginger snaps.",
        "Evan loses Evan's keys on a weekly basis.",
        "Evan is currently searching for a new job.",
    ]

    def setUp(self):
        self.store_dir = "./tests/test_structural_clustering_data"
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)
        self.belief_store = BeliefStore(data_dir=self.store_dir)
        self.vector_store = create_vector_store("chromadb")
        self.consolidator = BeliefConsolidator(self.belief_store, lambda p: "Rollup summary.")

    def tearDown(self):
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)

    def _seed(self, prefix, contents):
        for i, content in enumerate(contents):
            emb = self.vector_store.embed_text(content)
            self.belief_store.add_belief(
                "premises", f"{prefix}_{i}", content, confidence=0.9, embedding=emb.tolist(),
            )

    def test_dense_clusters_discovered_and_rolled_up_without_deleting_originals(self):
        self._seed("health", self.HEALTH_FACTS)
        self._seed("car", self.CAR_FACTS)
        self._seed("noise", self.NOISE_FACTS)

        stats = self.consolidator.discover_and_consolidate_clusters(min_cluster_size=3)
        self.assertEqual(stats["subjects_scanned"], 1)
        self.assertGreaterEqual(stats["clusters_found"], 2)
        self.assertGreaterEqual(stats["rollups_created"], 2)

        rollups = _rollup_beliefs(self.belief_store)
        self.assertGreaterEqual(len(rollups), 2)

        # Originals must survive untouched -- no lost episodic detail.
        all_beliefs = self.belief_store.get_all_beliefs_flat()
        originals = [b for b in all_beliefs if b["id"].startswith(("health_", "car_", "noise_"))]
        self.assertEqual(len(originals), len(self.HEALTH_FACTS) + len(self.CAR_FACTS) + len(self.NOISE_FACTS))

    def test_does_not_retrigger_on_already_consolidated_cluster(self):
        self._seed("health", self.HEALTH_FACTS)
        self._seed("car", self.CAR_FACTS)

        stats1 = self.consolidator.discover_and_consolidate_clusters(min_cluster_size=3)
        self.assertGreaterEqual(stats1["rollups_created"], 1)
        first_count = len(_rollup_beliefs(self.belief_store))

        stats2 = self.consolidator.discover_and_consolidate_clusters(min_cluster_size=3)
        self.assertEqual(stats2["rollups_created"], 0)  # already covered, skipped
        self.assertEqual(len(_rollup_beliefs(self.belief_store)), first_count)  # no duplicates

    def test_below_min_cluster_size_is_skipped(self):
        self._seed("health", self.HEALTH_FACTS[:2])  # well below any reasonable threshold
        stats = self.consolidator.discover_and_consolidate_clusters(min_cluster_size=5)
        self.assertEqual(stats["rollups_created"], 0)

    def test_rollups_are_excluded_from_reclustering(self):
        """A rollup belief must not itself become raw material for a future
        clustering pass (no nested rollups)."""
        self._seed("health", self.HEALTH_FACTS)
        self._seed("car", self.CAR_FACTS)
        self.consolidator.discover_and_consolidate_clusters(min_cluster_size=3)
        rollup_count_after_first_pass = len(_rollup_beliefs(self.belief_store))
        self.assertGreaterEqual(rollup_count_after_first_pass, 1)

        # Re-running with no new beliefs must not pull existing rollups back
        # in as clustering input and produce more rollups from them.
        self.consolidator.discover_and_consolidate_clusters(min_cluster_size=3)
        self.assertEqual(len(_rollup_beliefs(self.belief_store)), rollup_count_after_first_pass)


if __name__ == "__main__":
    unittest.main()
