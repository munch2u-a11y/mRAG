import os
import shutil
import unittest
from pathlib import Path

from mrag import BeliefStore
from mrag.adapters.soul_importer import import_agent_soul, parse_file

class TestSoulImporter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "./tests/test_soul_import_data"
        os.makedirs(cls.test_dir, exist_ok=True)
        
        # 1. Create a sample markdown soul file
        cls.md_path = Path(cls.test_dir) / "hermes_soul.md"
        cls.md_path.write_text("""# Identity
- I am an autonomous daemon named Hermes.
- My goal is to assist developers.

## Personality & Ethos
- Keep all replies extremely direct and tech-focused.
- Do not apologize under any circumstances.

## Core Facts
- Python is preferred over other coding languages.
""", encoding="utf-8")

        # 2. Create a sample JSON configuration
        cls.json_path = Path(cls.test_dir) / "openclaw_config.json"
        cls.json_path.write_text("""[
            {
                "category": "premises",
                "content": "Running locally on a Linux sandbox.",
                "confidence": 0.95
            },
            {
                "category": "propositions",
                "content": "Local database port defaults to 5432.",
                "confidence": 0.8
            }
        ]""", encoding="utf-8")

        cls.store_dir = "./tests/test_soul_store_db"
        cls.belief_store = BeliefStore(data_dir=cls.store_dir)

    @classmethod
    def tearDownClass(cls):
        for d in [cls.test_dir, cls.store_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)

    def test_markdown_parsing(self):
        items = parse_file(self.md_path)
        # Should have extracted 5 statements:
        # - 2 under Identity (category: premises)
        # - 2 under Personality (category: preferences)
        # - 1 under Core Facts (category: propositions)
        self.assertEqual(len(items), 5)
        
        premises = [i for i in items if item_category_matches(i, "premises")]
        preferences = [i for i in items if item_category_matches(i, "preferences")]
        props = [i for i in items if item_category_matches(i, "propositions")]
        
        self.assertEqual(len(premises), 2)
        self.assertEqual(len(preferences), 2)
        self.assertEqual(len(props), 1)

    def test_structured_parsing(self):
        items = parse_file(self.json_path)
        self.assertEqual(len(items), 2)
        self.assertTrue(any(i.category == "premises" for i in items))
        self.assertTrue(any(i.category == "propositions" for i in items))

    def test_import_agent_soul(self):
        count = import_agent_soul(self.test_dir, self.belief_store)
        self.assertEqual(count, 7) # 5 md + 2 json
        
        # Verify lookups from belief store
        self.belief_store.load_into_cache()
        all_beliefs = self.belief_store.get_all_beliefs_flat()
        self.assertEqual(len(all_beliefs), 7)
        
        # Check source tags
        self.assertTrue(all("imported_" in b.get("source", "") for b in all_beliefs))

def item_category_matches(item, expected):
    return item.category == expected

if __name__ == "__main__":
    unittest.main()
