import unittest
from mrag.core.tagger import extract_tags, get_tag_counts

class TestTagger(unittest.TestCase):
    
    def test_example_sentence(self):
        text = "Mike loves to go camping with his friends, Joel and Terry"
        tags = extract_tags(text)
        expected = ["[person]", "[relation]", "[event]", "[person]", "[person]"]
        self.assertEqual(tags, expected)
        
        counts = get_tag_counts(text)
        self.assertEqual(counts.get("[person]"), 3)
        self.assertEqual(counts.get("[relation]"), 1)
        self.assertEqual(counts.get("[event]"), 1)

if __name__ == "__main__":
    unittest.main()
