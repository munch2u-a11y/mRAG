import unittest

from mrag.core.token_counting import (
    count_chat_tokens,
    count_text_tokens,
    describe_token_counter,
)


class TestTokenCounting(unittest.TestCase):
    def test_count_text_tokens_empty(self):
        self.assertEqual(count_text_tokens(""), 0)

    def test_count_text_tokens_non_empty(self):
        self.assertGreater(count_text_tokens("Hello world"), 0)

    def test_count_chat_tokens_includes_overhead(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Summarize this text."},
        ]
        raw_total = sum(count_text_tokens(message["content"]) for message in messages)
        self.assertGreater(count_chat_tokens(messages), raw_total)

    def test_describe_token_counter_has_backend(self):
        meta = describe_token_counter()
        self.assertIn(meta["backend"], {"tiktoken", "heuristic"})
        self.assertTrue(meta["source"])


if __name__ == "__main__":
    unittest.main()
