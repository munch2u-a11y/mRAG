import unittest
from datetime import datetime, timedelta
from mrag.core.context_compressor import ContextCompressor, SUMMARY_PREFIX

def mock_llm(prompt: str) -> str:
    return "Recollection summary."

class TestCompressorSession(unittest.TestCase):
    
    def test_window_limit_trigger(self):
        # Window-limit trigger protects tail turns from being summarized
        compressor = ContextCompressor(mock_llm, context_token_limit=800, protect_first_n=1)
        
        # Make tokens exceed 65% of 800 (i.e. 520 tokens) to trigger should_compress
        messages = [{"role": "system", "content": "system instruction"}]
        for i in range(35):
            messages.append({"role": "user", "content": f"Turn {i}: details about testing mRAG and ensuring it works properly."})
            
        compressed = compressor.compress(messages, is_end_of_session=False)
        self.assertTrue(any(m.get("is_compressed_summary") for m in compressed))
        
        # Verify tail turns are preserved at the end verbatim (last 10 turns)
        tail_msgs = [m for m in compressed if not m.get("is_compressed_summary") and m.get("role") != "system" and not m.get("is_session_divider")]
        self.assertTrue(len(tail_msgs) <= 10)
        self.assertTrue(len(tail_msgs) >= 1)

    def test_end_of_session_trigger(self):
        # End of session triggers compression even if under token limit, and includes divider
        compressor = ContextCompressor(mock_llm, context_token_limit=10000, protect_first_n=1)
        
        # Under threshold tokens, but is_end_of_session=True
        last_date = datetime.now() - timedelta(hours=3)
        last_date_str = last_date.strftime("%Y-%m-%d %H:%M")
        
        messages = [
            {"role": "system", "content": "system instruction"},
            {"role": "user", "content": "We talked about skiing yesterday."},
            {"role": "model", "content": "Skiing is fun!"},
            {"role": "user", "content": "Yes, I love it."},
            {"role": "model", "content": "Me too!"},
            {"role": "user", "content": f"Date: {last_date_str}\nLet's plan a trip."}
        ]
        
        compressed = compressor.compress(messages, is_end_of_session=True)
        self.assertTrue(any(m.get("is_compressed_summary") for m in compressed))
        
        # Verify we have the time divider message
        dividers = [m for m in compressed if m.get("is_session_divider")]
        self.assertEqual(len(dividers), 1)
        self.assertIn("Time Elapsed since compression: 3 hours, 0 minutes", dividers[0]["content"])

if __name__ == "__main__":
    unittest.main()
