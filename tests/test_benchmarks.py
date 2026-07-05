import unittest
import timeit
import os
import shutil
from mrag import BeliefStore, create_vector_store, PreGenerativeInjector, ContextCompressor

def mock_llm_for_compression(prompt: str) -> str:
    # Simulates a basic fast local response or network stub
    return "User and Model discussed benchmarking."

class TestMicroRAGBenchmarks(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "./test_benchmark_data"
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
            
        cls.belief_store = BeliefStore(data_dir=cls.test_dir)
        cls.vector_store = create_vector_store("dummy")
            
        cls.injector = PreGenerativeInjector(cls.belief_store, cls.vector_store)
        
        # Populate with 1000 beliefs to test latency
        print("Populating belief store with 1,000 items for benchmarking...")
        for i in range(1000):
            cls.belief_store.add_belief("premises", f"bel_{i}", f"This is mock belief number {i}.", 0.9, "Benchmark")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_injector_latency(self):
        # Measure how long it takes to filter and inject from 1000 beliefs
        def run_inject():
            self.injector.inject("Tell me about mock belief number 500.")
            
        # Run 10 times
        time_taken = timeit.timeit(run_inject, number=10)
        avg_ms = (time_taken / 10) * 1000
        print(f"\n[Benchmark] Pre-Generative Injection (1000 beliefs): {avg_ms:.2f} ms per call")
        
        # Assert it's reasonably fast (e.g. under 500ms on most hardware even with heavy mock)
        self.assertLess(avg_ms, 1500)

    def test_compression_overhead(self):
        compressor = ContextCompressor(mock_llm_for_compression, context_token_limit=100, protect_first_n=1)
        
        # Create a large history
        history = [{"role": "system", "content": "System setup."}]
        for i in range(50):
            history.append({"role": "user", "content": f"User message {i}" * 10})
            history.append({"role": "model", "content": f"Model response {i}" * 10})
            
        def run_compress():
            compressor.compress(history)
            
        time_taken = timeit.timeit(run_compress, number=10)
        avg_ms = (time_taken / 10) * 1000
        print(f"[Benchmark] Context Compression Overhead (100 msgs): {avg_ms:.2f} ms per call")
        
        # Assert overhead of algorithm (excluding actual LLM network time) is fast
        self.assertLess(avg_ms, 500)

if __name__ == "__main__":
    unittest.main()
