import unittest
import os
from mrag.core.vector_store import (
    create_vector_store,
    ChromaVectorStore,
    PineconeVectorStore,
    DummyVectorStore
)

class TestVectorStoreFactory(unittest.TestCase):
    
    def setUp(self):
        # Clear env vars if present
        self.old_env = os.environ.get("MRAG_VECTOR_STORE")
        if "MRAG_VECTOR_STORE" in os.environ:
            del os.environ["MRAG_VECTOR_STORE"]

    def tearDown(self):
        if self.old_env:
            os.environ["MRAG_VECTOR_STORE"] = self.old_env

    def test_factory_error_on_empty(self):
        with self.assertRaises(ValueError) as ctx:
            create_vector_store()
        
        self.assertIn("No VectorStore backend selected", str(ctx.exception))
        self.assertIn("1. 'chromadb'", str(ctx.exception))

    def test_factory_resolves_dummy(self):
        store = create_vector_store("dummy", dimension=128)
        self.assertIsInstance(store, DummyVectorStore)
        self.assertEqual(store.dimension, 128)
        
        emb = store.embed_text("test")
        self.assertEqual(emb.shape, (128,))

    def test_factory_resolves_from_env(self):
        os.environ["MRAG_VECTOR_STORE"] = "dummy"
        store = create_vector_store()
        self.assertIsInstance(store, DummyVectorStore)

    def test_factory_unsupported_backend(self):
        with self.assertRaises(ValueError) as ctx:
            create_vector_store("invalid_db")
        self.assertIn("Unsupported VectorStore backend 'invalid_db'", str(ctx.exception))

    def test_pinecone_missing_credentials(self):
        with self.assertRaises(ValueError) as ctx:
            create_vector_store("pinecone")
        self.assertIn("Pinecone VectorStore requires a valid PINECONE_API_KEY", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
