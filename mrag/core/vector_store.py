import json
import urllib.request
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any

from mrag._compat import np

logger = logging.getLogger("mrag.core.vector_store")

class VectorStore(ABC):
    """Abstract Vector Store for embedding and similarities."""

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Embeds a single text string into a numpy array."""
        pass

    @abstractmethod
    def add_vectors(self, ids: List[str], embeddings: List[np.ndarray], metadatas: Optional[List[dict]] = None):
        """Indexes vectors in the database."""
        pass

    @abstractmethod
    def query_top_k(self, query_embedding: np.ndarray, k: int = 100) -> List[Tuple[str, float]]:
        """Queries database for the top K nearest neighbors, returning List of (id, similarity_score)."""
        pass

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two 1D numpy arrays."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class ChromaVectorStore(VectorStore):
    """Vector store using ChromaDB."""

    def __init__(self, persist_dir: Optional[str] = None, collection_name: str = "mrag_beliefs", embedding_function: Optional[Any] = None):
        try:
            import chromadb
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            
            if persist_dir:
                self._client = chromadb.PersistentClient(path=persist_dir)
            else:
                self._client = chromadb.EphemeralClient()

            self._embedder = embedding_function or DefaultEmbeddingFunction()
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._embedder,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"ChromaVectorStore initialized. Persist: {persist_dir}, Collection: {collection_name}")
        except ImportError:
            raise ImportError(
                "ChromaDB is not installed. Please install it using "
                "`pip install -e .[chromadb]` or `pip install chromadb`."
            )

    def embed_text(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(384)
        
        try:
            if hasattr(self._embedder, "__call__"):
                results = self._embedder([text])
            else:
                results = self._embedder.embed_documents([text])
                
            if results and len(results) > 0:
                return np.array(results[0], dtype=np.float32)
        except Exception as e:
            logger.error(f"ChromaVectorStore embedding error: {e}")
        return np.zeros(384)

    def add_vectors(self, ids: List[str], embeddings: List[np.ndarray], metadatas: Optional[List[dict]] = None):
        # Convert embeddings list to standard python floats list for Chroma
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        # Upsert so re-indexing the same belief ids (e.g. after a process
        # restart with a persistent collection) never raises on duplicates.
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas
        )

    def query_top_k(self, query_embedding: np.ndarray, k: int = 100) -> List[Tuple[str, float]]:
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k
            )
            
            output = []
            if results and 'ids' in results and len(results['ids']) > 0:
                ids = results['ids'][0]
                # Chroma returns distances by default. We convert distance to similarity metric.
                # cosine similarity = 1 - cosine distance
                distances = results.get('distances', [[0.0] * len(ids)])[0]
                for belief_id, dist in zip(ids, distances):
                    sim = 1.0 - dist
                    output.append((belief_id, sim))
            return output
        except Exception as e:
            logger.error(f"ChromaVectorStore query error: {e}")
            return []


class PineconeVectorStore(VectorStore):
    """Vector store adapter for Pinecone."""

    def __init__(self, api_key: Optional[str] = None, index_name: Optional[str] = None):
        api_key = api_key or os.environ.get("PINECONE_API_KEY")
        index_name = index_name or os.environ.get("PINECONE_INDEX_NAME")

        if not api_key:
            raise ValueError(
                "Pinecone VectorStore requires a valid PINECONE_API_KEY. "
                "Set it as an environment variable or pass it to the constructor."
            )
        if not index_name:
            raise ValueError(
                "Pinecone VectorStore requires a PINECONE_INDEX_NAME. "
                "Set it as an environment variable or pass it to the constructor."
            )

        try:
            from pinecone import Pinecone
            self._pc = Pinecone(api_key=api_key)
            self._index = self._pc.Index(index_name)
            logger.info(f"PineconeVectorStore successfully initialized for index '{index_name}'.")
        except ImportError:
            raise ImportError(
                "Pinecone client is not installed. Please install it using "
                "`pip install -e .[pinecone]` or `pip install pinecone`."
            )

    def embed_text(self, text: str) -> np.ndarray:
        try:
            response = self._pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[text],
                parameters={"input_type": "query"}
            )
            return np.array(response.data[0].values, dtype=np.float32)
        except Exception as e:
            logger.error(f"PineconeVectorStore embedding error: {e}")
            raise RuntimeError("Pinecone embedding generation failed.")

    def add_vectors(self, ids: List[str], embeddings: List[np.ndarray], metadatas: Optional[List[dict]] = None):
        vectors = []
        for i, (belief_id, emb) in enumerate(zip(ids, embeddings)):
            meta = metadatas[i] if metadatas else {}
            vectors.append((belief_id, emb.tolist(), meta))
            
        # Bulk upsert
        self._index.upsert(vectors=vectors)

    def query_top_k(self, query_embedding: np.ndarray, k: int = 100) -> List[Tuple[str, float]]:
        try:
            response = self._index.query(
                vector=query_embedding.tolist(),
                top_k=k,
                include_metadata=False
            )
            return [(match.id, match.score) for match in response.matches]
        except Exception as e:
            logger.error(f"PineconeVectorStore query error: {e}")
            return []


class DummyVectorStore(VectorStore):
    """A zero-dependency vector store for offline/local sandbox testing."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._vectors: Dict[str, np.ndarray] = {}
        logger.warning(
            "WARNING: DummyVectorStore initialized. This is a fallback/testing store "
            "and does not generate real semantic embeddings. For production, please "
            "configure ChromaVectorStore, PineconeVectorStore, or provide your own subclass."
        )

    def embed_text(self, text: str) -> np.ndarray:
        # Returns deterministic constant vector so it behaves predictably in testing
        return np.ones(self.dimension, dtype=np.float32)

    def add_vectors(self, ids: List[str], embeddings: List[np.ndarray], metadatas: Optional[List[dict]] = None):
        for belief_id, emb in zip(ids, embeddings):
            self._vectors[belief_id] = emb

    def query_top_k(self, query_embedding: np.ndarray, k: int = 100) -> List[Tuple[str, float]]:
        scores = []
        for belief_id, emb in self._vectors.items():
            sim = self.cosine_similarity(query_embedding, emb)
            scores.append((belief_id, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class OllamaEmbeddingFunction:
    """ChromaDB compatible embedding function using local Ollama model API."""

    def __init__(self, host: str = "http://localhost:11434", model_name: str = "nomic-embed-text"):
        self.host = host.rstrip("/")
        self.model_name = model_name

    def name(self) -> str:
        return "ollama"

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        for text in input:
            if not text.strip():
                # Provide a zero fallback (e.g. 768d for nomic)
                embeddings.append([0.0] * 768)
                continue

            url = f"{self.host}/api/embeddings"
            data = json.dumps({
                "model": self.model_name,
                "prompt": text
            }).encode("utf-8")
            
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            
            try:
                with urllib.request.urlopen(req, timeout=15) as response:
                    resp_data = json.loads(response.read().decode("utf-8"))
                    emb = resp_data.get("embedding")
                    if emb:
                        embeddings.append(emb)
                    else:
                        embeddings.append([0.0] * 768)
            except Exception as e:
                logger.error(f"OllamaEmbeddingFunction error: {e}")
                embeddings.append([0.0] * 768)
        return embeddings


def create_vector_store(backend: Optional[str] = None, **kwargs) -> VectorStore:
    """Factory function to build a VectorStore backend.

    Guards and instructs developers to choose at least one backend.
    """
    backend = backend or os.environ.get("MRAG_VECTOR_STORE")

    if not backend:
        raise ValueError(
            "\n[MicroRAG Configuration Error]: No VectorStore backend selected.\n"
            "You must choose a backend when initializing the system. Options:\n"
            "1. 'chromadb' (Requires `pip install -e .[chromadb]`)\n"
            "2. 'pinecone' (Requires `pip install -e .[pinecone]`)\n"
            "3. 'dummy'    (Zero-dependency local sandbox testing)\n\n"
            "Usage:\n"
            "  from mrag.core.vector_store import create_vector_store\n"
            "  vector_store = create_vector_store('chromadb')\n"
            "Or set the MRAG_VECTOR_STORE environment variable."
        )

    backend = backend.lower().strip()
    if backend == "chromadb":
        return ChromaVectorStore(**kwargs)
    elif backend == "pinecone":
        return PineconeVectorStore(**kwargs)
    elif backend == "dummy":
        return DummyVectorStore(**kwargs)
    else:
        raise ValueError(
            f"Unsupported VectorStore backend '{backend}'. Choose from: "
            "'chromadb', 'pinecone', 'dummy'."
        )
