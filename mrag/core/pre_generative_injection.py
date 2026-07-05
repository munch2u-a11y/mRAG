import logging
import re
from typing import List, Dict, Any, Tuple
from mrag._compat import np

from mrag.memory.belief_store import BeliefStore
from mrag.core.vector_store import VectorStore

logger = logging.getLogger("mrag.core.pre_generative_injection")


class PreGenerativeInjector:
    """Pre-generative text filtering and belief injection pipeline for Micro-RAG.

    Filters incoming text, retrieves structurally and semantically relevant 
    beliefs from the BeliefStore, and weaves them into a formatted context block.
    Maintains a rolling blacklist to prevent repetitive belief injection.
    """

    def __init__(
        self,
        belief_store: BeliefStore,
        vector_store: VectorStore,
        blacklist_memory_size: int = 15,
        enable_graph_expansion: bool = True,
        top_k_candidates: int = 15
    ):
        self._belief_store = belief_store
        self._vector_store = vector_store
        self._blacklist_memory_size = blacklist_memory_size
        self.enable_graph_expansion = enable_graph_expansion
        self.top_k_candidates = top_k_candidates
        
        # Blacklist stores recent belief IDs to avoid repetitive injection
        self._recent_injections: List[str] = []
        self._indexed_belief_ids: set[str] = set()
        self._last_synced_version: Optional[int] = None

    def inject(self, trigger_text: str, current_context: str = "") -> str:
        """Process incoming trigger text and return formatted injected context."""
        
        # Clean text
        text_for_query = f"{current_context} {trigger_text}".strip()
        if not text_for_query:
            return ""

        # 1. Pull relevant beliefs based on text
        beliefs = self._pull_relevant_beliefs(text_for_query, limit=5)
        
        # 2. Build injection string
        if not beliefs:
            return ""

        lines = ["--- Injected Context ---"]
        for b in beliefs:
            conf = b.get("confidence", 0.5)
            content = b.get("content", "")
            lines.append(f"• {content} [{conf:.2f}]")
            
            # Add to blacklist
            bid = b.get("id")
            if bid and bid not in self._recent_injections:
                self._recent_injections.append(bid)
                
        # Trim blacklist
        if len(self._recent_injections) > self._blacklist_memory_size:
            self._recent_injections = self._recent_injections[-self._blacklist_memory_size:]

        lines.append("------------------------")
        
        return "\n".join(lines)

    def sync_index(self):
        """Sync uncached beliefs into the VectorStore once."""
        if not self._belief_store._cache_loaded:
            self._belief_store.load_into_cache()

        # Fast path: nothing changed in the store since the last sync, so the
        # per-belief scan (O(N) per inject call otherwise) can be skipped.
        store_version = getattr(self._belief_store, "version", None)
        if store_version is not None and store_version == self._last_synced_version:
            return

        all_beliefs = list(self._belief_store._beliefs_cache.values())
        to_add_ids = []
        to_add_embs = []
        to_add_metas = []
        
        updated_categories = set()
        for belief in all_beliefs:
            bid = belief.get("id")
            if not bid:
                continue

            cached_embedding = belief.get("embedding") or belief.get("embedding_384d")
            if bid in self._indexed_belief_ids and cached_embedding:
                continue

            # If we don't have the embedding cached, generate it now
            if not cached_embedding:
                emb = self._vector_store.embed_text(belief.get("content", ""))
                cached_embedding = emb.tolist()
                belief["embedding"] = cached_embedding

                # Update memory cache directly
                self._belief_store._beliefs_cache[bid] = belief
                updated_categories.add(belief.get("_category", "premises"))

            to_add_ids.append(bid)
            to_add_embs.append(np.array(cached_embedding, dtype=np.float32))
            to_add_metas.append({"category": belief.get("_category", "premises")})
            
        # Write to disk once per updated category (batch update)
        for category in updated_categories:
            category_beliefs = [
                b for b in self._belief_store._beliefs_cache.values()
                if b.get("_category") == category
            ]
            self._belief_store._write_category(category, category_beliefs)
            
        if to_add_ids:
            self._vector_store.add_vectors(to_add_ids, to_add_embs, to_add_metas)
            self._indexed_belief_ids.update(to_add_ids)

        self._last_synced_version = store_version

    def _pull_relevant_beliefs(self, text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Finds beliefs relevant to the text using Top-K vector query + local structural reranking."""
        self.sync_index()
        
        # 1. Embed query
        query_embedding = self._vector_store.embed_text(text)
        
        # 2. Query vector database for Top-K candidates
        top_k_results = self._vector_store.query_top_k(query_embedding, k=self.top_k_candidates)
        
        # 3. Apply local structural reranking on candidates
        scored_beliefs = []
        blacklisted_beliefs = []
        for belief_id, sim in top_k_results:
            belief = self._belief_store.get_belief(belief_id)
            if not belief:
                continue

            # Compute structural relevance
            relevance = belief.get("relevance", self._belief_store.compute_relevance(belief))
            score = sim * relevance

            if score > 0.05: # Threshold
                # Recently injected beliefs are held back to avoid repetition,
                # but kept as a fallback so a repeated query never comes up empty.
                if belief_id in self._recent_injections:
                    blacklisted_beliefs.append((score, belief))
                else:
                    scored_beliefs.append((score, belief))

        # Sort initial semantic matches
        scored_beliefs.sort(key=lambda x: x[0], reverse=True)
        
        # 4. Graph relation expansion (1-hop)
        if self.enable_graph_expansion:
            expanded_candidates = {}
            for score, belief in scored_beliefs[:15]: # Expand on top 15 matches
                expanded_candidates[belief["id"]] = (score, belief)
                
                # Fetch related beliefs
                related = self._belief_store.get_related(belief["id"])
                for rel_belief in related:
                    rel_id = rel_belief.get("id")
                    if rel_id in self._recent_injections:
                        continue
                        
                    # Attenuate relation score (it inherits 50% score from parent match)
                    rel_score = score * 0.5
                    
                    if rel_id not in expanded_candidates or expanded_candidates[rel_id][0] < rel_score:
                        expanded_candidates[rel_id] = (rel_score, rel_belief)
            
            final_candidates = list(expanded_candidates.values())
        else:
            final_candidates = scored_beliefs

        # Fallback: everything relevant was recently injected — surface the
        # blacklisted matches rather than pretending no memory exists.
        if not final_candidates and blacklisted_beliefs:
            final_candidates = blacklisted_beliefs

        final_candidates.sort(key=lambda x: x[0], reverse=True)
        return [b for score, b in final_candidates[:limit]]

    def clear_blacklist(self):
        """Reset the rolling blacklist."""
        self._recent_injections.clear()
