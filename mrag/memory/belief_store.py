import json
import os
import math
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger("mrag.memory.belief_store")

def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

BELIEF_CATEGORIES = {
    "premises": "premises.json",            # Foundational truths, axioms, self-observations
    "propositions": "propositions.json",    # Learned/derived facts, conditional rules
    "preferences": "preferences.json",      # Values, likes, behavioral norms
    "people": "people.json",                # Entity profiles
    "skills": "skills.json",                # Proven tool-backed workflows
    "desires": "desires.json",              # Long-term goals and aspirations
    "concepts": "concepts.json",            # Consolidated conceptual understanding
}

_SINGULAR_MAP = {
    "premise": "premises", "proposition": "propositions",
    "preference": "preferences", "person": "people",
    "skill": "skills", "desire": "desires", "concept": "concepts",
}

def _canonical_category(category: str) -> str:
    return _SINGULAR_MAP.get(category, category)

class BeliefStore:
    """Categorized belief management for Micro-RAG.
    Maintains structurally connected beliefs across categories.
    """
    
    PRUNING_THRESHOLD = 0.20
    _DECAY_BASE = 0.20
    # Per-pass interpolation rate toward the structural equilibrium score.
    # Keeps decay gradual so a fresh high-confidence belief is not pruned overnight.
    _DECAY_RATE = 0.30

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._ensure_files()
        self._beliefs_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_loaded = False
        # Reverse relation index (target_id -> ids that point at it).
        # Only authoritative while the cache is loaded.
        self._inbound_relations: Dict[str, set] = {}
        # Monotonic mutation counter so consumers (e.g. the injector's index
        # sync) can skip work when nothing has changed.
        self.version = 0

    def _normalize_belief(
        self,
        belief: Dict[str, Any],
        *,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """Normalize belief fields into a canonical runtime shape."""
        belief["confidence"] = max(0.0, min(1.0, float(belief.get("confidence", 0.5))))
        belief["stability_index"] = max(0.0, min(1.0, float(belief.get("stability_index", 0.5))))
        belief["verifications"] = float(belief.get("verifications", 1.0))
        belief["access_count"] = int(belief.get("access_count", 0))
        belief["relations"] = list(dict.fromkeys(belief.get("relations", []) or []))
        belief["memory_refs"] = list(dict.fromkeys(belief.get("memory_refs", []) or []))
        belief["tags"] = list(dict.fromkeys(belief.get("tags", []) or []))
        belief["weight"] = self._resolve_weight(belief["confidence"])

        # Relevance replaces affect-based mass, computed purely structurally.
        belief["relevance"] = round(self.compute_relevance(belief), 4)

        if category:
            belief["_category"] = category

        return belief

    def _ensure_files(self):
        """Create empty category files if they don't exist."""
        for category, filename in BELIEF_CATEGORIES.items():
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                with open(filepath, 'w') as f:
                    json.dump([], f, indent=2)

    def _read_category(self, category: str) -> List[Dict[str, Any]]:
        """Read all beliefs from a category file."""
        category = _canonical_category(category)
        filename = BELIEF_CATEGORIES.get(category)
        if not filename:
            logger.warning(f"Unknown belief category: {category}")
            return []
        filepath = os.path.join(self.data_dir, filename)
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {filepath}: {e}")
            return []

    def _write_category(self, category: str, beliefs: List[Dict[str, Any]]):
        """Write all beliefs to a category file."""
        category = _canonical_category(category)
        filename = BELIEF_CATEGORIES.get(category)
        if not filename:
            return
        filepath = os.path.join(self.data_dir, filename)
        serialized = []
        for belief in beliefs:
            clean = dict(belief)
            clean.pop("_category", None)
            serialized.append(clean)
        try:
            with open(filepath, 'w') as f:
                json.dump(serialized, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write {filepath}: {e}")

    def get_all_beliefs_flat(self) -> List[Dict[str, Any]]:
        """Get ALL beliefs across all categories as one flat list."""
        if self._cache_loaded:
            return list(self._beliefs_cache.values())

        all_beliefs = []
        for category in BELIEF_CATEGORIES:
            beliefs = [
                self._normalize_belief(dict(b), category=category)
                for b in self._read_category(category)
            ]
            all_beliefs.extend(beliefs)
        return all_beliefs

    def load_into_cache(self):
        """Loads all beliefs from disk into the in-memory cache for O(1) lookups."""
        self._beliefs_cache.clear()
        for category in BELIEF_CATEGORIES:
            beliefs = self._read_category(category)
            for b in beliefs:
                normalized = self._normalize_belief(dict(b), category=category)
                self._beliefs_cache[normalized["id"]] = normalized
        self._cache_loaded = True
        self._rebuild_inbound_index()
        self.version += 1

    def _rebuild_inbound_index(self):
        self._inbound_relations.clear()
        for bid, belief in self._beliefs_cache.items():
            for rel_id in belief.get("relations", []):
                self._inbound_relations.setdefault(rel_id, set()).add(bid)

    def _index_relations(self, belief_id: str, relations: List[str]):
        for rel_id in relations:
            self._inbound_relations.setdefault(rel_id, set()).add(belief_id)

    def _unindex_relations(self, belief_id: str, relations: List[str]):
        for rel_id in relations:
            sources = self._inbound_relations.get(rel_id)
            if sources:
                sources.discard(belief_id)
                if not sources:
                    self._inbound_relations.pop(rel_id, None)

    def update_belief(self, category: str, belief: Dict[str, Any]) -> bool:
        """Updates a belief's contents on disk and in cache."""
        bid = belief.get("id")
        if not bid:
            return False

        normalized = self._normalize_belief(dict(belief), category=category)
        if self._cache_loaded:
            old = self._beliefs_cache.get(bid)
            if old:
                self._unindex_relations(bid, old.get("relations", []))
            self._index_relations(bid, normalized.get("relations", []))
        self._beliefs_cache[bid] = normalized
        self.version += 1

        beliefs = self._read_category(category)
        updated = False
        for i, b in enumerate(beliefs):
            if b.get("id") == bid:
                beliefs[i] = normalized
                updated = True
                break
        if not updated:
            beliefs.append(normalized)
            
        self._write_category(category, beliefs)
        return True

    def add_belief(
        self,
        category: str,
        belief_id: str,
        content: str,
        confidence: float = 0.5,
        source: str = "system",
        verifications: float = 1.0,
        stability_index: float = 0.5,
        relations: list = None,
        memory_refs: list = None,
        **extra_fields,
    ) -> bool:
        # Read from cache if loaded, otherwise from file
        if self._cache_loaded:
            if belief_id in self._beliefs_cache:
                return False
        else:
            beliefs = self._read_category(category)
            for b in beliefs:
                if b.get("id") == belief_id:
                    return False

        now = _now_iso()
        belief = {
            "id": belief_id,
            "content": content,
            "confidence": max(0.0, min(1.0, confidence)),
            "source": source,
            "created_at": now,
            "last_accessed": now,
            "access_count": 0,
            "verifications": float(verifications),
            "stability_index": float(stability_index),
            "relations": relations or [],
            "memory_refs": memory_refs or [],
        }

        for key, val in extra_fields.items():
            if key not in belief:
                belief[key] = val

        belief = self._normalize_belief(belief, category=category)
        
        # Write to disk
        beliefs = self._read_category(category)
        beliefs.append(belief)
        self._write_category(category, beliefs)
        
        # Update cache
        self._beliefs_cache[belief_id] = belief
        if self._cache_loaded:
            self._index_relations(belief_id, belief.get("relations", []))
        self.version += 1
        return True

    def merge_or_add_belief(
        self,
        category: str,
        belief_id: str,
        content: str,
        confidence: float = 0.5,
        source: str = "system",
        relations: list = None,
        memory_refs: list = None,
        vector_store: Optional[Any] = None,
        **extra_fields,
    ) -> bool:
        """Adds a belief to the store. If an exact duplicate or semantic equivalent is found,
        it merges their metadata (updating confidence, verifications, and relations) instead
        of creating a duplicate entry.
        """
        if not self._cache_loaded:
            self.load_into_cache()

        target_belief_id = belief_id
        is_duplicate = False

        if belief_id in self._beliefs_cache:
            target_belief_id = belief_id
            is_duplicate = True

        if not is_duplicate and vector_store is not None:
            try:
                query_emb = vector_store.embed_text(content)
                results = vector_store.query_top_k(query_emb, k=1)
                if results and len(results) > 0:
                    matched_id, similarity = results[0]
                    if similarity >= 0.90 and matched_id in self._beliefs_cache:
                        target_belief_id = matched_id
                        is_duplicate = True
                        logger.info(f"Detected semantic duplicate (similarity {similarity:.4f}). Merging '{content}' into matched belief '{self._beliefs_cache[matched_id]['content']}'.")
            except Exception as e:
                logger.error(f"Semantic deduplication query failed: {e}")

        if is_duplicate:
            existing = self._beliefs_cache[target_belief_id]
            old_conf = existing.get("confidence", 0.5)
            existing["confidence"] = max(0.0, min(1.0, old_conf + (1.0 - old_conf) * 0.2))
            
            existing["verifications"] = existing.get("verifications", 1.0) + 1.0
            existing["stability_index"] = max(0.0, min(1.0, existing.get("stability_index", 0.5) + (1.0 - existing.get("stability_index", 0.5)) * 0.1))
            
            new_rels = relations or []
            existing["relations"] = list(dict.fromkeys(existing.get("relations", []) + new_rels))
            
            new_refs = memory_refs or []
            existing["memory_refs"] = list(dict.fromkeys(existing.get("memory_refs", []) + new_refs))
            
            existing["last_accessed"] = _now_iso()
            existing["source"] = f"{existing.get('source', '')}; {source}"
            
            for key, val in extra_fields.items():
                if key not in existing:
                    existing[key] = val

            category_of_existing = existing.get("_category") or category
            existing = self._normalize_belief(existing, category=category_of_existing)
            self._beliefs_cache[target_belief_id] = existing
            
            beliefs_list = [
                b for b in self._beliefs_cache.values()
                if b.get("_category") == category_of_existing
            ]
            self._write_category(category_of_existing, beliefs_list)
            self._index_relations(target_belief_id, existing.get("relations", []))
            self.version += 1
            return True
        else:
            return self.add_belief(
                category=category,
                belief_id=belief_id,
                content=content,
                confidence=confidence,
                source=source,
                verifications=1.0,
                stability_index=0.5,
                relations=relations,
                memory_refs=memory_refs,
                **extra_fields
            )

    def get_belief(self, belief_id: str) -> Optional[Dict[str, Any]]:
        if self._cache_loaded:
            return self._beliefs_cache.get(belief_id)

        for category in BELIEF_CATEGORIES:
            beliefs = self._read_category(category)
            for b in beliefs:
                if b.get("id") == belief_id:
                    return self._normalize_belief(dict(b), category=category)
        return None

    def remove_belief(self, category: str, belief_id: str) -> bool:
        removed = self._beliefs_cache.pop(belief_id, None)
        if self._cache_loaded and removed:
            self._unindex_relations(belief_id, removed.get("relations", []))
        beliefs = self._read_category(category)
        original_count = len(beliefs)
        beliefs = [b for b in beliefs if b.get("id") != belief_id]
        if len(beliefs) < original_count:
            self._write_category(category, beliefs)
            self.version += 1
            return True
        return False

    def get_related(self, belief_id: str) -> List[Dict[str, Any]]:
        related = []
        belief = self.get_belief(belief_id)
        if not belief:
            return related

        seen = {belief_id}
        for rel_id in belief.get("relations", []):
            rel = self.get_belief(rel_id)
            if rel and rel_id not in seen:
                related.append(rel)
                seen.add(rel_id)

        if self._cache_loaded:
            # O(degree) inbound lookup instead of a full-store scan.
            for src_id in self._inbound_relations.get(belief_id, ()):
                if src_id in seen:
                    continue
                src = self._beliefs_cache.get(src_id)
                if src:
                    related.append(src)
                    seen.add(src_id)
            return related

        all_beliefs = self.get_all_beliefs_flat()
        for b in all_beliefs:
            if b.get("id") not in seen and belief_id in b.get("relations", []):
                related.append(b)
                seen.add(b.get("id"))

        return related

    def compute_relevance(self, belief: dict) -> float:
        """Compute structural relevance for retrieval (replaces mass).
        Derived from confidence, stability, and relation graph connectivity.
        """
        c = belief.get("confidence", 0.5)
        stability = max(0.0, min(1.0, float(belief.get("stability_index", 0.5))))
        
        # Structural mass = confidence * (0.5 + stability)
        base_relevance = c * (0.5 + stability)
        
        # We factor in relation density (how connected is this belief?)
        # Instead of global O(N) inbound scan every time, we approximate with outbound relations
        # and base confidence. The actual "heavy" relation lookup is done during attrition.
        rel_count = len(belief.get("relations", []))
        rel_multiplier = 1.0 + min(0.5, rel_count * 0.05)
        
        return max(0.01, base_relevance * rel_multiplier)

    def decay_all_beliefs(self) -> dict:
        """Run the belief decay equations across all beliefs."""
        now = datetime.now()
        stats = {"pruned": 0, "demoted": 0, "promoted": 0, "updated": 0}

        for category in BELIEF_CATEGORIES:
            beliefs = self._read_category(category)
            if not beliefs:
                continue

            inbound_counts = {}
            for b in beliefs:
                for rel_id in b.get("relations", []):
                    inbound_counts[rel_id] = inbound_counts.get(rel_id, 0) + 1

            surviving = []
            for b in beliefs:
                old_conf = max(0.0, min(1.0, float(b.get("confidence", 0.5))))

                try:
                    created = b.get("created_at", "")
                    if "T" in created:
                        formed_date = datetime.fromisoformat(created.split("T")[0])
                    else:
                        formed_date = datetime.strptime(created, "%Y-%m-%d")
                    days_held = max(0.0, (now - formed_date).days)
                except (ValueError, TypeError):
                    days_held = 30.0

                t_score = 0.40 * min(1.0, math.log2(days_held + 1) / math.log2(31))
                
                r_count = inbound_counts.get(b.get("id", ""), 0)
                r_score = 0.20 * min(1.0, r_count / 5.0)

                v_count = float(b.get("verifications", 1.0))
                v_score = 0.20 * min(1.0, v_count / 10.0)

                s_index = float(b.get("stability_index", 0.5))
                s_modifier = 0.5 + s_index

                base = self._DECAY_BASE
                structural_score = min(1.0, (base + t_score + r_score + v_score) * s_modifier)
                # Decay gradually toward the structural equilibrium instead of
                # replacing confidence outright, so one pass can't crater a
                # freshly formed high-confidence belief.
                new_conf = min(1.0, old_conf + (structural_score - old_conf) * self._DECAY_RATE)

                if v_count > 0.0:
                    b["verifications"] = max(0.0, v_count - 0.05)

                if new_conf < self.PRUNING_THRESHOLD:
                    stats["pruned"] += 1
                    pruned = self._beliefs_cache.pop(b.get("id", ""), None)
                    if self._cache_loaded and pruned:
                        self._unindex_relations(pruned.get("id", ""), pruned.get("relations", []))
                    continue

                old_weight = self._resolve_weight(old_conf)
                new_weight = self._resolve_weight(new_conf)
                if old_weight in ("core", "deep") and new_weight == "surface":
                    stats["demoted"] += 1
                elif old_weight == "surface" and new_weight in ("core", "deep"):
                    stats["promoted"] += 1

                b["confidence"] = round(new_conf, 3)
                self._normalize_belief(b, category=category)
                stats["updated"] += 1
                surviving.append(b)
                # Keep the in-memory cache consistent with what we persist.
                bid = b.get("id")
                if bid and (self._cache_loaded or bid in self._beliefs_cache):
                    self._beliefs_cache[bid] = b

            self._write_category(category, surviving)

        self.version += 1
        return stats

    @staticmethod
    def _resolve_weight(confidence: float) -> str:
        if confidence < 0.60:
            return "surface"
        elif confidence < 0.85:
            return "deep"
        else:
            return "core"
