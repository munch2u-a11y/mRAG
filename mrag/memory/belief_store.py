import json
import os
import math
import re
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger("mrag.memory.belief_store")

def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

# A named-entity relation (see learn_relation_expansion) must be an actual
# proper name — 1-3 capitalized words, letters only. Guards against the LLM
# tagging a vague description ("2 younger kids", "youngest child") or a
# possessive phrase ("Melanie's kids") as if it were a specific person's
# name: those aren't useful search-vocabulary links and, worse, act as
# generic extra heads that dilute retrieval for everything on that broad
# topic rather than pointing at one specific belief.
_PROPER_NOUN_ENTITY_RE = re.compile(r"^[A-Z][a-zA-Z]*(?:\s[A-Z][a-zA-Z]*){0,2}$")

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

# Common sentence-position capitalizations that carry no factual payload.
_SALIENT_STOPWORDS = {
    "i", "the", "a", "an", "user", "model", "assistant", "it", "he", "she",
    "they", "we", "you", "my", "his", "her", "their", "our", "this", "that",
    "these", "those", "if", "when", "then", "there", "remember", "also",
}

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][\w.\-/]*")

def _salient_tokens(text: str) -> set:
    """Extract the value-bearing tokens of a belief statement.

    Salient = tokens containing digits (versions, ports, IPs, ids) plus
    capitalized words that aren't common sentence-position words (names,
    products, places). Used to tell contradictions apart from paraphrases:
    two statements that embed as near-duplicates but swap salient tokens
    ("...prefers Python" vs "...prefers Rust") are conflicting facts, not
    restatements.
    """
    salient = set()
    for token in _TOKEN_PATTERN.findall(text):
        lowered = token.lower().rstrip(".-/")
        if any(ch.isdigit() for ch in token):
            salient.add(lowered)
        elif token[0].isupper() and lowered not in _SALIENT_STOPWORDS:
            salient.add(lowered)
    return salient

def _statement_template(text: str):
    """Reduce a statement to its phrasing skeleton plus its salient tokens.

    "Adam says he prefers to use Python." -> ("<v> says he prefers to use <v>",
    {"adam", "python"}). Two beliefs with the same template that share an
    anchor token but swap the remaining values are the same fact slot holding
    conflicting values — detectable regardless of embedding distance (opposite
    statements like "prefers Python"/"prefers Rust" can embed far apart).
    """
    salient = _salient_tokens(text)
    parts = []
    for token in _TOKEN_PATTERN.findall(text):
        lowered = token.lower().rstrip(".-/")
        parts.append("<v>" if lowered in salient else lowered)
    return " ".join(parts), salient

_TIMESTAMP_PREFIX_RE = re.compile(r'^\[([\d\-: ]+)\]\s*')
def _strip_timestamp_prefix_local(text: str) -> str:
    return _TIMESTAMP_PREFIX_RE.sub("", text)

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
        # Phrasing-skeleton index (template -> belief ids) used to detect
        # same-fact-slot contradictions that embeddings place far apart.
        # Only authoritative while the cache is loaded.
        self._template_index: Dict[str, set] = {}
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

        # Extract conceptual tags automatically if not already present
        if "conceptual_tags" not in belief:
            from mrag.core.tagger import extract_tags
            content = belief.get("content", "")
            clean_content = _strip_timestamp_prefix_local(content)
            belief["conceptual_tags"] = extract_tags(clean_content)

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
        self._template_index.clear()
        for bid, belief in self._beliefs_cache.items():
            for rel_id in belief.get("relations", []):
                self._inbound_relations.setdefault(rel_id, set()).add(bid)
            self._index_template(bid, belief.get("content", ""))

    def _index_template(self, belief_id: str, content: str):
        template, salient = _statement_template(content)
        if salient:
            self._template_index.setdefault(template, set()).add(belief_id)

    def _unindex_template(self, belief_id: str, content: str):
        template, _ = _statement_template(content)
        ids = self._template_index.get(template)
        if ids:
            ids.discard(belief_id)
            if not ids:
                self._template_index.pop(template, None)

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
                self._unindex_template(bid, old.get("content", ""))
            self._index_relations(bid, normalized.get("relations", []))
            self._index_template(bid, normalized.get("content", ""))
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
            self._index_template(belief_id, belief.get("content", ""))
        self.version += 1
        return True

    def learn_concept_expansion(self, category: str, instance: str) -> bool:
        """Persists a learned category -> concrete-instance mapping (e.g.
        "martial arts" -> "taekwondo") into the 'concepts' belief category.

        This lets retrieval expand an abstract query term to the concrete
        vocabulary actually seen in the data, without requiring every
        possible category to be hand-curated upfront: BeliefConsolidator
        calls this when session extraction tags a fact with a category, and
        PreGenerativeInjector merges these into its concept-expansion map at
        sync time. Returns False if the pair was already known (no-op).
        """
        category = category.strip().lower()
        instance = instance.strip().lower()
        if not category or not instance:
            return False

        slug = re.sub(r'[^a-z0-9]+', '_', category).strip('_')
        if not slug:
            return False
        concept_id = f"concept_{slug}"

        if not self._cache_loaded:
            self.load_into_cache()

        existing = self._beliefs_cache.get(concept_id)
        if existing:
            instances = existing.get("instances", [])
            if instance in instances:
                return False
            existing["instances"] = instances + [instance]
            existing["last_accessed"] = _now_iso()
            self._beliefs_cache[concept_id] = existing
            beliefs_list = [
                b for b in self._beliefs_cache.values()
                if b.get("_category") == "concepts"
            ]
            self._write_category("concepts", beliefs_list)
            self.version += 1
            return True

        return self.add_belief(
            category="concepts",
            belief_id=concept_id,
            content=category,
            confidence=1.0,
            source="concept_learning",
            instances=[instance],
        )

    def get_all_concept_expansions(self) -> Dict[str, List[str]]:
        """Returns all learned category -> instances mappings for the
        injector to merge into its concept-expansion map."""
        expansions: Dict[str, List[str]] = {}
        for belief in self.get_all_beliefs_flat():
            if belief.get("_category") == "concepts" and not belief.get("rel_subject"):
                instances = belief.get("instances", [])
                if instances:
                    expansions[belief.get("content", "")] = list(instances)
        return expansions

    def learn_relation_expansion(self, subject: str, relation: str, entities: List[str]) -> bool:
        """Persists a learned (subject, relation) -> named-entity mapping
        (e.g. (John, friends) -> [Bob, Cindy]) into the 'concepts' belief
        category.

        Mirrors learn_concept_expansion but keyed on a specific person's
        relation rather than an abstract category — named entities are
        conversation-specific facts that can't be hand-curated as defaults.
        PreGenerativeInjector uses this to add a relation's named entities as
        extra search heads whenever a query mentions both the subject and
        the relation word (e.g. "John" + "friends"), so "who are John's
        friends" reaches "Bob"/"Cindy" even though the query never says
        their names.
        """
        subject_key = subject.strip().lower()
        relation_key = relation.strip().lower()
        clean_entities = [
            e.strip() for e in entities
            if e and e.strip()
            and _PROPER_NOUN_ENTITY_RE.match(e.strip())
            and e.strip().lower() != subject_key
        ]
        if not subject_key or not relation_key or not clean_entities:
            return False

        slug = re.sub(r'[^a-z0-9]+', '_', f"{subject_key}_{relation_key}").strip('_')
        if not slug:
            return False
        concept_id = f"relation_{slug}"

        if not self._cache_loaded:
            self.load_into_cache()

        existing = self._beliefs_cache.get(concept_id)
        if existing:
            instances = existing.get("instances", [])
            existing_lower = {i.lower() for i in instances}
            new_instances = [e for e in clean_entities if e.lower() not in existing_lower]
            if not new_instances:
                return False
            existing["instances"] = instances + new_instances
            existing["last_accessed"] = _now_iso()
            self._beliefs_cache[concept_id] = existing
            beliefs_list = [
                b for b in self._beliefs_cache.values()
                if b.get("_category") == "concepts"
            ]
            self._write_category("concepts", beliefs_list)
            self.version += 1
            return True

        return self.add_belief(
            category="concepts",
            belief_id=concept_id,
            content=f"relation:{subject_key}:{relation_key}",
            confidence=1.0,
            source="relation_learning",
            rel_subject=subject_key,
            rel_type=relation_key,
            instances=clean_entities,
        )

    def get_all_relation_expansions(self) -> Dict[str, Dict[str, List[str]]]:
        """Returns all learned (subject, relation) -> entity-name mappings,
        keyed by lowercased subject then lowercased relation type, for the
        injector to merge into its relation-expansion map."""
        expansions: Dict[str, Dict[str, List[str]]] = {}
        for belief in self.get_all_beliefs_flat():
            if belief.get("_category") != "concepts":
                continue
            rel_subject = belief.get("rel_subject")
            rel_type = belief.get("rel_type")
            instances = belief.get("instances", [])
            if rel_subject and rel_type and instances:
                expansions.setdefault(rel_subject, {})[rel_type] = list(instances)
        return expansions

    def record_structural_cluster(self, belief_ids: List[str], rollup_belief_id: str) -> bool:
        """Records a geometrically-discovered cluster (see
        BeliefConsolidator.discover_and_consolidate_clusters) that has been
        rolled up, so a later discovery pass can recognize the same cluster
        and skip re-consolidating it. Unlike tag-based clusters (keyed by an
        explicit subject/category the LLM assigned), these are found by
        embedding-similarity geometry alone, so they're identified by member
        overlap rather than a stable key.
        """
        if not belief_ids:
            return False
        if not self._cache_loaded:
            self.load_into_cache()
        record_id = f"structcluster_{rollup_belief_id}"
        return self.add_belief(
            category="concepts",
            belief_id=record_id,
            content=f"structural_cluster_for:{rollup_belief_id}",
            confidence=1.0,
            source="structural_cluster_discovery",
            member_ids=list(belief_ids),
        )

    def get_structural_clusters(self) -> List[List[str]]:
        """Returns member-ID lists for every previously-recorded structural
        cluster, so a new discovery pass can skip clusters that substantially
        overlap with one already rolled up."""
        clusters = []
        for belief in self.get_all_beliefs_flat():
            if belief.get("_category") == "concepts" and "member_ids" in belief:
                clusters.append(list(belief["member_ids"]))
        return clusters

    @staticmethod
    def _cluster_id(subject: str, category: str) -> str:
        subject_slug = re.sub(r'[^a-z0-9]+', '_', subject.strip().lower()).strip('_')
        category_slug = re.sub(r'[^a-z0-9]+', '_', category.strip().lower()).strip('_')
        return f"cluster_{subject_slug}_{category_slug}"

    def tag_cluster_membership(self, subject: str, category: str, belief_id: str) -> int:
        """Records that belief_id belongs to the (subject, category) cluster
        (e.g. Maria's "volunteering" facts). BeliefConsolidator checks the
        returned count against CLUSTER_SIZE_THRESHOLD (and the cluster's
        total token footprint against CLUSTER_TOKEN_FRACTION_THRESHOLD) to
        decide whether the cluster has grown large enough to be diluting
        retrieval and needs a rollup consolidation. Returns the cluster's
        current member count.
        """
        if not subject or not category or not belief_id:
            return 0
        if not self._cache_loaded:
            self.load_into_cache()

        cid = self._cluster_id(subject, category)
        existing = self._beliefs_cache.get(cid)
        if existing:
            members = existing.get("members", [])
            if belief_id not in members:
                members = members + [belief_id]
                existing["members"] = members
                existing["last_accessed"] = _now_iso()
                self._beliefs_cache[cid] = existing
                beliefs_list = [
                    b for b in self._beliefs_cache.values()
                    if b.get("_category") == "concepts"
                ]
                self._write_category("concepts", beliefs_list)
                self.version += 1
            return len(existing.get("members", []))

        self.add_belief(
            category="concepts",
            belief_id=cid,
            content=f"cluster:{subject.strip().lower()}:{category.strip().lower()}",
            confidence=1.0,
            source="cluster_tracking",
            members=[belief_id],
            consolidated=False,
        )
        return 1

    def get_cluster_members(self, subject: str, category: str) -> List[str]:
        belief = self.get_belief(self._cluster_id(subject, category))
        return list(belief.get("members", [])) if belief else []

    def is_cluster_consolidated(self, subject: str, category: str) -> bool:
        belief = self.get_belief(self._cluster_id(subject, category))
        return bool(belief.get("consolidated")) if belief else False

    def mark_cluster_consolidated(self, subject: str, category: str):
        cid = self._cluster_id(subject, category)
        if not self._cache_loaded:
            self.load_into_cache()
        existing = self._beliefs_cache.get(cid)
        if not existing:
            return
        existing["consolidated"] = True
        existing["last_accessed"] = _now_iso()
        self._beliefs_cache[cid] = existing
        beliefs_list = [
            b for b in self._beliefs_cache.values()
            if b.get("_category") == "concepts"
        ]
        self._write_category("concepts", beliefs_list)
        self.version += 1

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
        similarity_threshold: float = 0.80,
        **extra_fields,
    ) -> bool:
        """Adds a belief to the store, merging instead when it restates or
        supersedes an existing one.

        Duplicate detection runs through three channels, in order:
        1. Exact belief id match.
        2. Template match: same phrasing skeleton, sharing an anchor token but
           swapping value tokens ("Adam prefers Python" vs "Adam prefers
           Rust"). Catches contradictions that embeddings place far apart.
        3. Semantic match: same-category vector hit above
           ``similarity_threshold``.

        Merges treat paraphrases as corroboration (confidence boost) and
        contradictions as supersession (newer content wins, evidence resets).
        """
        if not self._cache_loaded:
            self.load_into_cache()

        canonical = _canonical_category(category)
        target_belief_id = belief_id
        is_duplicate = False
        match_similarity = 1.0

        if belief_id in self._beliefs_cache:
            target_belief_id = belief_id
            is_duplicate = True

        if not is_duplicate:
            # Template channel: O(1) lookup for same-fact-slot statements.
            new_template, new_salient = _statement_template(content)
            if new_salient:
                best_overlap = 0
                for cid in self._template_index.get(new_template, ()):
                    cand = self._beliefs_cache.get(cid)
                    if not cand or cand.get("_category") != canonical:
                        continue
                    cand_salient = _salient_tokens(cand.get("content", ""))
                    shared = cand_salient & new_salient
                    # Same skeleton + shared anchor + values swapped on both
                    # sides = conflicting values for the same fact slot.
                    if shared and (cand_salient - new_salient) and (new_salient - cand_salient):
                        if len(shared) > best_overlap:
                            best_overlap = len(shared)
                            target_belief_id = cid
                            is_duplicate = True
                if is_duplicate:
                    logger.info(
                        f"Detected same-fact-slot statement via template match. Merging '{content}' "
                        f"into belief '{self._beliefs_cache[target_belief_id]['content']}'."
                    )

        if not is_duplicate and vector_store is not None:
            try:
                query_emb = vector_store.embed_text(content)
                results = vector_store.query_top_k(query_emb, k=5)
                for matched_id, similarity in results or []:
                    if similarity < similarity_threshold:
                        break
                    matched = self._beliefs_cache.get(matched_id)
                    # Only merge semantically within the same category: a global
                    # match could otherwise absorb e.g. a preference into a
                    # skills/tool belief and overwrite its content.
                    if matched is not None and matched.get("_category") == canonical:
                        target_belief_id = matched_id
                        is_duplicate = True
                        match_similarity = similarity
                        logger.info(f"Detected semantic duplicate (similarity {similarity:.4f}). Merging '{content}' into matched belief '{matched['content']}'.")
                        break
            except Exception as e:
                logger.error(f"Semantic deduplication query failed: {e}")

        if is_duplicate:
            existing = self._beliefs_cache[target_belief_id]
            old_content = existing.get("content", "")
            is_contradiction = False

            if old_content != content:
                old_salient = _salient_tokens(old_content)
                new_salient = _salient_tokens(content)
                # Both sides carrying tokens the other lacks means the statements
                # swap values (Python vs Rust, one IP vs another): a conflicting
                # fact, not a restatement.
                is_contradiction = bool(old_salient - new_salient) and bool(new_salient - old_salient)

                # Adopt the newer wording when it supersedes (contradiction),
                # adds salient information (strict superset), or is a
                # high-confidence paraphrase. Mid-band paraphrases (0.80-0.90)
                # and vaguer restatements keep the existing, more established
                # wording — merging still counts as corroboration below.
                adopt_newer = (
                    is_contradiction
                    or new_salient > old_salient
                    or (new_salient == old_salient and match_similarity >= 0.90)
                )
                if adopt_newer:
                    logger.info(
                        f"Updating content of belief {target_belief_id} from '{old_content}' to newer "
                        f"'{content}' ({'contradiction supersedes' if is_contradiction else 'refinement'})."
                    )
                    self._unindex_template(target_belief_id, old_content)
                    existing["content"] = content
                    self._index_template(target_belief_id, content)
                    # Invalidate the cached embedding so retrieval reindexes
                    # the new wording.
                    existing.pop("embedding", None)
                    existing.pop("embedding_384d", None)

            if is_contradiction:
                # A reversal is not corroborating evidence: restart the evidence
                # trail at the newer statement's own confidence and destabilize.
                existing["previous_content"] = old_content
                existing["confidence"] = max(0.0, min(1.0, confidence))
                existing["verifications"] = 1.0
                existing["stability_index"] = max(0.0, min(1.0, existing.get("stability_index", 0.5) * 0.7))
            else:
                old_conf = existing.get("confidence", 0.5)
                existing["confidence"] = max(0.0, min(1.0, old_conf + (1.0 - old_conf) * 0.2))
                existing["verifications"] = existing.get("verifications", 1.0) + 1.0
                existing["stability_index"] = max(0.0, min(1.0, existing.get("stability_index", 0.5) + (1.0 - existing.get("stability_index", 0.5)) * 0.1))

            new_rels = relations or []
            existing["relations"] = list(dict.fromkeys(existing.get("relations", []) + new_rels))

            new_refs = memory_refs or []
            existing["memory_refs"] = list(dict.fromkeys(existing.get("memory_refs", []) + new_refs))

            existing["last_accessed"] = _now_iso()
            merged_source = f"{existing.get('source', '')}; {source}"
            # Keep the provenance chain bounded across many merges.
            existing["source"] = merged_source[-300:]

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
            self._unindex_template(belief_id, removed.get("content", ""))
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
                        self._unindex_template(pruned.get("id", ""), pruned.get("content", ""))
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
