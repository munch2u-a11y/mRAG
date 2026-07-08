import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

from mrag._compat import np
from mrag.core.token_counting import count_text_tokens
from mrag.memory.belief_store import BeliefStore
from mrag.core.vector_store import VectorStore

logger = logging.getLogger("mrag.core.pre_generative_injection")

# Beliefs are stored with a leading "[2023-05-25 13:14] " session-date prefix
# for the model's own temporal grounding. That prefix is noise for semantic
# matching/deduplication — the timestamp isn't part of the fact itself — so
# it's stripped before embedding and before any similarity comparison.
_TIMESTAMP_PREFIX = re.compile(r'^\[([\d\-: ]+)\]\s*')

def _strip_timestamp_prefix(text: str) -> str:
    return _TIMESTAMP_PREFIX.sub("", text)

def _parse_timestamp_prefix(text: str) -> datetime:
    m = _TIMESTAMP_PREFIX.match(text)
    if not m:
        return datetime.min
    raw = m.group(1).strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return datetime.min

# Near-duplicate purge (results-overlap purge): two candidate beliefs whose
# (timestamp-stripped) embeddings are at least this similar are treated as the
# same fact restated, not two distinct facts, and only one survives into the
# final injection. Calibrated against real LoCoMo belief clusters: genuine
# restatements measured 0.87-0.95 similarity, genuinely distinct facts about
# the same topic measured 0.47-0.66 — 0.90 sits above the one confirmed false
# positive (two different screenplays at 0.873) while still catching true
# restatements.
DUPLICATE_SIMILARITY_THRESHOLD = 0.90
# Survivor selection is primarily "most similar to the raw full query"; when
# two members of a cluster are within this margin of each other, the more
# recent belief (by its session-date prefix) wins instead.
DUPLICATE_TIE_EPSILON = 0.01

STOPWORDS = {
    "about", "above", "after", "again", "against", "along", "around",
    "before", "behind", "below", "beneath", "beside", "between", "beyond",
    "during", "under", "within", "without", "would", "could", "should",
    "their", "there", "these", "those", "other", "another", "through",
    "first", "second", "third", "since", "until", "while", "where",
    "which", "whose", "what", "when", "with", "from", "that", "this",
    "then", "here", "there", "have", "been", "were", "was", "is", "are",
    "did", "does", "do", "has", "had", "being", "will", "shall",
    "you", "your", "and", "for",
}

# Interrogative pronouns are excluded from single-word/noun-phrase heads
# separately from STOPWORDS since they're meaningful for other NLP purposes.
_QUESTION_WORDS = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how"}

# Generic words that pass the single-word head filter (length >= 3, not a
# stopword) but carry almost no discriminating signal on their own — as a
# search head their embedding pulls in whatever's topically nearby rather
# than anything specific to the query, inflating match_count for otherwise
# irrelevant beliefs. Excluded from single-word head generation only; they're
# still valid content elsewhere (bigrams, raw query text).
GENERIC_FILLER_WORDS = {
    "kinds", "kind", "things", "thing", "type", "types", "sort", "sorts",
    "way", "ways", "item", "items", "stuff", "some", "any", "many", "much",
    "both", "done", "have", "had",
}

# Bounds on how many parallel embedding queries a single inject() call fires.
# Keeps worst-case latency bounded as more head-generation strategies stack up.
MAX_SEARCH_HEADS = 16
MAX_CONCEPT_EXPANSION_HEADS = 8
MAX_RELATION_EXPANSION_HEADS = 6

# Lightweight concept -> concrete-vocabulary expansion map (Pattern A fix).
# Abstract question wording ("achievement", "relationship status") sits far
# from concrete memory phrasing ("finished her screenplay", "single parent")
# in embedding space. Each trigger (matched as a case-insensitive substring
# of the query) adds its concrete synonyms as extra search heads so at least
# one head lands near the concrete memory. Extend/override via the
# `concept_expansions` constructor argument for domain-specific vocabulary.
DEFAULT_CONCEPT_EXPANSIONS: Dict[str, List[str]] = {
    "achievement": ["finished", "completed", "accomplished", "won", "succeeded", "milestone", "proud"],
    "accomplishment": ["finished", "completed", "accomplished", "won", "succeeded", "milestone"],
    "relationship status": ["single", "married", "dating", "divorced", "engaged", "widowed", "partner", "boyfriend", "girlfriend", "husband", "wife"],
    "interest": ["hobby", "enjoys", "passionate", "likes", "loves"],
    "hobby": ["enjoys", "passionate about", "likes", "spends time"],
    "occupation": ["works as", "job", "career", "profession", "employed"],
    "career path": ["works as", "job", "profession", "wants to become", "studying"],
    "volunteering": ["volunteers", "shelter", "charity", "donates", "helps out"],
    "health": ["sick", "diagnosed", "illness", "recovering", "therapy", "doctor"],
    "feeling": ["happy", "sad", "excited", "anxious", "proud", "stressed", "grateful"],
    "mood": ["happy", "sad", "excited", "anxious", "stressed"],
    "conflict": ["argument", "disagreement", "fight", "struggle", "tension"],
    "plan": ["going to", "intends to", "upcoming", "planning to"],
    "living situation": ["lives", "moved", "apartment", "house", "roommate"],
    "education": ["studies", "degree", "school", "college", "university", "enrolled"],
}

def _stem_word(word: str) -> str:
    word = word.lower().strip()
    # Basic suffix stemming
    if word.endswith("ing"):
        return word[:-3]
    if word.endswith("ed"):
        return word[:-2]
    # "es" only marks a genuine inserted-vowel plural/conjugation after a
    # sibilant ("boxes"->"box", "watches"->"watch", "buses"->"bus"). Any
    # other "...es" word ("loves", "hikes", "moves", "explores") is a plain
    # silent-e word + "s" and must fall through to the general "s" branch
    # below ("loves"->"love", not "lov") — checking bare "es" before "s"
    # here silently broke exact-word/stem matching for a wide class of
    # common verbs.
    if word.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    # Deliberately no blanket "er"/"est" comparative-adjective stripping:
    # far more common English words simply end in "er" as part of the root
    # (prefer, offer, consider, remember, deliver, cover, water, letter...)
    # than are actual comparatives ("bigger"->"big"), so stripping it
    # unconditionally does more harm than good — and it was inconsistent
    # besides, since a word already handled by the "s" branch above (e.g.
    # "prefers"->"prefer") never reached this branch, while its singular
    # form ("prefer") did, silently stemming the same root two different
    # ways depending on inflection.
    return word

def estimate_tokens(text: str) -> int:
    return count_text_tokens(text)


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
        top_k_candidates: int = 30,
        token_budget_fraction: float = 0.15,
        max_injected_tokens: int = 800,
        concept_expansions: Optional[Dict[str, List[str]]] = None,
    ):
        self._belief_store = belief_store
        self._vector_store = vector_store
        self._blacklist_memory_size = blacklist_memory_size
        self.enable_graph_expansion = enable_graph_expansion
        self.top_k_candidates = top_k_candidates
        self.token_budget_fraction = token_budget_fraction
        self.max_injected_tokens = max_injected_tokens
        # Three layers, increasing priority: built-in defaults, expansions
        # learned automatically from consolidated facts (see
        # BeliefStore.learn_concept_expansion), and explicit user overrides.
        # Defaults and learned entries are additive (unioned per category);
        # an explicit user override for a category replaces it outright.
        self._user_concept_expansions = concept_expansions or {}
        self._learned_concept_expansions: Dict[str, List[str]] = {}
        self._concept_expansions = self._compute_concept_expansions()
        # Named-entity relation expansions (e.g. John -> {"friends": ["Bob",
        # "Cindy"]}) are purely learned, never hand-curated defaults — unlike
        # concept vocabulary, named entities are specific to one conversation
        # and can't be seeded upfront without hardcoding eval-specific data.
        # See BeliefStore.learn_relation_expansion / get_all_relation_expansions.
        self._learned_relation_expansions: Dict[str, Dict[str, List[str]]] = {}

        import os
        from mrag.core.context_compressor import resolve_context_limit
        try:
            model_name = os.environ.get("MRAG_MODEL_NAME")
            self.context_limit = resolve_context_limit(None, model_name)
        except Exception:
            self.context_limit = 8192
            
        # Blacklist stores recent belief IDs to avoid repetitive injection
        self._recent_injections: List[str] = []
        self._indexed_belief_ids: set[str] = set()
        self._last_synced_version: Optional[int] = None

    def inject(self, trigger_text: str, current_context: str = "", limit: Optional[int] = None,
               current_time: Optional[Any] = None) -> str:
        """Process incoming trigger text and return formatted injected context.

        current_time is the reference "now" stamped at the top of the context so
        the model can resolve relative-date questions ("how many days ago?")
        against the "[YYYY-MM-DD HH:MM]" prefix each belief carries. It accepts a
        datetime or a pre-formatted string and defaults to the real wall-clock
        time. Callers replaying historical or simulated conversations (e.g.
        benchmarks) must pass the simulated present rather than the machine clock,
        otherwise relative-date arithmetic anchors to the wrong year.
        """

        # Clean text
        text_for_query = f"{current_context} {trigger_text}".strip()
        if not text_for_query:
            return ""

        # 1. Pull relevant beliefs based on text
        beliefs = self._pull_relevant_beliefs(text_for_query, limit=limit)
        if not beliefs:
            return ""

        # Stamp the current time in the same "[YYYY-MM-DD HH:MM]" form the beliefs
        # use, so the model can diff it directly against any belief's timestamp.
        if current_time is None:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        elif isinstance(current_time, datetime):
            now_str = current_time.strftime("%Y-%m-%d %H:%M")
        else:
            now_str = str(current_time).strip()

        lines = [f"Current time: [{now_str}]", "--- Injected Context ---"]
        for b in beliefs:
            content = b.get("content", "")
            lines.append(f"• {content}")
            
            # Add to blacklist
            bid = b.get("id")
            if bid and bid not in self._recent_injections:
                self._recent_injections.append(bid)
                
        # Trim blacklist
        if len(self._recent_injections) > self._blacklist_memory_size:
            self._recent_injections = self._recent_injections[-self._blacklist_memory_size:]

        lines.append("------------------------")
        
        return "\n".join(lines)

    def _compute_concept_expansions(self) -> Dict[str, List[str]]:
        """Merges the three concept-expansion layers: built-in defaults and
        learned instances are additive (unioned per category); an explicit
        user override for a category replaces it outright."""
        merged: Dict[str, List[str]] = {k: list(v) for k, v in DEFAULT_CONCEPT_EXPANSIONS.items()}
        for category, instances in self._learned_concept_expansions.items():
            bucket = merged.setdefault(category, [])
            for instance in instances:
                if instance not in bucket:
                    bucket.append(instance)
        for category, instances in self._user_concept_expansions.items():
            merged[category] = list(instances)
        return merged

    def sync_index(self):
        """Sync uncached beliefs into the VectorStore once."""
        if not self._belief_store._cache_loaded:
            self._belief_store.load_into_cache()

        # Fast path: nothing changed in the store since the last sync, so the
        # per-belief scan (O(N) per inject call otherwise) can be skipped.
        store_version = getattr(self._belief_store, "version", None)
        if store_version is not None and store_version == self._last_synced_version:
            return

        # Pick up any concept expansions learned since the last sync (e.g.
        # a consolidation pass tagging "taekwondo" under "martial arts").
        # Piggybacks on the same version check since learning a concept also
        # bumps the store version.
        learned = self._belief_store.get_all_concept_expansions()
        if learned != self._learned_concept_expansions:
            self._learned_concept_expansions = learned
            self._concept_expansions = self._compute_concept_expansions()

        learned_relations = self._belief_store.get_all_relation_expansions()
        if learned_relations != self._learned_relation_expansions:
            self._learned_relation_expansions = learned_relations

        all_beliefs = [
            b for b in self._belief_store._beliefs_cache.values()
            # 'concepts' beliefs are category->instance metadata for the
            # expansion map above, not retrievable content — never index
            # them as candidate facts.
            if b.get("_category") != "concepts"
        ]
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

            # If we don't have the embedding cached, generate it now.
            # Embed the timestamp-stripped content: the leading session-date
            # prefix is for the model's temporal grounding, not the memory
            # system's semantic matching.
            if not cached_embedding:
                emb = self._vector_store.embed_text(_strip_timestamp_prefix(belief.get("content", "")))
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

    def _generate_search_heads(self, text: str) -> List[str]:
        """Builds the set of parallel embedding queries (Multi-Head Retrieval).

        Priority order (highest-value heads first, since the total is capped
        at MAX_SEARCH_HEADS):
        1. The raw query text itself.
        2. Capitalized words (proper nouns: names, places).
        3. Consecutive-noun-phrase bigrams ("charity race", "homeless shelter").
        4. Individual significant single words (Pattern B fix) — a bigram
           extractor alone misses key nouns that appear alone in the query,
           e.g. "school" in "give a speech at a school".
        5. Concept-expansion vocabulary (Pattern A fix) — when the query uses
           abstract language ("achievement", "relationship status"), add the
           concrete synonyms a memory would actually be phrased with.
        6. Named-entity relation expansion — when the query mentions both a
           known subject ("John") and one of that subject's learned relation
           words ("friends"), add the specific named entities from that
           relation ("Bob", "Cindy") so a query never has to name them
           itself for them to become reachable.
        """
        heads = [text]
        words = text.split()
        clean_words = [re.sub(r'[^\w]', '', w) for w in words]
        lowered_words = [w.lower() for w in clean_words]

        if len(words) > 1:
            # Proper nouns
            for w_clean in clean_words[1:]:
                if w_clean and w_clean[0].isupper() and w_clean.lower() not in STOPWORDS:
                    heads.append(w_clean)

            # Noun-phrase bigrams
            noun_phrases = []
            for i in range(len(clean_words) - 1):
                w1, w2 = lowered_words[i], lowered_words[i + 1]
                if w1 not in STOPWORDS and w2 not in STOPWORDS and len(w1) > 2 and len(w2) > 2:
                    if w1 not in _QUESTION_WORDS:
                        noun_phrases.append(f"{clean_words[i]} {clean_words[i + 1]}")
            if noun_phrases:
                heads.extend(noun_phrases[:2])

            # Individual significant single words (Pattern B fix)
            for w_clean, w_lower in zip(clean_words, lowered_words):
                if (
                    len(w_lower) >= 3
                    and w_lower not in STOPWORDS
                    and w_lower not in _QUESTION_WORDS
                    and w_lower not in GENERIC_FILLER_WORDS
                    and w_lower.isalpha()
                ):
                    heads.append(w_lower)

        # Concept expansion (Pattern A fix): abstract query wording gets
        # mapped to the concrete vocabulary a memory would actually use.
        text_lower = text.lower()
        expansion_heads: List[str] = []
        for trigger, expansions in self._concept_expansions.items():
            if trigger in text_lower:
                expansion_heads.extend(expansions)
        expansion_heads = list(dict.fromkeys(expansion_heads))[:MAX_CONCEPT_EXPANSION_HEADS]

        # Named-entity relation expansion: requires BOTH the subject and one
        # of its tagged relation words to appear in the query — matching on
        # the subject alone would fire on every mention of "John" regardless
        # of topic. Stemmed comparison handles simple number/tense variation
        # ("friend" vs "friends") between the learned relation word and the
        # query's own phrasing.
        relation_heads: List[str] = []
        if self._learned_relation_expansions:
            query_word_stems = {_stem_word(w) for w in lowered_words if w}
            for subj, relations in self._learned_relation_expansions.items():
                if subj not in text_lower:
                    continue
                for rel_type, entities in relations.items():
                    if _stem_word(rel_type) in query_word_stems or rel_type in text_lower:
                        relation_heads.extend(entities)
        relation_heads = list(dict.fromkeys(relation_heads))[:MAX_RELATION_EXPANSION_HEADS]

        # Reserve room for expansion/relation heads so a long question's
        # single-word heads can't crowd out these targeted fixes at the
        # final cap — both are the highest-value additions and must survive
        # truncation.
        priority_heads = list(dict.fromkeys(expansion_heads + relation_heads))
        heads = list(dict.fromkeys(heads))
        base_budget = max(1, MAX_SEARCH_HEADS - len(priority_heads))
        heads = heads[:base_budget]

        return list(dict.fromkeys(heads + priority_heads))[:MAX_SEARCH_HEADS]

    def _suppress_merged_constituents(
        self,
        candidates: List[Tuple[float, Dict[str, Any]]],
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """When a turn_synthesis merge belief (see BeliefConsolidator.
        _synthesize_session_merges) is present in the candidate pool
        alongside one or more of its own constituent beliefs, drops the
        constituents — the merge is guaranteed, by the synthesis prompt's
        own contract, to name every specific detail from each constituent,
        so keeping both only spends extra token budget restating the same
        facts. Uses the merge's own `relations` list — an exact link, not a
        similarity estimate — since a merge sentence and a single-item
        original are often phrased too differently to cluster together on
        cosine similarity alone (see _purge_near_duplicates, which handles
        the approximate case).

        Scoped to turn_synthesis only: cluster/structural rollups summarize
        a *pattern* ("2-3 notable highlights"), not a guaranteed-complete
        enumeration, so suppressing their constituents could lose real
        detail that the rollup never claimed to preserve.
        """
        superseded_ids = set()
        for _, belief in candidates:
            if belief.get("source") == "turn_synthesis":
                superseded_ids.update(belief.get("relations", []))
        if not superseded_ids:
            return candidates
        return [(score, belief) for score, belief in candidates if belief.get("id") not in superseded_ids]

    def _purge_near_duplicates(
        self,
        candidates: List[Tuple[float, Dict[str, Any]]],
        full_query_embedding,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Collapses near-duplicate beliefs (the same fact restated across
        sessions) before token-budget selection.

        This is a distinct problem from belief-merge-time deduplication: an
        agent that genuinely volunteers at a shelter every week should keep
        every distinct episodic event (donated a car, organized a meal,
        received a medal — real, different facts). What should collapse is
        the *literal restatement* of one fact ("Maria volunteers at a
        homeless shelter" said three times with only the date changing) —
        those compete for the same limited injection slots as distinct facts
        without adding new information.

        Survivor selection: highest cosine similarity to the raw full query
        text wins; when two members of a cluster are within
        DUPLICATE_TIE_EPSILON of each other, the most recent belief (by its
        session-date prefix) wins instead.
        """
        clusters: List[Dict[str, Any]] = []
        for score, belief in candidates:
            raw_emb = belief.get("embedding") or belief.get("embedding_384d")
            if not raw_emb:
                # No cached embedding to compare with — keep it standalone
                # rather than risk a wrong merge.
                clusters.append({"score": score, "belief": belief, "emb": None, "full_sim": 0.0})
                continue

            emb = np.array(raw_emb, dtype=np.float32)
            full_sim = self._vector_store.cosine_similarity(full_query_embedding, emb)

            placed = False
            for cluster in clusters:
                if cluster["emb"] is None:
                    continue
                if self._vector_store.cosine_similarity(emb, cluster["emb"]) >= DUPLICATE_SIMILARITY_THRESHOLD:
                    is_tied = abs(full_sim - cluster["full_sim"]) <= DUPLICATE_TIE_EPSILON
                    is_better = full_sim > cluster["full_sim"] + DUPLICATE_TIE_EPSILON
                    if is_better or (is_tied and _parse_timestamp_prefix(belief.get("content", "")) > _parse_timestamp_prefix(cluster["belief"].get("content", ""))):
                        cluster.update(score=score, belief=belief, emb=emb, full_sim=full_sim)
                    placed = True
                    break
            if not placed:
                clusters.append({"score": score, "belief": belief, "emb": emb, "full_sim": full_sim})

        return [(c["score"], c["belief"]) for c in clusters]

    def _pull_relevant_beliefs(self, text: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Finds beliefs relevant to the text using Hybrid Union candidate selection
        (keyword + vector top 30) and Three-Way Rank Averaging (keyword rank,
        cosine rank, and meta subject tag rank)."""
        self.sync_index()

        from mrag.core.tagger import extract_keywords_and_phrases, get_tag_counts

        # 1. Parse incoming text for up to 4 keyword and keyphrase searchterms
        query_keywords = extract_keywords_and_phrases(text, limit=4)
        query_tag_counts = get_tag_counts(text)

        # 2. Split keywords into conceptually similar terms based on ever expanding vocabulary (concept expansion)
        keyword_search_groups = []
        for kw in query_keywords:
            terms = {kw.lower()}
            kw_lower = kw.lower()
            for trigger, expansions in self._concept_expansions.items():
                if trigger in kw_lower:
                    for exp in expansions:
                        terms.add(exp.lower())
            keyword_search_groups.append(terms)

        all_beliefs = [
            b for b in self._belief_store.get_all_beliefs_flat()
            if b.get("_category") != "concepts"
        ]

        if not all_beliefs:
            return []
            
        dynamic_top_k = max(30, min(60, int(len(all_beliefs) * 0.10)))

        # 3. Compute keyword and tag matches for all beliefs
        keyword_scores = {}
        tag_scores = {}
        for b in all_beliefs:
            bid = b["id"]
            content_lower = _strip_timestamp_prefix(b.get("content", "")).lower()
            
            # Count how many of the query's keyword groups this belief matches
            keyword_matches = 0
            for terms in keyword_search_groups:
                matched = False
                for term in terms:
                    if " " in term:
                        if term in content_lower:
                            matched = True
                            break
                    else:
                        if re.search(r'\b' + re.escape(term) + r'\b', content_lower):
                            matched = True
                            break
                if matched:
                    keyword_matches += 1

            # Count tag overlaps
            b_tags = b.get("conceptual_tags", [])
            tag_matches = 0
            for tag, q_count in query_tag_counts.items():
                b_count = b_tags.count(tag) if isinstance(b_tags, list) else 0
                tag_matches += min(q_count, b_count)

            keyword_scores[bid] = keyword_matches
            tag_scores[bid] = tag_matches

        # 4. Hybrid Union Candidate Selection
        # A. Keyword Candidates: sort by (keyword_matches, relevance) descending
        sorted_kw_candidates = sorted(
            all_beliefs,
            key=lambda b: (keyword_scores[b["id"]], b.get("relevance", 0.0)),
            reverse=True
        )
        keyword_top_k = sorted_kw_candidates[:dynamic_top_k]

        # B. Vector Candidates: query vector store
        query_embedding = self._vector_store.embed_text(text)
        vector_top_k_results = self._vector_store.query_top_k(query_embedding, k=dynamic_top_k)
        vector_top_k_ids = [res[0] for res in vector_top_k_results]

        vector_top_k = []
        for bid in vector_top_k_ids:
            if bid in self._belief_store._beliefs_cache:
                vector_top_k.append(self._belief_store._beliefs_cache[bid])

        # C. Construct the Union candidate pool
        union_dict = {}
        for b in keyword_top_k:
            union_dict[b["id"]] = b
        for b in vector_top_k:
            union_dict[b["id"]] = b

        union_candidates = list(union_dict.values())
        if not union_candidates:
            union_candidates = all_beliefs[:dynamic_top_k]

        # 5. Three-Way Soft-Score Fusion
        # Compute cosine similarity for all candidates in union
        cosine_similarities = {}
        for b in union_candidates:
            bid = b["id"]
            cached_embedding = b.get("embedding") or b.get("embedding_384d")
            if not cached_embedding:
                emb = self._vector_store.embed_text(_strip_timestamp_prefix(b.get("content", "")))
                cached_embedding = emb.tolist()
                b["embedding"] = cached_embedding
                self._belief_store._beliefs_cache[bid] = b
            
            emb_np = np.array(cached_embedding, dtype=np.float32)
            cosine_similarities[bid] = self._vector_store.cosine_similarity(query_embedding, emb_np)

        import math
        # Compute dynamic nudge weight based on the standard deviation of similarities
        # This ensures the bump is always meaningful relative to the local score gradient,
        # but never overpowering. We cap it between 0.01 and 0.10.
        cos_sims = [cosine_similarities[b["id"]] for b in union_candidates]
        sim_std = np.std(cos_sims) if cos_sims else 0.05
        nudge_weight = max(0.01, min(0.10, sim_std * 1.5))
        
        fused_candidates = []
        for b in union_candidates:
            bid = b["id"]
            cos_sim = cosine_similarities[bid]
            kw_match = keyword_scores.get(bid, 0)
            tag_match = tag_scores.get(bid, 0)
            
            # Logarithmic scaling so 1 match gives a good nudge, but multiple matches don't blow up the score
            kw_boost = nudge_weight * math.log(1 + kw_match)
            tag_boost = nudge_weight * 0.5 * math.log(1 + tag_match)
            
            fused_score = cos_sim + kw_boost + tag_boost
            fused_candidates.append((b, fused_score))

        # 6. Repetition Prevention and Sorting
        # Sort by: (is_blacklisted, -fused_score, -relevance)
        fused_candidates.sort(key=lambda x: (
            x[0]["id"] in self._recent_injections,
            -x[1],
            -x[0].get("relevance", 0.0)
        ))

        final_candidates = [b for b, fused_score in fused_candidates]

        # Suppress constituents covered by turn synthesis merges
        final_candidates_with_dummy_scores = [(1.0, b) for b in final_candidates]
        suppressed = self._suppress_merged_constituents(final_candidates_with_dummy_scores)
        final_candidates = [b for _, b in suppressed]

        # Results-overlap near-duplicate purge
        purged = self._purge_near_duplicates([(1.0, b) for b in final_candidates], query_embedding)
        final_candidates = [b for _, b in purged]

        # Get average token length of all beliefs in the store
        all_store_beliefs = [
            b for b in self._belief_store.get_all_beliefs_flat()
            if b.get("_category") != "concepts"
        ]
        if all_store_beliefs:
            total_store_tokens = sum(estimate_tokens(b.get("content", "")) for b in all_store_beliefs)
            avg_belief_tokens = total_store_tokens / len(all_store_beliefs)
        else:
            avg_belief_tokens = 30.0

        max_tokens = min(15.0 * avg_belief_tokens, self.max_injected_tokens)

        final_beliefs = []
        current_tokens = 0
        
        for b in final_candidates:
            content = b.get("content", "")
            b_tokens = estimate_tokens(content)
            
            if not final_beliefs:
                # Always include at least one belief
                final_beliefs.append(b)
                current_tokens += b_tokens
                continue
                
            new_tokens = current_tokens + b_tokens
            if new_tokens > max_tokens:
                break
                
            final_beliefs.append(b)
            current_tokens += b_tokens
                
        if limit is not None:
            return final_beliefs[:limit]
            
        return final_beliefs

    def clear_blacklist(self):
        """Reset the rolling blacklist."""
        self._recent_injections.clear()
