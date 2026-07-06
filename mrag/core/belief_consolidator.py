import hashlib
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from mrag.memory.belief_store import BeliefStore
from mrag.core.pre_generative_injection import estimate_tokens, _strip_timestamp_prefix

logger = logging.getLogger("mrag.core.belief_consolidator")

class BeliefConsolidator:
    """Consolidates conversational logs into structured beliefs.

    Extracts premises directly from interaction logs session-by-session using a fast,
    single-session LLM extraction step that converts dialogue into clean third-person facts.
    """

    # A cluster (same subject + learned category, e.g. Maria's "volunteering"
    # facts) is consolidated into one rollup belief the moment either bound is
    # crossed — modeled on ContextCompressor's immediate, threshold-triggered
    # design rather than a deferred/batched queue. A pending cluster is a
    # known-degraded retrieval topic; queuing it for a later batch pass would
    # leave every query on that topic worse off until the batch runs, for a
    # batching-efficiency win that rarely materializes (clusters don't arrive
    # in the kind of bursts that make batching pay off).
    CLUSTER_SIZE_THRESHOLD = 10
    CLUSTER_TOKEN_FRACTION_THRESHOLD = 0.25

    def __init__(
        self,
        belief_store: BeliefStore,
        llm_callable: Callable[[str], str],
        context_limit: Optional[int] = None,
        ratio: float = 0.40,
        vector_store: Optional[Any] = None,
        max_injected_tokens: int = 800,
        enable_session_synthesis: bool = True,
    ):
        self._store = belief_store
        self.llm = llm_callable
        self._vector_store = vector_store
        self.enable_session_synthesis = enable_session_synthesis
        # Reference point for the cluster token-threshold — mirrors
        # PreGenerativeInjector's default so a cluster is flagged right around
        # the point where it would actually crowd out other injected facts.
        self.max_injected_tokens = max_injected_tokens

        import os
        from mrag.core.context_compressor import resolve_context_limit
        try:
            model_name = os.environ.get("MRAG_MODEL_NAME")
            self.context_limit = resolve_context_limit(context_limit, model_name)
        except ValueError:
            self.context_limit = 8192

        self.ratio = ratio
        self.backlog_threshold_tokens = min(int(self.context_limit * ratio), 10000)
        self._backlog: List[Dict[str, Any]] = []
        self._backlog_tokens = 0

    def add_conversation_turn(self, turn: Dict[str, Any]):
        """Directly extracts and summarizes factual statements from the conversation session
        using a single fast LLM translation call, prepending a standardized timestamp.
        """
        content = turn.get("content", "")
        if not content:
            return

        lines = content.split("\n")
        date_context = ""

        # Check if the first line is Date
        if lines and lines[0].startswith("Date:"):
            date_context = lines[0].replace("Date:", "").strip()
            # Convert date to standard short timestamp format if possible
            date_context = self._parse_date_to_timestamp(date_context)
            lines = lines[1:]

        # Re-assemble the turns text
        turns_text = "\n".join(lines).strip()
        if not turns_text:
            return

        # Call fast LLM extractor for this session
        facts = self._extract_session_factual_summaries(turns_text, date_context)
        touched_clusters = set()
        session_beliefs: List[tuple] = []

        for fact in facts:
            summary = fact.get("content")
            if not summary:
                continue

            # Prepend the short timestamp prefix
            if date_context:
                belief_content = f"[{date_context}] {summary}"
            else:
                belief_content = summary

            content_digest = hashlib.sha256(belief_content.encode("utf-8")).hexdigest()[:12]
            belief_id = f"bel_{content_digest}"

            self._store.merge_or_add_belief(
                category="premises",
                belief_id=belief_id,
                content=belief_content,
                confidence=1.0,
                source="session_summarization",
                vector_store=self._vector_store
            )
            logger.info(f"Extracted factual belief: {belief_content}")
            session_beliefs.append((belief_id, belief_content))

            category = fact.get("category")
            instance = fact.get("instance")
            subject = fact.get("subject")
            relation = fact.get("relation")
            entities = fact.get("entities")
            if category and instance:
                self._store.learn_concept_expansion(category, instance)
            if category and subject:
                self._store.tag_cluster_membership(subject, category, belief_id)
                touched_clusters.add((subject.strip().lower(), category.strip().lower()))
            if relation and entities and subject:
                self._store.learn_relation_expansion(subject, relation, entities)

        for subject, category in touched_clusters:
            self._check_and_consolidate_cluster(subject, category)

        if len(session_beliefs) >= 3 and self.enable_session_synthesis:
            self._synthesize_session_merges(session_beliefs, date_context)

    def _synthesize_session_merges(self, session_beliefs: List[tuple], date_context: str):
        """Additive per-session pass: reviews the beliefs just extracted
        from this session and, without replacing any of them, adds a
        combined belief for any genuine same-subject-same-relation group
        (e.g. several "John loves X" facts -> one "John loves X, Y, and Z"
        belief). Runs every session rather than waiting on a cluster-size
        threshold, since the goal here is same-session compaction/entity-
        linking, not managing a long-lived pileup across many sessions
        (that's _check_and_consolidate_cluster's job — this supplements,
        never replaces, that mechanism).

        Uses the LLM's own holistic judgment over the full list of
        just-extracted facts rather than pre-assigned category/predicate
        tags matched mechanically — real conversational phrasing varies too
        much ("loves" vs "enjoys" vs "is into") for rigid template matching
        to reliably find these groups (established earlier this session
        with the belief-merge contradiction detector).
        """
        facts_list = "\n".join(f"{i + 1}. {content}" for i, (_, content) in enumerate(session_beliefs))
        prompt = f"""Here are {len(session_beliefs)} facts extracted from a single conversation session:
{facts_list}

Find groups of 3 OR MORE facts that are each a SEPARATE, PARALLEL ITEM of the exact same kind under
the exact same predicate for the exact same person — the kind of facts that would naturally read as a
list joined by "and" if put in one sentence (e.g. three separate hobbies, three separate named friends,
three separate pets, three separate possessions). For each such group, produce ONE sentence combining
the items as a list, naming every specific detail from the originals.

Do NOT merge:
- Facts that are part of one narrative or chronological account of a single event/conversation, even if
  they mention the same person multiple times (e.g. "attended a support group" + "found it inspiring" +
  "felt more courage" is ONE experience described in steps, not 3 parallel items — do not touch this).
- Fewer than 3 facts (2 facts save almost nothing and aren't worth adding a new belief for).
- Facts about different, unrelated topics just because they mention the same person.
Only include groups you are highly confident are genuinely parallel list items. It is correct and
expected for most sessions to have ZERO qualifying groups — do not force a match to have something to
return.

Return ONLY a JSON object:
{{"merged": [{{"content": "combined sentence naming every specific detail", "source_indices": [1, 3, 5]}}]}}

If nothing qualifies, return {{"merged": []}}.
"""
        try:
            response = self.llm(prompt).strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            data = json.loads(response.strip())
        except Exception as e:
            logger.error(f"Failed to synthesize session merges: {e}")
            return

        if not isinstance(data, dict):
            return

        for item in data.get("merged", []) or []:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            indices = item.get("source_indices", []) or []
            if not content or not isinstance(indices, list) or len(indices) < 3:
                continue

            source_ids = []
            for idx in indices:
                try:
                    i = int(idx) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= i < len(session_beliefs):
                    source_ids.append(session_beliefs[i][0])
            if len(source_ids) < 3:
                continue

            belief_content = f"[{date_context}] {content}" if date_context else content
            digest = hashlib.sha256(f"synth:{belief_content}".encode("utf-8")).hexdigest()[:12]
            merge_id = f"bel_synth_{digest}"
            self._store.add_belief(
                category="premises",
                belief_id=merge_id,
                content=belief_content,
                confidence=1.0,
                source="turn_synthesis",
                relations=source_ids,
            )
            logger.info(f"Synthesized merged belief from session facts {indices}: {belief_content}")

    def _check_and_consolidate_cluster(self, subject: str, category: str):
        """Fires the moment a (subject, category) cluster crosses either
        threshold — see CLUSTER_SIZE_THRESHOLD / CLUSTER_TOKEN_FRACTION_THRESHOLD.
        """
        member_ids = self._store.get_cluster_members(subject, category)
        if self._store.is_cluster_consolidated(subject, category):
            return

        members = [self._store.get_belief(bid) for bid in member_ids]
        members = [b for b in members if b]
        if not members:
            return

        token_total = sum(estimate_tokens(b.get("content", "")) for b in members)
        token_limit = self.max_injected_tokens * self.CLUSTER_TOKEN_FRACTION_THRESHOLD

        if len(members) < self.CLUSTER_SIZE_THRESHOLD and token_total < token_limit:
            return

        rollup_content = self._generate_cluster_rollup(subject, category, members)
        if not rollup_content:
            return

        rollup_digest = hashlib.sha256(f"rollup:{subject}:{category}".encode("utf-8")).hexdigest()[:12]
        rollup_id = f"bel_rollup_{rollup_digest}"
        self._store.add_belief(
            category="premises",
            belief_id=rollup_id,
            content=rollup_content,
            confidence=1.0,
            source="cluster_consolidation",
            relations=list(member_ids),
        )
        self._store.mark_cluster_consolidated(subject, category)
        logger.info(
            f"Consolidated cluster ({subject}, {category}, {len(members)} beliefs, "
            f"{token_total} tokens) into rollup: {rollup_content}"
        )

    def _generate_cluster_rollup(self, subject: str, category: Optional[str], members: List[Dict[str, Any]]) -> Optional[str]:
        facts_text = "\n".join(f"- {b.get('content', '')}" for b in members)
        about = f"{subject}'s {category}" if category else f"{subject}"
        prompt = f"""You are consolidating a cluster of {len(members)} related facts about {about} into ONE concise rollup statement.

Facts:
{facts_text}

Produce ONE sentence (or short paragraph) that captures the overall pattern — frequency, timespan, and
2-3 of the most notable specific highlights from the facts above. Do not invent anything not present in
the facts. Do not lose specificity that would make this generic — name concrete standout details where
relevant, not just "has done many things".

Return ONLY the rollup sentence. No preamble, no markdown, no quotes.
"""
        try:
            return self.llm(prompt).strip().strip('"')
        except Exception as e:
            logger.error(f"Failed to generate cluster rollup for ({subject}, {category}): {e}")
            return None

    @staticmethod
    def _infer_subject(content: str) -> Optional[str]:
        """Extracts the belief's subject from its content when no explicit
        subject tag exists. Consolidation-authored facts consistently open
        with the subject's name (see _extract_session_factual_summaries rule
        2: "Rewrite all pronouns to refer to the specific person's name"),
        so the first capitalized word is a reliable, cheap proxy.
        """
        content = re.sub(r'^\[[\d\-: ]+\]\s*', '', content)
        m = re.match(r"([A-Z][a-zA-Z]+)", content)
        return m.group(1) if m else None

    def discover_and_consolidate_clusters(
        self,
        min_cluster_size: int = 5,
        overlap_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """Periodic structural pass: finds naturally dense clusters of
        beliefs by embedding-similarity geometry (HDBSCAN), independent of
        whether the LLM tagged them with an explicit category — catching
        pileups the immediate, per-turn tag-based consolidation misses
        (see _check_and_consolidate_cluster) because no clean category name
        applied. Rollups from this pass supplement, they never replace, the
        immediate tag-based mechanism, which remains the primary defense.

        Unlike the tag-based trigger, this is meant to be called
        periodically/on-demand (like run_nightly_decay), not per-turn:
        HDBSCAN needs a set of points to cluster meaningfully, so re-running
        it after every single new belief would mean re-clustering a
        subject's entire history on every turn.

        Requires the optional 'clustering' extra: pip install mrag[clustering].
        """
        try:
            import numpy as np
            from sklearn.cluster import HDBSCAN
        except ImportError:
            raise ImportError(
                "Structural cluster discovery requires scikit-learn. "
                "Install it with `pip install mrag[clustering]` or `pip install scikit-learn>=1.3`."
            )

        if not self._store._cache_loaded:
            self._store.load_into_cache()

        by_subject: Dict[str, List[Dict[str, Any]]] = {}
        for belief in self._store.get_all_beliefs_flat():
            if belief.get("_category") not in ("premises", "propositions", "preferences"):
                continue
            if belief.get("source") in ("cluster_consolidation", "structural_cluster_discovery"):
                continue  # never re-cluster rollups themselves
            subject = self._infer_subject(belief.get("content", ""))
            if subject:
                by_subject.setdefault(subject, []).append(belief)

        existing_clusters = [set(c) for c in self._store.get_structural_clusters()]
        stats = {"subjects_scanned": 0, "clusters_found": 0, "rollups_created": 0}

        for subject, beliefs in by_subject.items():
            stats["subjects_scanned"] += 1
            if len(beliefs) < min_cluster_size:
                continue

            embeddings, valid_beliefs = [], []
            for b in beliefs:
                emb = b.get("embedding") or b.get("embedding_384d")
                if not emb:
                    # Beliefs only get a cached embedding lazily, the first
                    # time PreGenerativeInjector.sync_index() runs. Clustering
                    # can be invoked as its own periodic pass with no
                    # injector having run yet, so compute and persist it here
                    # rather than silently skipping every un-synced belief.
                    if not self._vector_store:
                        continue
                    computed = self._vector_store.embed_text(_strip_timestamp_prefix(b.get("content", "")))
                    emb = computed.tolist()
                    b["embedding"] = emb
                    self._store.update_belief(b.get("_category", "premises"), b)
                if emb:
                    embeddings.append(emb)
                    valid_beliefs.append(b)
            if len(valid_beliefs) < min_cluster_size:
                continue

            X = np.array(embeddings, dtype=np.float64)
            labels = HDBSCAN(min_cluster_size=min_cluster_size, metric="cosine").fit_predict(X)

            cluster_groups: Dict[int, List[Dict[str, Any]]] = {}
            for label, belief in zip(labels, valid_beliefs):
                if label == -1:  # HDBSCAN's noise label -- not a real cluster
                    continue
                cluster_groups.setdefault(int(label), []).append(belief)

            for members in cluster_groups.values():
                stats["clusters_found"] += 1
                member_ids = {b["id"] for b in members}

                already_covered = any(
                    len(member_ids & existing) / len(member_ids) >= overlap_threshold
                    for existing in existing_clusters
                )
                if already_covered:
                    continue

                rollup_content = self._generate_cluster_rollup(subject, None, members)
                if not rollup_content:
                    continue

                rollup_digest = hashlib.sha256(
                    f"structrollup:{subject}:{sorted(member_ids)}".encode("utf-8")
                ).hexdigest()[:12]
                rollup_id = f"bel_structrollup_{rollup_digest}"
                self._store.add_belief(
                    category="premises",
                    belief_id=rollup_id,
                    content=rollup_content,
                    confidence=1.0,
                    source="structural_cluster_discovery",
                    relations=list(member_ids),
                )
                self._store.record_structural_cluster(list(member_ids), rollup_id)
                existing_clusters.append(member_ids)
                stats["rollups_created"] += 1
                logger.info(
                    f"Structural cluster discovered for {subject} ({len(members)} beliefs) "
                    f"-> rollup: {rollup_content}"
                )

        return stats

    def _parse_date_to_timestamp(self, date_str: str) -> str:
        """Parses complex date strings into standard short YYYY-MM-DD HH:MM formats."""
        months = {
            "january": "01", "february": "02", "march": "03", "april": "04",
            "may": "05", "june": "06", "july": "07", "august": "08",
            "september": "09", "october": "10", "november": "11", "december": "12",
            "jan": "01", "feb": "02", "mar": "03", "apr": "04",
            "jun": "06", "jul": "07", "aug": "08", "sep": "09",
            "oct": "10", "nov": "11", "dec": "12"
        }
        date_str_clean = date_str.lower().strip()
        
        # Match format: "8:56 pm on 20 July, 2023"
        m1 = re.search(r'(\d+):(\d+)\s*(am|pm)\s+on\s+(\d+)\s+([a-z]+),?\s+(\d{4})', date_str_clean)
        if m1:
            h, m, am_pm, d, mon, y = m1.groups()
            h_int = int(h)
            if am_pm == "pm" and h_int < 12:
                h_int += 12
            elif am_pm == "am" and h_int == 12:
                h_int = 0
            mon_num = months.get(mon, "01")
            return f"{y}-{mon_num}-{int(d):02d} {h_int:02d}:{int(m):02d}"
            
        # Match format: "may 8, 2023"
        m2 = re.search(r'([a-z]+)\s+(\d+),?\s+(\d{4})', date_str_clean)
        if m2:
            mon, d, y = m2.groups()
            mon_num = months.get(mon, "01")
            return f"{y}-{mon_num}-{int(d):02d}"
            
        return date_str

    def _extract_session_factual_summaries(self, turns_text: str, date_context: str) -> List[Dict[str, Any]]:
        """Returns a list of {"content", "category", "instance", "subject",
        "relation", "entities"} dicts (all but "content" optional/None).
        Tagging a fact serves three purposes downstream:
        - (category, instance) grows the injector's concept-expansion
          vocabulary (see BeliefStore.learn_concept_expansion) so abstract
          query wording ("What martial arts...") keeps reaching concrete
          facts ("...practices taekwondo") without hand-curating every
          category upfront.
        - (subject, relation, entities) grows the injector's relation-
          expansion vocabulary (see BeliefStore.learn_relation_expansion) so
          "John's friends are Bob and Cindy" makes "Bob"/"Cindy" reachable
          from any future query that mentions "John" and "friends", even
          though the query never names them.
        - (subject, category) identifies which cluster a fact belongs to
          (e.g. Maria's "volunteering" facts), so a cluster that grows large
          enough to dilute retrieval can be consolidated (see
          BeliefStore.tag_cluster_membership / _check_and_consolidate_cluster).
        Tags live on the fact itself (not a parallel array) so there's no
        ambiguity about which belief a tag applies to.
        """
        prompt = f"""You are a factual extractor. Translate the following conversation turns from a single session into structured JSON.

Rules:
1. Extract specific facts, dates, events, plans, preferences, declarations, and background info.
2. Rewrite all pronouns to refer to the specific person's name (e.g. "I" -> "Caroline" or "Melanie", "my kids" -> "Melanie's kids").
3. Convert relative dates (like "yesterday", "last weekend", "next month", "this past Friday") to absolute dates or time periods using the session Date context.
4. NEVER generalize away a concrete noun, object, named entity, or specific detail. Preserve it exactly as stated.
   - WRONG: "Researching adoption agencies" -> "Caroline is conducting research."
   - RIGHT: "Researching adoption agencies" -> "Caroline is researching adoption agencies."
   - WRONG: "reading to the kids at the shelter" -> "Caroline volunteers."
   - RIGHT: "reading to the kids at the shelter" -> "Caroline reads to kids at a homeless shelter."
5. A casual tone does NOT make a sentence fluff. If a casual or offhand remark contains a concrete
   time reference, named activity, or event, extract it as a fact — do not discard it just because it
   reads like small talk or is attached to a photo/caption.
   - WRONG (discarded as fluff): "Here's a pic from when we met up last week!" -> (nothing extracted)
   - RIGHT: "Here's a pic from when we met up last week!" -> "Caroline met up with friends last week."
6. Only exclude turns that carry zero factual, temporal, or event content at all (pure greetings like
   "Hey, how are you?", reactions like "lol nice", or bare questions with no answer given).
7. If there is no factual info in the session, return an empty "facts" array.
8. Additionally, for any fact that names a SPECIFIC INSTANCE of a common general category (a type of
   sport, martial art, hobby, cuisine, instrument, pet, genre, etc.), add "category", "instance", and
   "subject" fields to that fact's object. "subject" is the person's name the fact is about. Omit these
   three fields entirely for facts that don't fit a clear general category — don't force it.
   - "John is participating in taekwondo" -> category: "martial arts", instance: "taekwondo", subject: "John"
   - "Maria adopted a corgi" -> category: "pet", instance: "corgi", subject: "Maria"
9. Additionally, whenever a fact names one or more SPECIFIC PEOPLE (actual proper names, e.g. "Bob",
   "Cindy") in an explicit relationship to another person (friends, kids/children, siblings, parents,
   spouse/partner, coworkers, pets by name), add "subject" (the person the relation belongs to),
   "relation" (a short, consistent word for the relation type, e.g. "friends", "kids", "siblings",
   "coworkers", "pets"), and "entities" (list of the specific PROPER NAMES ONLY) fields to that fact's
   object. "entities" must be actual names, never a description, count, or possessive phrase — if no
   specific name is given, omit these three fields entirely rather than inventing a placeholder.
   - "John's friends are Bob and Cindy" -> subject: "John", relation: "friends", entities: ["Bob", "Cindy"]
   - "Dan's kids Jane and Lucy went to the fair" -> subject: "Dan", relation: "kids", entities: ["Jane", "Lucy"]
   - "Melanie has two younger kids" -> omit relation/entities entirely (no names given)
   - WRONG: entities: ["2 younger kids"], entities: ["Melanie's kids"], entities: ["youngest child"]

Session Date context: {date_context}
Turns:
{turns_text}

Output format: Return ONLY a JSON object, for example:
{{
  "facts": [
    {{"content": "Caroline went to an LGBTQ support group on May 7, 2023."}},
    {{"content": "John is participating in taekwondo on December 22, 2022.", "category": "martial arts", "instance": "taekwondo", "subject": "John"}},
    {{"content": "Dan's kids Jane and Lucy went to the fair on June 2, 2023.", "subject": "Dan", "relation": "kids", "entities": ["Jane", "Lucy"]}}
  ]
}}
"""
        try:
            response = self.llm(prompt).strip()

            # Clean up markdown if the LLM leaked it
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            data = json.loads(response.strip())

            # Backward/defensive compatibility: accept a bare list of plain
            # strings (older schema, or a model that ignores the tagging
            # instructions) rather than losing the facts entirely.
            if isinstance(data, list):
                return [{"content": str(item)} for item in data if item]

            if isinstance(data, dict):
                facts = []
                for item in data.get("facts", []) or []:
                    if isinstance(item, str):
                        if item:
                            facts.append({"content": item})
                        continue
                    if not isinstance(item, dict):
                        continue
                    content = str(item.get("content", "")).strip()
                    if not content:
                        continue
                    fact = {"content": content}
                    category = str(item.get("category", "") or "").strip()
                    instance = str(item.get("instance", "") or "").strip()
                    subject = str(item.get("subject", "") or "").strip()
                    relation = str(item.get("relation", "") or "").strip()
                    entities_raw = item.get("entities", []) or []
                    entities = [str(e).strip() for e in entities_raw if str(e).strip()] if isinstance(entities_raw, list) else []
                    if category and instance:
                        fact["category"] = category
                        fact["instance"] = instance
                    if subject and (category or (relation and entities)):
                        fact["subject"] = subject
                    if relation and entities:
                        fact["relation"] = relation
                        fact["entities"] = entities
                    facts.append(fact)
                return facts

            return []
        except Exception as e:
            logger.error(f"Failed to extract session factual summaries: {e}")
            return []

    def clear_backlog(self):
        self._backlog = []
        self._backlog_tokens = 0

    def run_consolidation_pass(self, conversation_turns: List[Dict[str, Any]]):
        # Already processed turn-by-turn, so no-op
        pass

    def run_nightly_decay(self):
        stats = self._store.decay_all_beliefs()
        logger.info(f"Nightly decay complete: {stats}")
        return stats
