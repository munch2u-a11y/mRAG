"""Raw-text, LLM-free memory ingestion.

Stores each atomic text event of a conversation — a user input, an assistant
output, a thinking block, a tool return — exactly as written, chunked into
~100-token parts, with a timestamp prefix and source metadata. This replaces
the front-end LLM belief-extraction pass entirely: writes are pure text
handling (chunk, hash, persist), so ingestion costs zero LLM calls and loses
zero information to extraction judgment.

Interpretation happens later and offline: BeliefConsolidator.run_nightly_review
sweeps unreviewed memory chunks in large batches and forms subjective
Layer 2 beliefs (term-anchored profile/concept statements) from them, with
provenance links back to the chunks. Because the raw record is permanent,
a formed belief is a derived index over history — never the only copy of it.
"""

import hashlib
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from mrag.memory.belief_store import BeliefStore
from mrag.core.belief_consolidator import parse_date_to_timestamp
from mrag.core.token_counting import count_text_tokens

logger = logging.getLogger("mrag.core.memory_ingestor")

# Target size of one stored chunk. Small enough that a retrieval hit injects
# only the relevant slice of a long event, large enough to keep a full
# thought/sentence group intact.
DEFAULT_CHUNK_TOKENS = 100
# A sentence group may finish its last sentence past the target rather than
# split mid-sentence; a single sentence longer than target * this factor is
# hard-split on word boundaries instead.
_OVERSHOOT_FACTOR = 1.5

# Sentence-ish boundaries: end punctuation followed by whitespace, or any
# newline run (covers code, logs, and list-formatted tool returns where
# punctuation-based splitting finds nothing).
_SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+|\n+')

# Conventional source labels. Free-form strings are accepted — these just
# name the four directions of text flow the design distinguishes.
SOURCE_USER_INPUT = "user_input"
SOURCE_ASSISTANT_OUTPUT = "assistant_output"
SOURCE_THINKING = "thinking"
SOURCE_TOOL_RETURN = "tool_return"


def chunk_text(text: str, target_tokens: int = DEFAULT_CHUNK_TOKENS) -> List[str]:
    """Split text into chunks of roughly target_tokens each.

    Sentences are packed greedily and never split unless a single sentence
    alone exceeds target_tokens * _OVERSHOOT_FACTOR, in which case it is
    hard-split on word boundaries. Text at or under the target comes back
    as a single chunk unchanged.
    """
    text = text.strip()
    if not text:
        return []
    if count_text_tokens(text) <= target_tokens:
        return [text]

    sentences = [s.strip() for s in _SENTENCE_BOUNDARY_RE.split(text) if s and s.strip()]

    pieces: List[str] = []
    for sentence in sentences:
        if count_text_tokens(sentence) > target_tokens * _OVERSHOOT_FACTOR:
            pieces.extend(_hard_split_words(sentence, target_tokens))
        else:
            pieces.append(sentence)

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for piece in pieces:
        piece_tokens = count_text_tokens(piece) or 0
        if current and current_tokens + piece_tokens > target_tokens:
            chunks.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(piece)
        current_tokens += piece_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def _hard_split_words(sentence: str, target_tokens: int) -> List[str]:
    """Fallback for a single sentence too long to store whole: split on
    word boundaries at the token budget."""
    words = sentence.split()
    parts: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for word in words:
        word_tokens = count_text_tokens(word) or 1
        if current and current_tokens + word_tokens > target_tokens:
            parts.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(word)
        current_tokens += word_tokens
    if current:
        parts.append(" ".join(current))
    return parts


def _normalize_timestamp(timestamp: Any) -> str:
    """Accepts a datetime or a date string in any of the formats the
    consolidator already understands; returns the store's canonical
    "YYYY-MM-DD HH:MM" short form (or the raw string if unparseable —
    consistent with parse_date_to_timestamp's passthrough behavior)."""
    if timestamp is None:
        return datetime.now().strftime("%Y-%m-%d %H:%M")
    if isinstance(timestamp, datetime):
        return timestamp.strftime("%Y-%m-%d %H:%M")
    return parse_date_to_timestamp(str(timestamp).strip())


class MemoryIngestor:
    """Chunks and persists raw conversation text into the 'memory'
    belief category. No LLM involved anywhere on this path."""

    def __init__(
        self,
        belief_store: BeliefStore,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    ):
        self._store = belief_store
        self.chunk_tokens = chunk_tokens

    def add_event(
        self,
        text: str,
        source: str = SOURCE_USER_INPUT,
        timestamp: Any = None,
        session_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        speaker: Optional[str] = None,
    ) -> List[str]:
        """Store one atomic text event (a single input, output, thinking
        block, or tool return) as memory chunks. Returns the belief ids of
        the chunks written (empty for empty/duplicate input).

        speaker, when given, is prefixed into each chunk's content
        ("[ts] Caroline: ...") — verbatim text keeps its first-person
        pronouns, so attribution has to travel with the text itself for
        retrieval and the answering model to resolve who "I" is.
        """
        text = (text or "").strip()
        if not text:
            return []

        ts = _normalize_timestamp(timestamp)
        chunks = chunk_text(text, self.chunk_tokens)
        if not chunks:
            return []

        # Session timestamps are often whole-day (or shared by every turn of
        # a replayed session), so they can't order events within a session.
        # A wall-clock ingestion stamp with microseconds preserves arrival
        # order for the nightly review's chronological batching.
        ingested_at = datetime.now().isoformat(timespec="microseconds")

        # Event identity is content-derived: replaying the same event is a
        # no-op (add_belief refuses duplicate ids), not a second copy.
        event_digest = hashlib.sha256(
            f"{ts}|{source}|{speaker or ''}|{text}".encode("utf-8")
        ).hexdigest()[:12]
        event_id = f"evt_{event_digest}"

        written: List[str] = []
        for index, chunk in enumerate(chunks):
            body = f"{speaker}: {chunk}" if speaker else chunk
            content = f"[{ts}] {body}" if ts else body

            chunk_digest = hashlib.sha256(
                f"{event_id}:{index}:{chunk}".encode("utf-8")
            ).hexdigest()[:12]
            belief_id = f"mem_{chunk_digest}"

            added = self._store.add_belief(
                category="memory",
                belief_id=belief_id,
                content=content,
                confidence=1.0,
                source=source,
                event_id=event_id,
                ingested_at=ingested_at,
                chunk_index=index,
                chunk_count=len(chunks),
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                reviewed_at=None,
            )
            if added:
                written.append(belief_id)

        if written:
            logger.info(
                "Stored memory event %s (%s, %d chunk(s), session=%s turn=%s)",
                event_id, source, len(written), session_id, turn_id,
            )
        return written

    def add_turn(
        self,
        turn: Dict[str, Any],
        session_id: Optional[str] = None,
        turn_id: Optional[str] = None,
    ) -> List[str]:
        """Convenience for the {"content": "Date: ...\\n<speaker>: ..."} turn
        shape the benchmark harnesses already produce: peels a leading
        "Date:" line into the timestamp and a leading "Name:" prefix into the
        speaker, then stores the remainder as one event."""
        content = (turn.get("content") or "").strip()
        if not content:
            return []

        lines = content.split("\n")
        timestamp = None
        if lines and lines[0].startswith("Date:"):
            timestamp = lines[0].replace("Date:", "").strip()
            lines = lines[1:]
        body = "\n".join(lines).strip()

        speaker = None
        m = re.match(r'^([A-Z][\w \-\'\.]{0,40}?):\s+(.*)$', body, re.DOTALL)
        if m:
            speaker, body = m.group(1), m.group(2)

        source = turn.get("source") or turn.get("role") or SOURCE_USER_INPUT
        return self.add_event(
            body,
            source=source,
            timestamp=timestamp,
            session_id=session_id,
            turn_id=turn_id or turn.get("turn_id"),
            speaker=speaker,
        )
