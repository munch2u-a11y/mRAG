import hashlib
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from mrag.memory.belief_store import BeliefStore

logger = logging.getLogger("mrag.core.belief_consolidator")

class BeliefConsolidator:
    """Consolidates conversational logs into structured beliefs.
    
    Extracts premises, propositions, and preferences from interaction logs
    and saves them into the BeliefStore with temporal and structural metadata.
    """

    def __init__(self, belief_store: BeliefStore, llm_callable: Callable[[str], str], context_limit: Optional[int] = None, ratio: float = 0.40, vector_store: Optional[Any] = None):
        """
        Args:
            belief_store: The cognitive belief store.
            llm_callable: Function that accepts a prompt string and returns LLM string output.
            context_limit: The model's context window size. Defaults to 8192 if not resolved.
            ratio: Fraction of the context window to accumulate before triggering consolidation.
                   Note: The threshold is capped at 10,000 tokens to preserve extraction recall and output limit compliance,
                   meaning this ratio is primarily effective for small context windows (< 25k tokens).
            vector_store: Vector store used for semantic duplicate matching.
            
        Note:
            The consolidation backlog is managed in-memory only in v0.2 and does not persist across restarts.
        """
        self._store = belief_store
        self.llm = llm_callable
        self._vector_store = vector_store
        
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
        """Adds a turn to the consolidation backlog and auto-triggers consolidation when full."""
        self._backlog.append(turn)
        content = turn.get("content", "")
        # Standard approximation: 1 token = 4 characters
        estimated_tokens = len(content) // 4
        self._backlog_tokens += estimated_tokens
        
        if self._backlog_tokens >= self.backlog_threshold_tokens:
            logger.info(f"Consolidation backlog tokens ({self._backlog_tokens}) exceeded threshold ({self.backlog_threshold_tokens}). Triggering consolidation pass...")
            self.run_consolidation_pass(self._backlog)
            self.clear_backlog()

    def clear_backlog(self):
        """Clears the backlog buffer."""
        self._backlog = []
        self._backlog_tokens = 0

    def run_consolidation_pass(self, conversation_turns: List[Dict[str, Any]]):
        """Analyze a batch of recent conversation turns and extract new beliefs."""
        if not conversation_turns:
            return

        serialized = []
        for turn in conversation_turns:
            role = str(turn.get("role", "unknown")).upper()
            content = str(turn.get("content", ""))
            serialized.append(f"[{role}]: {content}")
            
        text_to_analyze = "\n".join(serialized)
        
        prompt = f"""You are a cognitive consolidator. Extract new beliefs from this conversation log.
Look for:
1. Premises: Foundational truths, axioms, self-observations (e.g. "I am an AI", "User lives in Seattle").
2. Propositions: Learned facts, conditional rules (e.g. "If X happens, Y occurs").
3. Preferences: Values, likes, behavioral norms (e.g. "User prefers Python over Java").

Conversation Log:
{text_to_analyze}

Output as a JSON array of objects with these exact keys:
[
  {{
    "category": "premises" | "propositions" | "preferences",
    "content": "The belief statement text",
    "confidence": 0.5 to 1.0 (how certain is this based on the text),
    "source": "The specific quote or event that formed this belief"
  }}
]
Return ONLY valid JSON. No markdown formatting or backticks.
"""

        try:
            response = self.llm(prompt).strip()
            
            # Clean up markdown if the LLM leaked it despite instructions
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
                
            beliefs_data = json.loads(response.strip())
            
            for bd in beliefs_data:
                category = bd.get("category", "premises")
                if category not in ["premises", "propositions", "preferences", "people", "skills", "desires", "concepts"]:
                    category = "premises"
                    
                content = bd.get("content")
                if not content:
                    continue
                    
                content_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
                belief_id = f"bel_{content_digest}"
                
                # Merge semantic duplicates or add new beliefs
                self._store.merge_or_add_belief(
                    category=category,
                    belief_id=belief_id,
                    content=content,
                    confidence=bd.get("confidence", 0.7),
                    source=bd.get("source", "consolidation_pass"),
                    vector_store=self._vector_store
                )
                logger.info(f"Consolidated belief: [{category}] {content}")
                
        except Exception as e:
            logger.error(f"Failed to consolidate beliefs: {e}")

    def run_nightly_decay(self):
        """Triggers the belief decay pass on the belief store.
        
        This recalculates relevance and decays confidence of unverified beliefs.
        """
        stats = self._store.decay_all_beliefs()
        logger.info(f"Nightly decay complete: {stats}")
        return stats
