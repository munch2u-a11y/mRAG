import hashlib
import json
import logging
from typing import Any, Callable, Dict, List

from mrag.memory.belief_store import BeliefStore

logger = logging.getLogger("mrag.core.belief_consolidator")

class BeliefConsolidator:
    """Consolidates conversational logs into structured beliefs.
    
    Extracts premises, propositions, and preferences from interaction logs
    and saves them into the BeliefStore with temporal and structural metadata.
    """

    def __init__(self, belief_store: BeliefStore, llm_callable: Callable[[str], str]):
        self._store = belief_store
        self.llm = llm_callable

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
                
                # We could do a deduplication check here, or let the BeliefStore handle it.
                self._store.add_belief(
                    category=category,
                    belief_id=belief_id,
                    content=content,
                    confidence=bd.get("confidence", 0.7),
                    source=bd.get("source", "consolidation_pass")
                )
                logger.info(f"Consolidated new belief: [{category}] {content}")
                
        except Exception as e:
            logger.error(f"Failed to consolidate beliefs: {e}")

    def run_nightly_decay(self):
        """Triggers the belief decay pass on the belief store.
        
        This recalculates relevance and decays confidence of unverified beliefs.
        """
        stats = self._store.decay_all_beliefs()
        logger.info(f"Nightly decay complete: {stats}")
        return stats
