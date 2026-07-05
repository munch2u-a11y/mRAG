import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("mrag.core.context_compressor")

SUMMARY_PREFIX = (
    "[COGNITIVE CONTINUITY] The following is a seamless continuation of your "
    "recent thoughts, actions, and experiences, compacted for memory efficiency. "
    "You have already experienced everything described here in exactly this order. "
    "Resume your train of thought naturally from the final events described. "
    "Respond ONLY to events that occur AFTER this narrative."
)

class ContextCompressor:
    """Rolling context compressor for Micro-RAG.

    Replaces hard context resets with rolling summarization that preserves
    conversational continuity and turn identifiers for belief extraction.
    Agnostic: expects messages as dicts with 'role' and 'content'.
    """

    def __init__(
        self,
        llm_callable: Callable[[str], str],
        context_token_limit: int = 128000,
        threshold_percent: float = 0.65,
        protect_first_n: int = 2,
        summary_target_ratio: float = 0.20,
    ):
        """
        Args:
            llm_callable: A function that takes a prompt string and returns the LLM's string response.
            context_token_limit: The maximum tokens for the context window.
            threshold_percent: Compress when tokens exceed this percent of the limit.
            protect_first_n: Keep the first N messages (usually system prompt + first user message).
            summary_target_ratio: How much of the context window to use for the tail (recent context).
        """
        self.llm = llm_callable
        self.context_token_limit = context_token_limit
        self.threshold_tokens = int(context_token_limit * threshold_percent)
        self.protect_first_n = protect_first_n
        self.tail_token_budget = int(self.threshold_tokens * summary_target_ratio)
        
        self._previous_summary: Optional[str] = None

    def should_compress(self, current_tokens: int) -> bool:
        return current_tokens >= self.threshold_tokens

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (1 token ~= 4 chars)."""
        return len(text) // 4

    def compress(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compress conversation messages by summarizing middle turns.
        
        Expects messages like: {"role": "user", "content": "Hello"}
        """
        n_messages = len(messages)
        min_for_compress = self.protect_first_n + 4
        if n_messages <= min_for_compress:
            return messages

        total_tokens = sum(
            self._estimate_tokens(str(message.get("content", "")))
            for message in messages
        )
        if not self.should_compress(total_tokens):
            return messages

        # 1. Find tail boundary
        tail_start = n_messages
        accumulated_tail = 0
        for i in range(n_messages - 1, self.protect_first_n - 1, -1):
            msg_tokens = self._estimate_tokens(str(messages[i].get("content", "")))
            if accumulated_tail + msg_tokens > self.tail_token_budget and (n_messages - i) >= 3:
                break
            accumulated_tail += msg_tokens
            tail_start = i

        tail_start = max(tail_start, self.protect_first_n + 1)
        
        if self.protect_first_n >= tail_start:
            return messages

        # 2. Summarize middle turns. A previous compression summary sitting in
        # the middle is adopted as the "previous recollection" instead of being
        # re-serialized into the prompt (which would duplicate its content).
        turns_to_summarize = []
        for msg in messages[self.protect_first_n:tail_start]:
            if msg.get("is_compressed_summary"):
                content = str(msg.get("content", ""))
                if content.startswith(SUMMARY_PREFIX):
                    content = content[len(SUMMARY_PREFIX):].strip()
                self._previous_summary = content
            else:
                turns_to_summarize.append(msg)

        if turns_to_summarize:
            summary = self._generate_summary(turns_to_summarize)
        elif self._previous_summary:
            summary = f"{SUMMARY_PREFIX}\n\n{self._previous_summary}"
        else:
            summary = ""

        # 3. Assemble compressed history
        compressed = []
        
        # Head
        for i in range(self.protect_first_n):
            compressed.append(messages[i])
            
        # Summary
        if summary:
            compressed.append({
                "role": "user",
                "content": summary,
                "is_compressed_summary": True # metadata tag
            })
            
        # Tail
        for i in range(tail_start, n_messages):
            compressed.append(messages[i])

        return compressed

    def _generate_summary(self, turns: List[Dict[str, Any]]) -> str:
        # Serialize turns with explicit turn identifiers
        parts = []
        for msg in turns:
            role = str(msg.get("role", "unknown")).upper()
            content = str(msg.get("content", ""))
            if len(content) > 3000:
                content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
            parts.append(f"[{role}]: {content}")
            
        serialized_turns = "\n\n".join(parts)
        
        template = """Compress these conversation turns into natural first-person recollection \
— the way someone would think back on what just happened.
- Use direct quotes for what people said.
- Include responses and thoughts naturally.
- Maintain strict chronological order with explicit [USER] and [MODEL] turn identifiers where relevant so belief extractors can parse it.
- Preserve specific facts, names, and unresolved threads exactly.
- Do not add commentary, analysis, or anything that wasn't in the original turns.
Only output the recollection, no preamble."""

        if self._previous_summary:
            prompt = (
                f"You are UPDATING a previous recollection. Preserve existing content, add new events.\n\n"
                f"PREVIOUS RECOLLECTION:\n{self._previous_summary}\n\n"
                f"NEW TURNS TO INCORPORATE:\n{serialized_turns}\n\n"
                f"Continue with:\n{template}"
            )
        else:
            prompt = (
                f"TURNS TO COMPRESS:\n{serialized_turns}\n\n"
                f"{template}"
            )
            
        try:
            summary_text = self.llm(prompt).strip()
            self._previous_summary = summary_text
            return f"{SUMMARY_PREFIX}\n\n{summary_text}"
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return f"{SUMMARY_PREFIX}\n\n[Summary generation unavailable due to error: {e}]"
