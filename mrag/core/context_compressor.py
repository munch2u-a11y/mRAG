import os
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from mrag.core.token_counting import count_text_tokens

logger = logging.getLogger("mrag.core.context_compressor")

def resolve_context_limit(
    context_token_limit: Optional[int] = None,
    model_name: Optional[str] = None,
    env_var: str = "MRAG_CONTEXT_LIMIT"
) -> int:
    """Resolves context window token limit for standard models or env configs.

    Priority:
    1. Explicit context_token_limit parameter
    2. Environment variable override (MRAG_CONTEXT_LIMIT)
    3. Model name auto-detection (best-effort fallback)
    """
    if context_token_limit is not None:
        return context_token_limit

    env_limit = os.environ.get(env_var)
    if env_limit:
        try:
            return int(env_limit)
        except ValueError:
            pass

    if not model_name:
        raise ValueError(
            "\n[MicroRAG Configuration Error]: LLM Context Window limit is not set.\n"
            "You must specify the context limit for your active model to enable dynamic memory compression.\n"
            "Options:\n"
            "1. Pass context_token_limit directly to the constructor.\n"
            "2. Set the MRAG_CONTEXT_LIMIT environment variable.\n"
            "3. Provide a standard model_name string (e.g. 'gpt-4o', 'claude-3-5-sonnet')."
        )

    m = model_name.lower().strip()
    if "gemini" in m:
        return 1000000  # Default fallback for Gemini (approximate)
    elif "claude" in m:
        return 200000   # Default fallback for Claude (approximate)
    elif "gpt-4" in m or "gpt-3.5" in m:
        return 128000   # Default fallback for GPT-4 (approximate)
    elif "llama" in m or "mistral" in m or "phi" in m or "gemma" in m or "qwen" in m:
        return 8192     # Default fallback for local models (approximate)
    else:
        raise ValueError(
            f"\n[MicroRAG Configuration Error]: Unknown model context limit for '{model_name}'.\n"
            "Please configure the context window limit manually by passing context_token_limit "
            "or setting the MRAG_CONTEXT_LIMIT environment variable."
        )


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
        context_token_limit: Optional[int] = None,
        threshold_percent: float = 0.65,
        protect_first_n: int = 2,
        summary_target_ratio: float = 0.20,
    ):
        """
        Args:
            llm_callable: A function that takes a prompt string and returns the LLM's string response.
            context_token_limit: Optional. The maximum tokens for the context window. Resolved from env/model name if omitted.
            threshold_percent: Compress when tokens exceed this percent of the limit.
            protect_first_n: Keep the first N messages (usually system prompt + first user message).
            summary_target_ratio: How much of the context window to use for the tail (recent context).
        """
        self.llm = llm_callable
        model_name = os.environ.get("MRAG_MODEL_NAME")
        self.context_token_limit = resolve_context_limit(context_token_limit, model_name)
        self.threshold_tokens = int(self.context_token_limit * threshold_percent)
        self.protect_first_n = protect_first_n
        self.tail_token_budget = int(self.threshold_tokens * summary_target_ratio)
        
        self._previous_summary: Optional[str] = None

    def should_compress(self, current_tokens: int) -> bool:
        return current_tokens >= self.threshold_tokens

    def _estimate_tokens(self, text: str) -> int:
        """Best-effort token count backed by tiktoken when available."""
        return count_text_tokens(text)

    def compress(self, messages: List[Dict[str, Any]], is_end_of_session: bool = False) -> List[Dict[str, Any]]:
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
        if not is_end_of_session and not self.should_compress(total_tokens):
            return messages

        # 1. Find tail boundary (save 10 turns up to 10% of total tokens)
        tail_start = n_messages
        accumulated_tail = 0
        limit_tokens = int(total_tokens * 0.10)
        
        for i in range(n_messages - 1, self.protect_first_n - 1, -1):
            msg_tokens = self._estimate_tokens(str(messages[i].get("content", "")))
            if (n_messages - i) > 10:
                break
            if accumulated_tail + msg_tokens > limit_tokens and (n_messages - i) > 1:
                break
            accumulated_tail += msg_tokens
            tail_start = i

        tail_start = max(tail_start, self.protect_first_n + 1)
        
        if self.protect_first_n >= tail_start:
            return messages

        # 2. Summarize middle turns. A previous compression summary sitting in
        # the middle is adopted as the "previous recollection" instead of being
        # re-serialized into the prompt (which would duplicate its content).
        # If is_end_of_session, summarize every turn from protect_first_n to the end.
        turns_to_summarize = []
        target_end = n_messages if is_end_of_session else tail_start
        for msg in messages[self.protect_first_n:target_end]:
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

        # 4. If end of session, add a divider indicating time passed since compression
        if is_end_of_session and n_messages > 0:
            import re
            from datetime import datetime
            
            # Extract timestamp from last turn of the compressed session
            last_turn_content = str(messages[-1].get("content", ""))
            
            def _parse_raw_date(raw: str) -> Optional[datetime]:
                for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%B %d, %Y %I:%M %p", "%B %d, %Y", "%I:%M %p on %d %B, %Y"):
                    try:
                        return datetime.strptime(raw, fmt)
                    except ValueError:
                        continue
                return None

            last_time = None
            m1 = re.search(r'Date:\s*([^\n]+)', last_turn_content)
            if m1:
                last_time = _parse_raw_date(m1.group(1).strip())
            if not last_time:
                m2 = re.search(r'^\[([\d\-: ]+)\]', last_turn_content)
                if m2:
                    last_time = _parse_raw_date(m2.group(1).strip())

            now = datetime.now()
            last_timestamp = last_time.strftime("%Y-%m-%d %H:%M") if last_time else "Unknown"
            new_timestamp = now.strftime("%Y-%m-%d %H:%M")
            
            time_diff_str = "Unknown"
            if last_time:
                elapsed = now - last_time
                if elapsed.days > 0:
                    time_diff_str = f"{elapsed.days} days, {elapsed.seconds // 3600} hours"
                else:
                    time_diff_str = f"{elapsed.seconds // 3600} hours, {(elapsed.seconds % 3600) // 60} minutes"

            divider_msg = {
                "role": "system",
                "content": (
                    f"--------------------------------------------------\n"
                    f"[Session Ended at: {last_timestamp}]\n"
                    f"[Time Elapsed since compression: {time_diff_str}]\n"
                    f"[New Session Started at: {new_timestamp}]\n"
                    f"--------------------------------------------------"
                ),
                "is_session_divider": True
            }
            compressed.append(divider_msg)

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
- Explicitly preserve and include all dates, timelines, times, schedules, and specific temporal details mentioned in the turns.
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
