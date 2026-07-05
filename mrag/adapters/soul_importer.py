import re
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from mrag.memory.belief_store import BeliefStore, BELIEF_CATEGORIES

logger = logging.getLogger("mrag.adapters.soul_importer")

# Mapping of keywords to categories
HEADING_RULES: List[Tuple[Tuple[str, ...], str]] = [
    (("history", "journal", "log", "diary", "session", "conversation",
      "timeline", "changelog", "memories", "memory", "episode"), "premises"), # map episodes/memories to premises/logs
    (("identity", "soul", "who i am", "about me", "self", "persona",
      "origin", "backstory", "biography"), "premises"),
    (("value", "principle", "belief", "ethos", "tone", "voice", "style",
      "personality", "temperament", "manner", "guardrail"), "preferences"),
    (("skill", "how to", "how-to", "workflow", "procedure", "playbook",
      "recipe", "capability", "command", "tool", "ability", "routine"), "skills"),
    (("goal", "mission", "objective", "aspiration", "purpose", "drive",
      "vision", "want", "ambition", "roadmap"), "desires"),
    (("concept", "glossary", "definition", "define", "term", "ontology",
      "vocabulary"), "concepts"),
    (("person", "people", "contact", "relationship", "team", "user"), "people"),
    (("fact", "knowledge", "rule", "policy", "guideline", "note", "lesson",
      "learning", "reference", "context", "instruction"), "propositions"),
]

DEFAULT_CATEGORY = "propositions"

FRAMEWORK_HINTS: List[Tuple[str, str]] = [
    ("claude.md", "claudecode"),
    (".claude", "claudecode"),
    ("agents.md", "codex"),
    ("soul", "hermes"),
    ("identity", "hermes"),
    (".hermes", "hermes"),
    ("hermes", "hermes"),
    ("openclau", "openclaw"),
    ("pi_agent", "pi_agent"),
    ("piagent", "pi_agent"),
]

SOURCE_EXTS = {".md", ".markdown", ".txt", ".json", ".yaml", ".yml", ".mdx"}


class ImportedItem:
    def __init__(
        self,
        category: str,
        content: str,
        confidence: float,
        source_file: str,
        framework: str,
        heading: str = ""
    ):
        self.category = category
        self.content = content
        self.confidence = confidence
        self.source_file = source_file
        self.framework = framework
        self.heading = heading


def detect_framework(path: Path) -> str:
    """Detect agent framework from filename or folder path."""
    name = path.name.lower()
    parent = path.parent.name.lower()
    
    for pattern, framework in FRAMEWORK_HINTS:
        if pattern in name or pattern in parent:
            return framework
            
    # Default fallback
    return "generic"


def classify_heading(heading: str, framework: str) -> str:
    """Classify section heading to standard belief category."""
    h = heading.lower()
    # Split heading into words
    words = set(re.findall(r"\b\w+\b", h))
    for keywords, category in HEADING_RULES:
        # Check if any keyword matches as a full word in the heading
        if any(k in words for k in keywords):
            return category
            
    # Framework fallbacks
    if framework == "hermes" and not heading:
        return "premises"
    return DEFAULT_CATEGORY


def _split_markdown_sections(text: str) -> List[Tuple[str, str]]:
    """Split Markdown into (heading, body) sections by ATX headings."""
    sections = []
    current_heading = ""
    buf = []

    def flush():
        body = "\n".join(buf).strip()
        if body or current_heading:
            sections.append((current_heading, body))

    for line in text.splitlines():
        m = re.match(r"^\s{0,3}(#{1,6})\s+(.*)$", line)
        if m:
            flush()
            current_heading = m.group(2).strip().strip("#").strip()
            buf = []
        else:
            buf.append(line)
    flush()
    return [s for s in sections if s[1] or s[0]]


def _atomic_statements(body: str) -> List[str]:
    """Break section body into atomic statements (bullets / sentences)."""
    statements = []
    # Collapse code blocks
    body = re.sub(r"```.*?```", lambda m: m.group(0).replace("\n", " "), body, flags=re.DOTALL)

    for raw in body.splitlines():
        line = raw.strip()
        if not line or line in {"---", "***"}:
            continue
        # Strip bullet/list indicators
        line = re.sub(r"^([-*+]|\d+[.)]|>)\s+", "", line).strip()
        line = re.sub(r"^\*\*(.+?)\*\*:?\s*", r"\1: ", line) # bold labels
        if len(line) <= 3:
            continue
            
        if len(line) > 220 and "." in line:
            for sent in re.split(r"(?<=[.!?])\s+", line):
                sent = sent.strip()
                if len(sent) > 3:
                    statements.append(sent)
        else:
            statements.append(line)
    return statements


def parse_structured(path: Path, framework: str) -> List[ImportedItem]:
    """Parse JSON/YAML files into items."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Could not read structured file {path}: {e}")
        return []

    data = None
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(raw)
        except Exception as e:
            logger.warning(f"Bad JSON in {path} — treating as text: {e}")
            return parse_markdown(path, framework, raw_override=raw)
    else: # yaml
        try:
            import yaml
            data = yaml.safe_load(raw)
        except Exception:
            # Fallback to markdown text parser if yaml is not installed or invalid
            return parse_markdown(path, framework, raw_override=raw)

    items = []
    
    def add(content: str, category: str, heading: str = "", conf: float = 0.8):
        content = str(content).strip()
        if len(content) > 3:
            if category not in BELIEF_CATEGORIES:
                category = DEFAULT_CATEGORY
            items.append(ImportedItem(
                category=category,
                content=content,
                confidence=conf,
                source_file=str(path),
                framework=framework,
                heading=heading
            ))

    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                content = entry.get("content") or entry.get("text") or entry.get("belief")
                cat = entry.get("category")
                if content:
                    category = cat if cat in BELIEF_CATEGORIES else classify_heading(str(cat or ""), framework)
                    add(content, category, str(cat or ""), float(entry.get("confidence", 0.8)))
            elif isinstance(entry, str):
                add(entry, DEFAULT_CATEGORY)
    elif isinstance(data, dict):
        for key, value in data.items():
            category = classify_heading(str(key), framework)
            if isinstance(value, (list, tuple)):
                for v in value:
                    add(str(v), category, str(key))
            elif isinstance(value, (str, int, float, bool)):
                add(f"{key}: {value}" if not str(key).isdigit() else str(value), category, str(key))
                
    return items


def parse_markdown(path: Path, framework: str, raw_override: Optional[str] = None) -> List[ImportedItem]:
    """Parse markdown / text files into items."""
    if raw_override is not None:
        text = raw_override
    else:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Could not read markdown {path}: {e}")
            return []

    items = []
    for heading, body in _split_markdown_sections(text):
        category = classify_heading(heading, framework)
        for statement in _atomic_statements(body):
            items.append(ImportedItem(
                category=category,
                content=statement,
                confidence=0.8,
                source_file=str(path),
                framework=framework,
                heading=heading
            ))
    return items


def parse_file(path: Path) -> List[ImportedItem]:
    """Auto-detects format and parses file into items."""
    framework = detect_framework(path)
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        return parse_structured(path, framework)
    return parse_markdown(path, framework)


def import_agent_soul(directory_path: str, belief_store: BeliefStore) -> int:
    """Scans and imports files from a directory into the BeliefStore.
    
    Supports importing existing profiles from Hermes, OpenClaw, ClaudeCode, and Pi Agents.
    """
    path = Path(directory_path)
    if not path.exists() or not path.is_dir():
        logger.warning(f"Soul directory path does not exist: {directory_path}")
        return 0

    imported_count = 0
    # Enable cache loading if not loaded
    if not belief_store._cache_loaded:
        belief_store.load_into_cache()

    for file_path in path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SOURCE_EXTS:
            try:
                items = parse_file(file_path)
                for item in items:
                    content_digest = hashlib.sha256(item.content.encode("utf-8")).hexdigest()[:12]
                    belief_id = f"bel_{content_digest}"
                    
                    added = belief_store.add_belief(
                        category=item.category,
                        belief_id=belief_id,
                        content=item.content,
                        confidence=item.confidence,
                        source=f"imported_{item.framework}",
                        stability_index=0.7,
                        source_file=item.source_file,
                        framework=item.framework,
                        heading=item.heading
                    )
                    if added:
                        imported_count += 1
            except Exception as e:
                logger.error(f"Failed to import file {file_path}: {e}")

    # Batch write updated categories to disk
    if imported_count > 0:
        for category in BELIEF_CATEGORIES:
            category_beliefs = [
                b for b in belief_store._beliefs_cache.values()
                if b.get("_category") == category
            ]
            belief_store._write_category(category, category_beliefs)
            
    logger.info(f"Successfully imported {imported_count} beliefs from soul preset directory.")
    return imported_count
