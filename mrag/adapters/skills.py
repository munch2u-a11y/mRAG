import os
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

# Optional YAML support
try:
    import yaml
except ImportError:
    yaml = None

from mrag.memory.belief_store import BeliefStore

logger = logging.getLogger("mrag.adapters.skills")


def import_openai_tools(tools: List[Dict[str, Any]], belief_store: BeliefStore) -> int:
    """Import skills from an OpenAI-format tools schema.

    Expects list of dicts:
    [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather info.",
                "parameters": {...}
            }
        }
    ]
    """
    imported_count = 0
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        name = func.get("name")
        desc = func.get("description", "")
        if not name:
            continue

        belief_id = f"tool_{name}"
        content = f"Tool '{name}': {desc}"
        metadata = {"tool_name": name, "schema": func}

        added = belief_store.add_belief(
            category="skills",
            belief_id=belief_id,
            content=content,
            confidence=1.0,
            source="openai_tools_import",
            metadata=metadata
        )
        if added:
            imported_count += 1

    return imported_count


def import_mcp_tools(mcp_response: Dict[str, Any], belief_store: BeliefStore) -> int:
    """Import skills from Model Context Protocol (MCP) tool definitions.

    Expects MCP get tools response:
    {
        "tools": [
            {
                "name": "calculate_sum",
                "description": "Sums two numbers",
                "inputSchema": {...}
            }
        ]
    }
    """
    imported_count = 0
    if isinstance(mcp_response, list):
        # Fallback if the list of tools is passed directly
        tools = mcp_response
    elif isinstance(mcp_response, dict):
        tools = mcp_response.get("tools", [])
    else:
        return 0
    if not isinstance(tools, list):
        return 0

    for tool in tools:
        name = tool.get("name")
        desc = tool.get("description", "")
        if not name:
            continue

        belief_id = f"mcp_{name}"
        content = f"MCP Tool '{name}': {desc}"
        metadata = {"tool_name": name, "schema": tool}

        added = belief_store.add_belief(
            category="skills",
            belief_id=belief_id,
            content=content,
            confidence=1.0,
            source="mcp_tools_import",
            metadata=metadata
        )
        if added:
            imported_count += 1

    return imported_count


def _default_file_parser(filepath: str, content: str) -> Tuple[Optional[str], Optional[str]]:
    """Tries to extract name/description from raw JSON or YAML content."""
    data = None
    if filepath.endswith(".json"):
        try:
            data = json.loads(content)
        except Exception:
            pass
    elif filepath.endswith((".yaml", ".yml")) and yaml:
        try:
            data = yaml.safe_load(content)
        except Exception:
            pass

    if isinstance(data, dict):
        # Look for typical fields used by Hermes, OpenClaw, or custom agents
        name = data.get("name") or data.get("id") or data.get("tool_name")
        desc = data.get("description") or data.get("desc") or data.get("info")
        if name and desc:
            return str(name), str(desc)

    return None, None


def import_from_directory(
    directory_path: str,
    belief_store: BeliefStore,
    file_extensions: Tuple[str, ...] = (".json", ".yaml", ".yml"),
    custom_parser: Optional[Callable[[str, str], Tuple[Optional[str], Optional[str]]]] = None
) -> int:
    """Scans a directory of skill files and imports them.

    Args:
        directory_path: Absolute or relative path to directory.
        belief_store: The destination BeliefStore.
        file_extensions: Tuple of file extensions to search for.
        custom_parser: A function taking (filepath, raw_content) and returning
                       (name, description) tuple. Overrides default JSON/YAML parser.
    """
    if not os.path.isdir(directory_path):
        logger.warning(f"Directory not found: {directory_path}")
        return 0

    parser = custom_parser or _default_file_parser
    imported_count = 0

    for root, _, files in os.walk(directory_path):
        for file in files:
            if not file.endswith(file_extensions):
                continue

            filepath = os.path.join(root, file)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                name, desc = parser(filepath, content)
                if name and desc:
                    belief_id = f"custom_skill_{name}"
                    content_str = f"Skill '{name}': {desc}"
                    metadata = {"filepath": filepath}

                    added = belief_store.add_belief(
                        category="skills",
                        belief_id=belief_id,
                        content=content_str,
                        confidence=1.0,
                        source="directory_skills_import",
                        metadata=metadata
                    )
                    if added:
                        imported_count += 1
            except Exception as e:
                logger.error(f"Failed to import skill file {filepath}: {e}")

    return imported_count
