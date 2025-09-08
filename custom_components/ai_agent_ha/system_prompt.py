"""System prompts for AI Agent HA."""

import os
from pathlib import Path

def _load_system_prompt() -> str:
    """Load system prompt from SYSTEM_PROMPT.md file."""
    current_dir = Path(__file__).parent
    prompt_file = current_dir / "SYSTEM_PROMPT.md"
    
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            content = f.read()
        # Remove markdown headers and return clean content
        return content.replace("# AI Agent HA System Prompt\n\n", "")
    except FileNotFoundError:
        # Fallback if file doesn't exist
        return "You are an AI assistant integrated with Home Assistant."

# Load the system prompt content
_PROMPT_CONTENT = _load_system_prompt()

# === SYSTEM PROMPT CONSTRUCTION ===
SYSTEM_PROMPT = {
    "role": "system",
    "content": _PROMPT_CONTENT,
}

# For now, SYSTEM_PROMPT_LOCAL is identical to SYSTEM_PROMPT
# This can be customized later for local-specific behavior
SYSTEM_PROMPT_LOCAL = SYSTEM_PROMPT.copy()