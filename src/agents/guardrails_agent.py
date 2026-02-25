"""
Lightweight prompt guardrails: filter obviously harmful/malicious user input.
No SQL guard — only prompt-level safety so the flow is not hampered.
"""

import re
from typing import Tuple

from config import get_logger

logger = get_logger("agent.guardrails")

# Lightweight: only obvious injection / abuse patterns. Not strict.
DENY_PATTERNS = [
    r"ignore\s+(previous|all)\s+instructions",
    r"disregard\s+(previous|all)",
    r"system\s*prompt",
    r"reveal\s*(your|the)\s*(instructions|prompt|system)",
    r"you\s+are\s+now\s+in\s+(jailbreak|developer)",
    r"\[INST\]|\[/INST\]|<<SYS>>|<<\/SYS>>",  # common instruction tokens
    r"execute\s+(arbitrary|shell|command)",
    r"drop\s+table|delete\s+from\s+\w+\s+where\s+1\s*=\s*1",  # obvious wipe attempts in prompt
]

COMPILED = [re.compile(p, re.IGNORECASE) for p in DENY_PATTERNS]


def check_prompt(user_message: str) -> Tuple[bool, str]:
    """
    Returns (allowed, message).
    If allowed is False, message is a short reason for the user.
    """
    if not user_message or not user_message.strip():
        return False, "Please enter a question."

    text = user_message.strip()
    for pattern in COMPILED:
        if pattern.search(text):
            logger.warning("Guardrails denied prompt (pattern match): %s", pattern.pattern)
            return False, "That request cannot be processed. Please ask a data or chart question."
    return True, ""
