"""
Switchable AI provider.

Set AI_PROVIDER in .env to "grok" or "scaleway".
Both providers expose the same function signature:

    get_ai_response(user_message, system_prompt, max_tokens, temperature) -> str

To add a new provider later:
1. Create services/my_provider_ai.py with a get_my_provider_response() function.
2. Add an elif branch below.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("german_bot.ai_client")

_PROVIDER = os.getenv("AI_PROVIDER", "grok").lower().strip()

if _PROVIDER == "grok":
    from .grok_ai import get_grok_response as get_ai_response
    logger.info("AI Provider: Grok (xAI) — model grok-4-1-fast-non-reasoning")

elif _PROVIDER == "scaleway":
    from .scaleway_ai import get_scaleway_response as get_ai_response
    logger.info("AI Provider: Scaleway — model llama-3.3-70b-instruct")

else:
    raise ValueError(
        f"❌ Unknown AI_PROVIDER '{_PROVIDER}' in .env. "
        "Supported values: 'grok', 'scaleway'"
    )

__all__ = ["get_ai_response"]
