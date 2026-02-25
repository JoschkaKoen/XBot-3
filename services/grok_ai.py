import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("german_bot.grok")

_client = None

_MODEL_FAST      = "grok-4-1-fast-non-reasoning"
_MODEL_REASONING = "grok-4-1-fast"       # fast reasoning variant
_MODEL_FLAGSHIP  = "grok-4"              # flagship model — best language quality


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("❌ XAI_API_KEY not found in .env!")
        _client = OpenAI(base_url="https://api.x.ai/v1", api_key=api_key)
    return _client


def _call(model: str, user_message: str, system_prompt: str, max_tokens: int, temperature: float) -> str:
    logger.debug("Grok request (%s): %.80s …", model, user_message)
    response = _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    result = response.choices[0].message.content
    logger.debug("Grok response: %.80s …", result)
    return result


def get_grok_response(
    user_message: str,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 1200,
    temperature: float = 0.7,
) -> str:
    return _call(_MODEL_FAST, user_message, system_prompt, max_tokens, temperature)


def get_grok_reasoning_response(
    user_message: str,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 1200,
    temperature: float = 0.7,
) -> str:
    return _call(_MODEL_REASONING, user_message, system_prompt, max_tokens, temperature)


def get_grok_flagship_response(
    user_message: str,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 1200,
    temperature: float = 0.7,
) -> str:
    return _call(_MODEL_FLAGSHIP, user_message, system_prompt, max_tokens, temperature)
