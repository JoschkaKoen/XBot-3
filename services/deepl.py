import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("german_bot.deepl")

_ENDPOINT = "https://api-free.deepl.com/v2/translate"


def _auth_key() -> str:
    key = os.getenv("DEEPL_AUTH_KEY")
    if not key:
        raise ValueError("❌ DEEPL_AUTH_KEY not found in .env!")
    return key


def translate_with_deepl(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text using DeepL free API.

    Args:
        text:        text to translate
        source_lang: ISO code, e.g. "DE"
        target_lang: ISO code, e.g. "EN"

    Returns:
        Translated string.
    """
    headers = {
        "Authorization": f"DeepL-Auth-Key {_auth_key()}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = (
        f"text={requests.utils.quote(text)}"
        f"&source_lang={source_lang}"
        f"&target_lang={target_lang}"
    ).encode("utf-8")

    logger.debug("DeepL translate [%s→%s]: %.60s …", source_lang, target_lang, text)
    resp = requests.post(_ENDPOINT, headers=headers, data=data, timeout=30)
    resp.raise_for_status()
    translation: str = resp.json()["translations"][0]["text"]
    logger.debug("DeepL result: %.60s …", translation)
    return translation
