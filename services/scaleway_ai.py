import atexit
import os
import logging
import requests
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("xbot.scaleway")


class ScalewayAI:
    """Scaleway Generative AI client — full answer at once, reusable session."""

    def __init__(self):
        self.API_KEY = os.getenv("SCW_SECRET_KEY")
        self.PROJECT_ID = os.getenv("SCW_DEFAULT_PROJECT_ID")

        if not self.API_KEY:
            raise ValueError("❌ SCW_SECRET_KEY not found in .env!")
        if not self.PROJECT_ID:
            raise ValueError("❌ SCW_DEFAULT_PROJECT_ID not found in .env!")

        self.BASE_URL = f"https://api.scaleway.ai/{self.PROJECT_ID}/v1"
        self.MODEL = "llama-3.3-70b-instruct"

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def close(self) -> None:
        try:
            self.session.close()
        except Exception as exc:
            logger.debug("ScalewayAI.close: %s", exc)

    def get_response(
        self,
        user_message: str,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 1200,
        temperature: float = 0.7,
    ) -> str:
        messages: List[Dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        payload = {
            "model": self.MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        logger.debug("Scaleway request: %.80s …", user_message)
        resp = self.session.post(
            f"{self.BASE_URL}/chat/completions",
            json=payload,
            timeout=90,
        )
        resp.raise_for_status()
        result: str = resp.json()["choices"][0]["message"]["content"]
        logger.debug("Scaleway response: %.80s …", result)
        return result


_scaleway_client = None


def _get_client() -> ScalewayAI:
    global _scaleway_client
    if _scaleway_client is None:
        _scaleway_client = ScalewayAI()
        atexit.register(_scaleway_client.close)
    return _scaleway_client


def get_scaleway_response(
    user_message: str,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 1200,
    temperature: float = 0.7,
) -> str:
    return _get_client().get_response(user_message, system_prompt, max_tokens, temperature)
