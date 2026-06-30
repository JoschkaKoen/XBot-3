"""
GrokImagineClient — image generation via the xAI Grok Imagine API.

Used by nodes/generate_image.py when IMAGE_PROVIDER=grok.

Model + resolution are config-driven (set in settings.env — NO code change to switch):
  GROK_IMAGE_MODEL = grok-imagine-image          → standard, $0.02 / 1K image  (default)
                   = grok-imagine-image-quality  → higher fidelity, $0.05 / 1K image
  GROK_IMAGE_RESOLUTION = 1k (default; 1280×720 @ 16:9) | 2k
"""

import base64
import logging
import os
from datetime import datetime
from typing import List

import requests

import config
from config import IMAGES_DIR
from utils.errors import FatalProviderError
from utils.retry import with_retry

logger = logging.getLogger("xbot.grok_imagine")

_DEFAULT_GROK_IMAGE_MODEL = "grok-imagine-image"
_XAI_BASE_URL = "https://api.x.ai/v1"


class GrokImagineClient:
    """Image generation via the xAI Grok Imagine API."""

    def __init__(self):
        self._api_key = os.getenv("XAI_API_KEY", "")
        if not self._api_key:
            raise ValueError("❌ XAI_API_KEY not found in .env — required for IMAGE_PROVIDER=grok")
        os.makedirs(IMAGES_DIR, exist_ok=True)

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    @with_retry(max_attempts=3, base_delay=2.0, label="grok_download")
    def _download_image(self, img_url: str, idx: int) -> str:
        from utils.io import stream_to_file
        resp = requests.get(img_url, stream=True, timeout=60)
        resp.raise_for_status()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"grok_{ts}_{idx}.png"
        path = os.path.join(IMAGES_DIR, filename)
        stream_to_file(resp, path)   # raises on truncation → retried by @with_retry
        logger.debug("Downloaded Grok image → %s", path)
        return path

    def generate(self, prompt: str, n: int = 1, aspect_ratio: str = "16:9") -> List[str]:
        # Model + resolution are config-driven (settings.env → GROK_IMAGE_MODEL /
        # GROK_IMAGE_RESOLUTION). Read at call time so reload_settings is honoured.
        model = (getattr(config, "GROK_IMAGE_MODEL", "") or _DEFAULT_GROK_IMAGE_MODEL).strip()
        resolution = (getattr(config, "GROK_IMAGE_RESOLUTION", "") or "1k").strip()
        payload = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "response_format": "url",
        }
        print(f"\r  ⏳  Requesting {n} {resolution} image(s) from Grok Imagine ({model}) …", flush=True)
        resp = requests.post(
            f"{_XAI_BASE_URL}/images/generations",
            headers=self._headers(),
            json=payload,
            timeout=120,
        )
        if resp.status_code in (401, 402, 403):
            raise FatalProviderError(
                f"Grok Imagine returned HTTP {resp.status_code} — check your XAI_API_KEY "
                f"and account credits. Bot stopping to avoid further charges."
            )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("data", [])
        if not items:
            raise RuntimeError(f"Grok Imagine returned no images: {data}")
        print()
        # The API reports exact spend (cost_in_usd_ticks, 1 tick = $1e-10) — log it so
        # per-cycle image cost is visible in the terminal/logs.
        ticks = (data.get("usage") or {}).get("cost_in_usd_ticks")
        cost_note = f" — cost ${ticks * 1e-10:.4f}" if isinstance(ticks, (int, float)) else ""
        logger.info("Grok Imagine (%s, %s) returned %d image(s)%s.", model, resolution, len(items), cost_note)
        paths = []
        for i, item in enumerate(items):
            url = item.get("url") or item.get("b64_json")
            if not url:
                logger.warning("Grok Imagine item %d has no URL — skipping.", i + 1)
                continue
            if url.startswith("data:") or not url.startswith("http"):
                # base64 fallback
                b64 = url.split(",", 1)[-1] if "," in url else url
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"grok_{ts}_{i + 1}.png"
                path = os.path.join(IMAGES_DIR, filename)
                with open(path, "wb") as f:
                    f.write(base64.b64decode(b64))
                logger.debug("Saved Grok b64 image → %s", path)
                paths.append(path)
            else:
                paths.append(self._download_image(url, i + 1))
        if not paths:
            raise RuntimeError("Grok Imagine: all returned items had no usable image data.")
        return paths
