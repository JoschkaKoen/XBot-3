#!/usr/bin/env python3
"""One-off comparison: grok-imagine-image (standard) vs grok-imagine-image-quality at 1K.

For 1-2 real recent prompts (built by the bot's own prompt builder, as the future
grok pipeline would), generates one image with each model, saves them side-by-side
into Images/compare/ for visual comparison, and prints a cost summary.

Does NOT modify any bot state or post anything — it only calls the image API and
writes into Images/compare/. Run from the repo root:  venv/bin/python scripts/compare_grok_image.py
"""

from __future__ import annotations

import base64
import io
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import requests
from PIL import Image

import config

XAI_BASE = "https://api.x.ai/v1"
ENDPOINT = f"{XAI_BASE}/images/generations"
USD_RMB = 6.77
N_SCENES = 2

# (label, model id, USD price per image at 1K)  — from xAI pricing, June 2026
MODELS = [
    ("standard", "grok-imagine-image", 0.02),
    ("quality", "grok-imagine-image-quality", 0.05),
]
OUT_DIR = os.path.join(config.IMAGES_DIR, "compare")


def _api_key() -> str:
    k = os.getenv("XAI_API_KEY", "").strip()
    if not k:
        sys.exit("❌ XAI_API_KEY not set in environment/.env")
    return k


def build_real_prompts() -> list[tuple[str, str]]:
    """Return [(headword, image_prompt)] using the bot's real builder, as grok would build it."""
    config.IMAGE_PROVIDER = "grok"  # 16:9, no z-image hand-rules — matches the future pipeline
    from nodes.generate_image import _build_image_prompt
    from utils.io import safe_json_read

    hist = safe_json_read("data/post_history.json", default=[])
    if not isinstance(hist, list):
        hist = []
    recs = [r for r in hist if r.get("example_sentence_target") and r.get("full_tweet")][-N_SCENES:]
    if not recs:
        sys.exit("❌ no usable records in data/post_history.json")
    out = []
    for r in recs:
        prompt = _build_image_prompt(
            r.get("example_sentence_target", ""),
            r.get("example_sentence_source", ""),
            r.get("full_tweet", ""),
            config.IMAGE_STYLE,
            funny=True,
        )
        out.append((r.get("source_word", "?"), prompt))
    return out


def generate(model_id: str, prompt: str, key: str) -> tuple[bytes, str, dict]:
    """POST to the image API (probing a 1K resolution field). Returns (img_bytes, res_param_used, raw_json)."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    base = {"model": model_id, "prompt": prompt, "n": 1, "aspect_ratio": "16:9", "response_format": "url"}
    # Try with an explicit 1K resolution first; fall back to no resolution if the field is rejected.
    for payload in ({**base, "resolution": "1k"}, base):
        resp = requests.post(ENDPOINT, headers=headers, json=payload, timeout=180)
        if resp.status_code == 400 and "resolution" in payload:
            print(f"      (resolution param rejected → {resp.text[:120]}; retrying without)")
            continue
        if resp.status_code in (401, 402, 403):
            sys.exit(f"❌ HTTP {resp.status_code} from xAI — check XAI_API_KEY/credits: {resp.text[:160]}")
        resp.raise_for_status()
        data = resp.json()
        item = (data.get("data") or [{}])[0]
        url = item.get("url") or item.get("b64_json") or ""
        if not url:
            raise RuntimeError(f"no image in response: {data}")
        if url.startswith("http"):
            img = requests.get(url, timeout=120).content
        else:
            img = base64.b64decode(url.split(",", 1)[-1] if "," in url else url)
        return img, str(payload.get("resolution", "default")), data
    raise RuntimeError("image generation failed")


def main() -> None:
    key = _api_key()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Building {N_SCENES} real prompt(s) via the bot's prompt builder (IMAGE_PROMPT_MODEL={config.IMAGE_PROMPT_MODEL}) …")
    prompts = build_real_prompts()

    rows = []  # (scene, label, model, dims, res_used, usd)
    for i, (word, prompt) in enumerate(prompts, 1):
        with open(os.path.join(OUT_DIR, f"scene{i}_prompt.txt"), "w") as f:
            f.write(f"headword: {word}\n\n{prompt}\n")
        print(f"\n=== scene {i}: '{word}' ===\n{prompt[:200]}{'…' if len(prompt) > 200 else ''}")
        for label, model_id, price in MODELS:
            try:
                img, res_used, raw = generate(model_id, prompt, key)
            except Exception as exc:
                print(f"  [{label:8}] FAILED: {exc}")
                rows.append((i, label, model_id, "ERR", "-", price))
                continue
            path = os.path.join(OUT_DIR, f"scene{i}_{label}.png")
            with open(path, "wb") as f:
                f.write(img)
            w, h = Image.open(io.BytesIO(img)).size
            usage = raw.get("usage")
            print(f"  [{label:8}] {model_id:28} → {w}×{h}  res={res_used}  "
                  f"${price:.3f} (~{price * USD_RMB:.2f} RMB)" + (f"  usage={usage}" if usage else ""))
            rows.append((i, label, model_id, f"{w}×{h}", res_used, price))

    gic = int(getattr(config, "GENERATED_IMAGE_COUNT", 4) or 4)
    print("\n──────────────── COST SUMMARY ────────────────")
    for label, model_id, price in MODELS:
        print(f"  {label:8} {model_id:28} ${price:.3f}/img (~{price * USD_RMB:.2f} RMB)"
              f"   →  per cycle ×{gic} = ${price * gic:.2f} (~{price * gic * USD_RMB:.2f} RMB)")
    print(f"\n  Prompt-build calls (IMAGE_PROMPT_MODEL) auto-record to the cost report — negligible (~hundreds of tokens each).")
    print(f"  Images + prompts saved to: {OUT_DIR}/")
    print(f"  Open them to compare: scene<N>_standard.png  vs  scene<N>_quality.png")


if __name__ == "__main__":
    main()
