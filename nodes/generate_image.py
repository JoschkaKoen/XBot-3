"""
Node: generate_image

Generates a Midjourney prompt via LLM, submits to Midjourney via TTAPI,
polls until done, downloads all images, and picks the best one using
ImageReward-v1.0. Falls back to the first image if the model is unavailable.
"""

import os
import time
import logging
import requests
from typing import List
from datetime import datetime

from config import TT_API_KEY, IMAGES_DIR, FUNNY_MODE
from services.ai_client import get_ai_response
from services.image_ranker import pick_best_image
from utils.retry import retry_call, with_retry
from utils.ui import stage_banner, ok, info

logger = logging.getLogger("german_bot.generate_image")


# ── Midjourney client (copied from midjourney.py) ─────────────────────────────

class MidjourneyClient:
    """Internal Midjourney client via TTAPI."""

    BASE_URL = "https://api.ttapi.io/midjourney/v1"

    def __init__(self):
        if not TT_API_KEY:
            raise ValueError("❌ TT_API_KEY not found in .env!")
        self.HEADERS = {
            "TT-API-KEY": TT_API_KEY,
            "Content-Type": "application/json",
        }
        os.makedirs(IMAGES_DIR, exist_ok=True)

    @with_retry(max_attempts=4, base_delay=3.0, label="mj_submit")
    def _submit_imagine(self, prompt: str, mode: str = "fast", aspect_ratio: str = "16:9") -> str:
        prompt = prompt.strip()
        if "--ar" not in prompt.lower() and "--aspect" not in prompt.lower():
            prompt += f" --ar {aspect_ratio}"
        if "--style" not in prompt.lower():
            prompt += " --style raw"
        if "--s " not in prompt.lower() and "--stylize" not in prompt.lower():
            prompt += " --s 0"

        payload = {"prompt": prompt, "mode": mode}
        resp = requests.post(f"{self.BASE_URL}/imagine", headers=self.HEADERS, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "SUCCESS":
            raise RuntimeError(f"Midjourney submit failed: {data}")

        job_id = data["data"]["jobId"]
        logger.info("Midjourney job submitted: %s (ar=%s)", job_id, aspect_ratio)
        return job_id

    def _poll_job(self, job_id: str, timeout_sec: int = 360, interval: int = 1):
        url = f"{self.BASE_URL}/fetch"
        start = time.time()
        dots = 0
        while time.time() - start < timeout_sec:
            resp = requests.post(url, headers=self.HEADERS, json={"jobId": job_id}, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            status = result.get("status")
            logger.debug("Midjourney poll status: %s", status)

            if status == "SUCCESS":
                print()   # newline after dots
                logger.info("Midjourney image generation complete.")
                return result.get("data", result)
            if status in ("FAILED", "CANCELLED"):
                print()
                raise RuntimeError(f"Midjourney job failed: {result}")

            dots += 1
            print(f"\r  ⏳  Generating image{'.' * (dots % 4):<4}", end="", flush=True)
            time.sleep(interval)
        raise TimeoutError(f"Midjourney job timed out after {timeout_sec}s")

    @with_retry(max_attempts=3, base_delay=2.0, label="mj_download")
    def _download_image(self, img_url: str, job_id: str, idx: int) -> str:
        resp = requests.get(img_url, stream=True, timeout=60)
        resp.raise_for_status()
        filename = f"mj_{job_id}_{idx}.png"
        path = os.path.join(IMAGES_DIR, filename)
        with open(path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        logger.debug("Downloaded image → %s", path)
        return path

    def generate(self, prompt: str, mode: str = "fast", aspect_ratio: str = "16:9") -> List[str]:
        job_id = self._submit_imagine(prompt, mode, aspect_ratio)
        job_data = self._poll_job(job_id)
        images = job_data.get("images", [])
        if not images:
            raise RuntimeError("Midjourney returned no images.")
        paths = [self._download_image(url, job_id, i + 1) for i, url in enumerate(images)]
        return paths


_mj_client = MidjourneyClient()


# ── node ──────────────────────────────────────────────────────────────────────

def generate_image(state: dict) -> dict:
    stage_banner(4)
    logger.info("Node: generate_image")

    example_en: str  = state["example_sentence_en"]
    example_de: str  = state.get("example_sentence_de", "")
    article: str     = state.get("article", "")
    german_word: str = state.get("german_word", "")

    # Build a gender hint so Midjourney shows the right sex when the word is
    # a gendered noun (der → male, die → female, das / non-noun → no hint).
    gender_hint = ""
    if article == "der":
        gender_hint = (
            f' IMPORTANT: The German word "{german_word}" is masculine (der). '
            "If the image shows a person, they must be clearly male (a man or a boy)."
        )
    elif article == "die":
        gender_hint = (
            f' IMPORTANT: The German word "{german_word}" is feminine (die). '
            "If the image shows a person, they must be clearly female (a woman or a girl)."
        )

    # 1. Generate Midjourney prompt via LLM
    _RULES = (
        "\n\nRULES:\n"
        "- Output ONLY the image description — no explanations, no preamble, no markdown\n"
        "- Do NOT include any parameter flags (no --v, --q, --style, --ar, etc.) — they are added automatically\n"
        "- Do NOT use double hyphens (--) anywhere in the text\n"
        "- Do NOT use quotation marks in the output"
    )

    _IMMERSIVE = (
        "Frame the shot so the viewer feels placed directly inside the scene: "
        "The composition should feel lived-in and immediate, as if the viewer just walked into the moment. "
    )

    if FUNNY_MODE and example_de:
        mj_req = (
            "Generate a Midjourney prompt for a photorealistic, cinematic image that is funny and aesthetically beautiful. "
            "The sentence below is a humorous observation — visually depict the joke. "
            "Show the ironic situation with expressive faces or body language (proud smile, knowing look, deadpan stare). "
            f"{_IMMERSIVE}"
            "Warm, well-lit, natural setting. Photorealistic photography, NOT illustration or cartoon.\n\n"
            f"Sentence: \"{example_en}\""
            f"{gender_hint}"
            f"{_RULES}"
        )
        system_prompt = (
            "You are an expert Midjourney prompt engineer specialising in immersive, photorealistic funny scenes. "
            "Prioritise compositions that place the viewer inside the scene. "
            "Always include specific camera model, lens, and lighting descriptors (e.g. 'shot on Sony A7IV, 50mm f/1.4, golden hour'). "
            "Never use words like 'painting', 'illustration', 'artistic', 'rendered', 'digital art'. "
            "No parameter flags. No double hyphens. Output only the description."
        )
    else:
        mj_req = (
            "Generate a Midjourney prompt for a beautiful, photorealistic 16:9 photography. "
            f"{_IMMERSIVE}"
            "No text in the image. Visually appealing for social media.\n\n"
            f"Sentence: \"{example_en}\""
            f"{gender_hint}"
            f"{_RULES}"
        )
        system_prompt = (
            "You are an expert Midjourney prompt engineer. "
            "You create vivid, immersive, photorealistic prompts that place the viewer inside the scene. "
            "Prioritise ground-level or eye-level framing with foreground depth. "
            "Always include specific camera model, lens, and lighting descriptors (e.g. 'shot on Sony A7IV, 50mm f/1.4, golden hour'). "
            "Never use words like 'painting', 'illustration', 'artistic', 'rendered', 'digital art'. "
            "No parameter flags. No double hyphens. Output only the description."
        )

    midjourney_prompt: str = retry_call(
        get_ai_response,
        mj_req,
        system_prompt,
        max_tokens=400,
        temperature=0.8,
        label="mj_prompt",
    ).strip()

    # Strip any --parameter flags the AI may have included despite instructions,
    # and replace any curly/smart quotes with straight ones to avoid parsing issues.
    import re
    midjourney_prompt = re.sub(r"\s*--\w[\w\d]*.*$", "", midjourney_prompt).strip()
    midjourney_prompt = midjourney_prompt.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", "").replace("\u201d", "")
    PHOTO_SUFFIX = (
        ", shot on Canon EOS R5, 35mm lens, natural lighting, "
        "RAW photo, ultra realistic, 8k UHD"
    )
    midjourney_prompt = midjourney_prompt.rstrip(".") + PHOTO_SUFFIX
    logger.debug("Midjourney prompt: %s", midjourney_prompt)
    print(f"  Prompt: {midjourney_prompt}", flush=True)

    # 2. Generate images
    image_paths = retry_call(
        _mj_client.generate,
        midjourney_prompt,
        mode="fast",
        aspect_ratio="16:9",
        max_attempts=3,
        base_delay=5.0,
        label="mj_generate",
    )

    # 3. Rank images with ImageReward and pick the best one
    # Use the English example sentence as the scoring prompt — it's the closest
    # natural-language description of what we wanted the image to depict.
    print(f"  ⏳  Ranking {len(image_paths)} images with ImageReward …", flush=True)
    chosen = pick_best_image(midjourney_prompt, image_paths)
    idx = image_paths.index(chosen) + 1
    ok(f"Best image: #{idx}/{len(image_paths)} → {os.path.basename(chosen)}")
    logger.info("Best image selected: %s (from %d options)", chosen, len(image_paths))

    return {
        **state,
        "midjourney_prompt": midjourney_prompt,
        "image_path": chosen,
    }
