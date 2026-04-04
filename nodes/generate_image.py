"""
Node: generate_image

Builds an image prompt via LLM from the tweet content, then calls the
configured provider (Midjourney via TTAPI or Grok Imagine), downloads
images, and picks the best with ImageReward.  Optionally overlays a
source/target flag badge (FLAG_OVERLAY).

================================================================================
 SECTIONS YOU MAY WANT TO EDIT
================================================================================
  _RULES                  — global rules appended to every image prompt
  _DISNEY_AESTHETIC       — style block for image_style=disney
  _IMMERSIVE / _AESTHETIC — style blocks for image_style=photographic
  STYLE_SUFFIX (Midjourney)— suffix added to prompts (e.g. "positive facial expressions")
  Flag overlay: _overlay_flags(), _create_flag_badge() — see _FLAGCDN_* constants
================================================================================

IMAGE_PROVIDER (settings.env): "midjourney" (TT_API_KEY), "grok" (XAI_API_KEY), or "z-image-turbo" (ComfyUI local).
"""

import os
import re
import time
import logging
import requests
from typing import List
from datetime import datetime

import config
from config import TT_API_KEY, IMAGES_DIR, resolve_image_style
from services.ai_client import get_ai_response
from services.image_ranker import pick_best_image, score_image
from services.zit_image import ComfyUIUnavailableError
from utils.retry import retry_call, with_retry
from utils.ui import stage_banner, ok, info, warn as ui_warn

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


# ── Grok Imagine client ───────────────────────────────────────────────────────

_GROK_IMAGE_MODEL = "grok-imagine-image"
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
        resp = requests.get(img_url, stream=True, timeout=60)
        resp.raise_for_status()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"grok_{ts}_{idx}.png"
        path = os.path.join(IMAGES_DIR, filename)
        with open(path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        logger.debug("Downloaded Grok image → %s", path)
        return path

    def generate(self, prompt: str, n: int = 1, aspect_ratio: str = "16:9") -> List[str]:
        payload = {
            "model": _GROK_IMAGE_MODEL,
            "prompt": prompt,
            "n": n,
            "aspect_ratio": aspect_ratio,
            "response_format": "url",
        }
        print(f"\r  ⏳  Requesting {n} image(s) from Grok Imagine …", flush=True)
        resp = requests.post(
            f"{_XAI_BASE_URL}/images/generations",
            headers=self._headers(),
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("data", [])
        if not items:
            raise RuntimeError(f"Grok Imagine returned no images: {data}")
        print()
        logger.info("Grok Imagine returned %d image(s).", len(items))
        paths = []
        for i, item in enumerate(items):
            url = item.get("url") or item.get("b64_json")
            if not url:
                logger.warning("Grok Imagine item %d has no URL — skipping.", i + 1)
                continue
            if url.startswith("data:") or not url.startswith("http"):
                # base64 fallback
                import base64
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


# ── image prompt builder ──────────────────────────────────────────────────────

def _build_image_prompt(
    example_en: str,
    example_de: str,
    full_tweet: str,
    image_style: str,
    funny: bool,
) -> str:
    """Call the LLM once and return a single image generation prompt string."""
    is_zit = config.IMAGE_PROVIDER in ("z-image-turbo", "z-image-base")

    _param_flag_rule = (
        "- Do NOT include any parameter flags (no --v, --q, --style, --ar, etc.) — they are added automatically\n"
        if config.IMAGE_PROVIDER == "midjourney" else
        "- Do NOT include any parameter flags (no --v, --q, --style, --ar, etc.)\n"
    )
    _RULES = (
        "\n\nRULES:\n"
        "- Output ONLY the image description — no explanations, no preamble, no markdown\n"
        + _param_flag_rule +
        "- Do NOT use double hyphens (--) anywhere in the text\n"
        "- Do NOT use quotation marks in the output\n"
        "- Any human or character faces must show a natural, positive expression "
        "(genuine smile, relaxed, engaged) — never shocked, disgusted, fearful, or negative\n"
        "- The image must be VISUALLY CLEAN and aesthetically pleasing at all times — "
        "no spills, stains, smears, mess, splatter, dirt, grime, or food/liquid on skin, "
        "clothing, or surfaces. Everything in the frame should look polished, tidy, and "
        "magazine-worthy. If the scene involves food, show it beautifully plated and "
        "untouched — never messy eating, dripping, or smeared\n"
    )
    if is_zit:
        _RULES += (
            "- When people or human-like characters appear: describe anatomy unambiguously — "
            "each person has two arms and two hands; say what each visible hand is doing or that a hand is out of frame\n"
            "- Unless essential to the tweet, avoid mirrors, dense overlapping crowds, or many raised arms in one cluster; "
            "prefer one main subject or clearly separated figures\n"
            "- Prefer simple relaxed hand poses over intricate interlocking or ambiguous gestures\n"
        )

    # Derive correct aspect ratio label from configured resolution
    if is_zit:
        if config.IMAGE_PROVIDER == "z-image-base":
            w, h = config.Z_IMAGE_BASE_WIDTH, config.Z_IMAGE_BASE_HEIGHT
        else:
            w, h = config.Z_IMAGE_TURBO_WIDTH, config.Z_IMAGE_TURBO_HEIGHT
        if w > h:
            _aspect_hint = f"wide {w}×{h} landscape frame"
        elif h > w:
            _aspect_hint = f"tall {w}×{h} portrait frame"
        else:
            _aspect_hint = "square frame"
    else:
        _aspect_hint = "16:9"

    if image_style == "disney":
        _DISNEY_AESTHETIC = (
            "Style: polished 3D CGI animation in the style of Pixar and Walt Disney. "
            "Stylised shapes with clear, confident silhouettes. "
            "Characters have expressive eyes and readable facial features — personality-driven, not overly saccharine. "
            "Colour palette: rich, harmonious tones — warm ambers, deep blues, forest greens, and saturated accents "
            "grounded by neutral mid-tones. "
            "Lighting: cinematic directional light with strong contrast, rim highlights, and atmospheric depth, "
            "as if lit for a Pixar feature film. "
            "Background: a purposeful environment with painterly detail, soft depth of field, and clear visual hierarchy. "
            "Everything feels polished, characterful, and cinematic. "
            "The image should look like a still from a Pixar or Disney animated feature."
        )
        if funny and example_de:
            tweet_context = f"Full tweet:\n{full_tweet}\n\n" if full_tweet else ""
            if is_zit:
                img_req = (
                    f"A {config.SOURCE_LANGUAGE} learning tweet contains a joke. "
                    f"Write a rich, detailed image description for a Disney/Pixar-style 3D animated scene "
                    f"in a {_aspect_hint} that shows the punchline clearly and with visual wit. "
                    "Use full natural-language sentences, not comma-separated tags. "
                    "Describe the setting, characters, their expressions and body language, the lighting, "
                    "and the colour palette in concrete detail — aim for 100–200 words.\n\n"
                    f"{tweet_context}"
                    f"{config.SOURCE_LANGUAGE} sentence: \"{example_de}\"\n"
                    f"{config.TARGET_LANGUAGE} sentence: \"{example_en}\"\n\n"
                    "Step 1 — Identify the punchline: find the ironic twist, absurd contrast, or subverted expectation.\n"
                    "Step 2 — Stage it visually: describe expressive body language and facial expressions that land the joke — "
                    "the comedy should be immediately readable from the image alone.\n"
                    "Step 3 — Describe the environment: background, lighting direction (rim, key, fill), colour temperature.\n"
                    "Step 4 — Keep it clean: ONE main character, ONE clear joke, uncluttered focused background.\n"
                    "Step 5 — Keep it family-friendly: warm, uplifting, never dark or unsettling.\n\n"
                    f"{_DISNEY_AESTHETIC}"
                    f"{_RULES}"
                )
                system_prompt = (
                    "You are an expert Disney/Pixar 3D animation image description writer for Z-Image-Turbo, "
                    "a model that excels at detailed natural-language scene descriptions. "
                    "Write rich, flowing prose that covers subject, environment, lighting, colour palette, and mood — "
                    "never use comma-separated tags. "
                    "Every description should feel like a scene memo from a Pixar director: "
                    "specific, visual, and full of personality. "
                    "Output only the image description."
                )
            else:
                img_req = (
                    f"A {config.SOURCE_LANGUAGE} learning tweet contains a joke. "
                    "Create an image generation prompt for a Disney/Pixar-style 3D animated scene "
                    "that shows the punchline of the joke clearly and with visual wit.\n\n"
                    f"{tweet_context}"
                    f"{config.SOURCE_LANGUAGE} sentence: \"{example_de}\"\n"
                    f"{config.TARGET_LANGUAGE} sentence: \"{example_en}\"\n\n"
                    "Step 1 — Identify the punchline: find the ironic twist, absurd contrast, or subverted expectation.\n"
                    "Step 2 — Stage it visually: use expressive body language and facial expressions to land the joke — "
                    "the comedy should be immediately readable from the image alone.\n"
                    "Step 3 — Make it cinematic and polished: deliberate lighting, strong composition, rich colours. "
                    "Think of a memorable frame from a Pixar feature — that level of craft and visual storytelling.\n"
                    "Step 4 — Keep it clean: ONE main character, ONE clear joke, uncluttered focused background.\n"
                    "Step 5 — Keep it family-friendly: warm, uplifting, never dark or unsettling.\n\n"
                    f"{_DISNEY_AESTHETIC}"
                    f"{_RULES}"
                )
                system_prompt = (
                    "You are an expert Disney/Pixar 3D animation prompt engineer. "
                    "You write image prompts that produce polished, expressive, and funny animated stills. "
                    "Every prompt you write feels like a frame from a Pixar feature: "
                    "strong character silhouettes, expressive faces, cinematic lighting, rich cohesive colours. "
                    "Humour is conveyed through clear visual storytelling and expressive performance, never saccharine excess. "
                    "Never mention photography, cameras, lenses, or film. "
                    "No parameter flags. No double hyphens. Output only the image description."
                )
        else:
            if is_zit:
                img_req = (
                    f"Write a rich, detailed image description for a Disney/Pixar-style 3D animated scene "
                    f"in a {_aspect_hint}. "
                    "Use full natural-language sentences, not comma-separated tags. "
                    "Describe the setting, the main character(s), their pose and expression, "
                    "the lighting (direction, quality, colour temperature), "
                    "and the colour palette in concrete detail — aim for 100–200 words.\n\n"
                    f"Scene to illustrate: \"{example_en}\"\n\n"
                    "Design a visually compelling, characterful scene that brings this sentence to life. "
                    "Characters should have expressive features and strong readable silhouettes. "
                    "The scene should look like a cinematic still from a Pixar or Disney animated feature — "
                    "polished, purposeful, and full of personality without being saccharine.\n\n"
                    f"{_DISNEY_AESTHETIC}"
                    "No text visible in the image."
                    f"{_RULES}"
                )
                system_prompt = (
                    "You are an expert Disney/Pixar 3D animation image description writer for Z-Image-Turbo, "
                    "a model that excels at detailed natural-language scene descriptions. "
                    "Write rich, flowing prose that covers subject, environment, lighting, colour palette, and mood — "
                    "never use comma-separated tags. Aim for the detail of a Pixar art director's scene brief. "
                    "Output only the image description."
                )
            else:
                img_req = (
                    "Create an image generation prompt for a Disney/Pixar-style 3D animated scene.\n\n"
                    f"Sentence: \"{example_en}\"\n\n"
                    "Design a visually compelling, characterful scene that brings this sentence to life. "
                    "Characters should have expressive features and strong readable silhouettes. "
                    "The scene should look like a cinematic still from a Pixar or Disney animated feature — "
                    "polished, purposeful, and full of personality without being saccharine.\n\n"
                    f"{_DISNEY_AESTHETIC}"
                    "No text in the image."
                    f"{_RULES}"
                )
                system_prompt = (
                    "You are an expert Disney/Pixar 3D animation prompt engineer. "
                    "You write image prompts that produce cinematic, expressive, Pixar-quality stills. "
                    "Strong character design, deliberate lighting, rich cohesive colour palette — "
                    "every element should feel polished, purposeful, and full of personality. "
                    "Avoid over-sweetening: aim for charming and engaging, not saccharine. "
                    "Never mention photography, cameras, lenses, or film. "
                    "No parameter flags. No double hyphens. Output only the image description."
                )
    else:
        _IMMERSIVE = (
            "Frame the shot so the viewer feels placed directly inside the scene: "
            "The composition should feel lived-in and immediate, as if the viewer just walked into the moment. "
        )
        _CLEAN_AESTHETIC = (
            "Composition: ONE clear subject, uncluttered frame, minimal background elements. "
            "The joke or mood must be immediately readable at a glance — never crowd the scene. "
        )
        _AESTHETIC = (
            "Aesthetics: make this image genuinely beautiful — not just technically correct. "
            "Think carefully about: harmonious colour palette (warm, vibrant, or richly contrasted), "
            "flattering and dramatic natural light (golden hour, soft side-light, or crisp morning sun), "
            "shallow depth of field to isolate the subject against a beautifully blurred background, "
            "and a composition that would stop someone mid-scroll. "
            "The image should look like a professional editorial photo that people want to share for its looks alone. "
        )
        if funny and example_de:
            tweet_context = f"Full tweet:\n{full_tweet}\n\n" if full_tweet else ""
            if is_zit:
                img_req = (
                    f"A {config.SOURCE_LANGUAGE} learning tweet contains a joke. "
                    f"Write a rich, detailed photographic scene description for Z-Image-Turbo in a {_aspect_hint} "
                    "that is BOTH visually stunning AND makes the punchline instantly obvious. "
                    "Use flowing natural-language sentences — not comma tags. "
                    "Cover: the subject and their expression/body language, the environment, "
                    "the lighting (direction, quality, colour temperature), the colour palette, "
                    "and the camera framing. Aim for 100–200 words.\n\n"
                    f"{tweet_context}"
                    f"{config.SOURCE_LANGUAGE} sentence: \"{example_de}\"\n"
                    f"{config.TARGET_LANGUAGE} sentence: \"{example_en}\"\n\n"
                    "Step 1 — Identify the punchline: the ironic twist, subverted expectation, or absurd contrast.\n"
                    "Step 2 — Stage it: describe a real scene where expressions or body language land the joke — "
                    "the comedy must read from the image alone.\n"
                    "Step 3 — Make it beautiful: golden-hour or cinematic light, rich colours, "
                    "shallow depth of field — a composition worth sharing.\n"
                    "Step 4 — Keep it photorealistic: no CGI, no animation, no cartoon. ONE subject, uncluttered frame.\n\n"
                    f"{_IMMERSIVE}"
                    f"{_CLEAN_AESTHETIC}"
                    f"{_AESTHETIC}"
                    f"{_RULES}"
                )
                system_prompt = (
                    "You are an expert photographic scene description writer for Z-Image-Turbo. "
                    "This model responds best to detailed natural-language prose describing subjects, "
                    "environments, lighting, and colour — never comma-separated tag lists. "
                    "Your descriptions are both visually stunning and instantly funny: "
                    "a clear visual punchline combined with cinematic photographic beauty. "
                    "ALL output must be photorealistic. Never describe CGI, animation, or cartoon styles. "
                    "Output only the scene description."
                )
            else:
                img_req = (
                    f"A {config.SOURCE_LANGUAGE} learning tweet contains a joke. Your job is to create an image generation prompt that is "
                    "BOTH visually stunning AND makes the punchline of the joke instantly obvious.\n\n"
                    f"{tweet_context}"
                    f"{config.SOURCE_LANGUAGE} sentence: \"{example_de}\"\n"
                    f"{config.TARGET_LANGUAGE} sentence: \"{example_en}\"\n\n"
                    "Step 1 — Identify the punchline: find the ironic twist, the subverted expectation, or the absurd contrast.\n"
                    "Step 2 — Stage it visually: design a real photographic scene that shows the punchline, "
                    "body language, or clever composition. The comedy must land from the image alone.\n"
                    "Step 3 — Make it beautiful: apply deliberate aesthetic choices — golden-hour light, rich colours, "
                    "shallow depth of field, a composition worth sharing for its looks alone. "
                    "Beauty and humour must coexist: a stunning photograph that is also funny.\n"
                    "Step 4 — Keep it clean and readable: ONE subject, ONE joke, uncluttered frame.\n"
                    "Step 5 — Keep it positive and photorealistic: warm, light-hearted, family-friendly. "
                    "The output MUST be a photorealistic photograph — never animation, illustration, or cartoon.\n\n"
                    f"{_IMMERSIVE}"
                    f"{_CLEAN_AESTHETIC}"
                    f"{_AESTHETIC}"
                    "Photorealistic photography ONLY — no CGI, no illustration, no cartoon, no animation."
                    f"{_RULES}"
                )
                system_prompt = (
                    "You are an expert photographic image prompt engineer who creates images that are both visually stunning "
                    "and instantly funny. Your prompts always combine: (1) a clear visual punchline readable from the photo alone, "
                    "and (2) deliberately beautiful photographic aesthetics — perfect light, rich colours, shallow depth of field, "
                    "editorial composition. "
                    "ALL output must be photorealistic photography. NEVER describe CGI, illustration, animation, or Pixar/Disney style — "
                    "even for unusual or humorous subjects. If a scene seems absurd, make it work as a clever, well-composed photograph. "
                    "Always include specific camera model, lens, and lighting descriptors "
                    "(e.g. 'shot on Sony A7IV, 50mm f/1.4, golden hour backlight'). "
                    "Never use words like 'painting', 'illustration', 'artistic', 'rendered', 'digital art', 'animated', 'cartoon'. "
                    "No parameter flags. No double hyphens. Output only the description."
                )
        else:
            if is_zit:
                img_req = (
                    f"Write a rich, detailed photographic scene description for Z-Image-Turbo in a {_aspect_hint}. "
                    "Use flowing natural-language sentences — not comma tags. "
                    "Describe: the main subject (identity, pose, expression), the environment and background, "
                    "the lighting (direction, quality, colour temperature — e.g. warm golden-hour backlight, "
                    "soft diffused daylight, cinematic key light), the colour palette, "
                    "and the overall mood. Aim for 100–200 words.\n\n"
                    f"Scene to illustrate: \"{example_en}\"\n\n"
                    f"{_IMMERSIVE}"
                    f"{_CLEAN_AESTHETIC}"
                    f"{_AESTHETIC}"
                    "Photorealistic scene — no text visible in the image."
                    f"{_RULES}"
                )
                system_prompt = (
                    "You are an expert photographic scene description writer for Z-Image-Turbo, "
                    "a model that excels at detailed natural-language descriptions. "
                    "Write rich, flowing prose that covers subject, environment, lighting direction and quality, "
                    "colour palette, and mood — never comma-separated tags. "
                    "Every description should be a complete cinematic scene brief: "
                    "specific enough that a photographer could recreate the exact shot. "
                    "Output only the scene description."
                )
            else:
                img_req = (
                    f"Generate an image generation prompt for a photorealistic, aesthetically stunning {_aspect_hint} photograph.\n\n"
                    f"Sentence: \"{example_en}\"\n\n"
                    f"{_IMMERSIVE}"
                    f"{_CLEAN_AESTHETIC}"
                    f"{_AESTHETIC}"
                    "No text in the image."
                    f"{_RULES}"
                )
                system_prompt = (
                    "You are an expert image generation prompt engineer who creates images that look like professional "
                    "editorial photography. Every prompt you write is deliberately beautiful: perfect light, "
                    "rich harmonious colours, shallow depth of field, and a composition people want to share. "
                    "ONE clear subject, uncluttered frame — every element serves the main subject. "
                    "Always include specific camera model, lens, and lighting descriptors (e.g. 'shot on Sony A7IV, 50mm f/1.4, golden hour'). "
                    "Never use words like 'painting', 'illustration', 'artistic', 'rendered', 'digital art'. "
                    "No parameter flags. No double hyphens. Output only the description."
                )

    max_tokens = 700 if is_zit else 400

    image_prompt: str = retry_call(
        get_ai_response,
        img_req,
        system_prompt,
        max_tokens=max_tokens,
        temperature=0.8,
        label="img_prompt",
    ).strip()

    image_prompt = image_prompt.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", "").replace("\u201d", "")

    if config.IMAGE_PROVIDER == "midjourney":
        image_prompt = re.sub(r"\s*--\w[\w\d]*.*$", "", image_prompt).strip()
        if image_style == "disney":
            STYLE_SUFFIX = (
                ", Pixar 3D animation style, expressive character design, "
                "strong silhouettes, cinematic directional lighting, rich saturated colours, "
                "8K render, polished and characterful"
            )
        else:
            STYLE_SUFFIX = (
                ", shot on Canon EOS R5, 35mm lens, natural lighting, "
                "RAW photo, ultra realistic, 8k UHD, "
                "positive joyful atmosphere, warm and welcoming, bright uplifting mood, "
                "subjects with natural warm smiles, positive facial expressions"
            )
        image_prompt = image_prompt.rstrip(".") + STYLE_SUFFIX

    if is_zit:
        # Strip any accidental --flags the LLM may have included
        image_prompt = re.sub(r"\s*--\w[\w\d]*.*$", "", image_prompt).strip()
        suffix = config.Z_IMAGE_PROMPT_SUFFIX
        if suffix:
            image_prompt = image_prompt.rstrip(".").rstrip(",") + ". " + suffix

    return image_prompt


# ── lazy client init (only the active provider is instantiated) ───────────────

def _make_client():
    if config.IMAGE_PROVIDER == "grok":
        return GrokImagineClient()
    if config.IMAGE_PROVIDER == "z-image-turbo":
        from services.zit_image import ZITImageClient
        return ZITImageClient()
    if config.IMAGE_PROVIDER == "z-image-base":
        from services.zimage_base import ZImageBaseClient
        return ZImageBaseClient()
    return MidjourneyClient()


# ── node ──────────────────────────────────────────────────────────────────────

def generate_image(state: dict) -> dict:
    stage_banner(4)
    logger.info("Node: generate_image")

    example_en: str  = state["example_sentence_target"]
    example_de: str  = state.get("example_sentence_source", "")
    full_tweet: str  = state.get("full_tweet", "")
    cycle: int       = state.get("cycle", 0)
    image_style: str = resolve_image_style(cycle)
    funny: bool      = config.resolve_tweet_style(cycle) == "funny"
    logger.info("Image style for cycle %d: %s", cycle, image_style)
    _image_client = _make_client()

    n = config.GENERATED_IMAGE_COUNT

    # ── 1. Build image prompt(s) ──────────────────────────────────────────────
    if config.INDIVIDUAL_IMAGE_PROMPTS and n > 1:
        info(f"  Individual prompts ON — generating {n} unique prompts …")
        prompts: list[str] = []
        for i in range(n):
            p = _build_image_prompt(example_en, example_de, full_tweet, image_style, funny)
            prompts.append(p)
            logger.debug("Prompt %d/%d: %s", i + 1, n, p)
            print(f"  Prompt {i + 1}/{n}: {p[:100]}{'…' if len(p) > 100 else ''}", flush=True)
        # Representative prompt for backwards-compatible state key: use the first one.
        image_prompt = prompts[0]
    else:
        image_prompt = _build_image_prompt(example_en, example_de, full_tweet, image_style, funny)
        prompts = [image_prompt] * n
        logger.debug("Image prompt (%s): %s", config.IMAGE_PROVIDER, image_prompt)
        print(f"  Prompt: {image_prompt}", flush=True)

    # ── 2. Generate images via the configured provider ────────────────────────
    # Each entry in prompt_image_pairs: (prompt_used, image_path)
    prompt_image_pairs: list[tuple[str, str]] = []

    try:
        if config.IMAGE_PROVIDER == "grok":
            if config.INDIVIDUAL_IMAGE_PROMPTS and n > 1:
                for i, prompt in enumerate(prompts):
                    print(f"  ⏳  Generating image {i + 1}/{n} (individual prompt) …", flush=True)
                    paths = retry_call(
                        _image_client.generate,
                        prompt,
                        n=1,
                        aspect_ratio="16:9",
                        max_attempts=3,
                        base_delay=5.0,
                        label=f"grok_generate_{i + 1}/{n}",
                    )
                    for p in paths:
                        prompt_image_pairs.append((prompt, p))
            else:
                paths = retry_call(
                    _image_client.generate,
                    image_prompt,
                    n=n,
                    aspect_ratio="16:9",
                    max_attempts=3,
                    base_delay=5.0,
                    label="grok_generate",
                )
                for p in paths:
                    prompt_image_pairs.append((image_prompt, p))

        elif config.IMAGE_PROVIDER == "z-image-turbo":
            import random
            _image_client.ensure_ready()
            _image_client.purge_vram_before_batch()
            for i, prompt in enumerate(prompts):
                seed = random.randint(0, 2**31 - 1)
                label = f"z_image_turbo_{i + 1}/{n}"
                print(f"  ⏳  Generating image {i + 1}/{n} (seed {seed}) …", flush=True)
                paths = retry_call(
                    _image_client.generate,
                    prompt,
                    seed=seed,
                    max_attempts=3,
                    base_delay=10.0,
                    label=label,
                )
                for p in paths:
                    prompt_image_pairs.append((prompt, p))
            _image_client.unload_models()

        elif config.IMAGE_PROVIDER == "z-image-base":
            import random
            seeds = [random.randint(0, 2**31 - 1) for _ in prompts]
            print(
                f"  ⏳  Generating {len(prompts)} image(s) via Z-Image Base "
                f"({config.Z_IMAGE_BASE_STEPS} steps, cfg {config.Z_IMAGE_BASE_GUIDANCE_SCALE}) …",
                flush=True,
            )
            all_paths = retry_call(
                _image_client.generate_batch,
                prompts,
                seeds,
                max_attempts=3,
                base_delay=15.0,
                label="z_image_base_batch",
            )
            for prompt, path in zip(prompts, all_paths):
                prompt_image_pairs.append((prompt, path))

        else:  # midjourney
            if config.INDIVIDUAL_IMAGE_PROMPTS and n > 1:
                for i, prompt in enumerate(prompts):
                    print(f"  ⏳  Generating image {i + 1}/{n} (individual prompt) …", flush=True)
                    paths = retry_call(
                        _image_client.generate,
                        prompt,
                        mode="fast",
                        aspect_ratio="16:9",
                        max_attempts=3,
                        base_delay=5.0,
                        label=f"mj_generate_{i + 1}/{n}",
                    )
                    for p in paths:
                        prompt_image_pairs.append((prompt, p))
            else:
                paths = retry_call(
                    _image_client.generate,
                    image_prompt,
                    mode="fast",
                    aspect_ratio="16:9",
                    max_attempts=3,
                    base_delay=5.0,
                    label="mj_generate",
                )
                for p in paths:
                    prompt_image_pairs.append((image_prompt, p))

    except ComfyUIUnavailableError as exc:
        ui_warn(f"ComfyUI unavailable — skipping image and video for this cycle. ({exc})")
        logger.warning("ComfyUI unavailable: %s", exc)
        return {
            **state,
            "midjourney_prompt":   image_prompt,
            "image_path":          None,
            "comfyui_unavailable": True,
        }

    if (
        config.IMAGE_PROVIDER in ("z-image-turbo", "z-image-base")
        and config.ENABLE_INSTRUCTIR_ENHANCE
        and prompt_image_pairs
    ):
        from services.instructir_enhance import enhance_image_path

        n_ir = len(prompt_image_pairs)
        print(f"  ⏳  Enhancing {n_ir} image(s) with InstructIR …", flush=True)
        _ir_pairs: list[tuple[str, str]] = []
        for i, (prompt, pth) in enumerate(prompt_image_pairs, start=1):
            print(f"  ⏳  InstructIR {i}/{n_ir} → {os.path.basename(pth)} …", flush=True)
            _ir_pairs.append((prompt, enhance_image_path(pth)))
        prompt_image_pairs = _ir_pairs
        ok(f"InstructIR: enhanced {n_ir} image(s).")

    # Shut down ComfyUI after all image work (generation + enhancement) so its
    # CUDA context is fully released before WAN2.1 loads its 14B model.
    # z-image-base runs in its own subprocess and releases VRAM automatically on exit.
    if config.IMAGE_PROVIDER == "z-image-turbo":
        from services.zit_image import shutdown_comfyui
        shutdown_comfyui()

    image_paths = [p for _, p in prompt_image_pairs]
    if len(set(image_paths)) != len(image_paths):
        logger.error(
            "Duplicate image paths in batch — files were likely overwritten; "
            "ranking vs prompts may be wrong. Paths: %s",
            image_paths,
        )
        ui_warn(
            "Duplicate saved image paths detected — one file may have been overwritten. "
            "Check Z-Image save naming (zit_image.py)."
        )

    # ── 3. Rank images and pick the best ──────────────────────────────────────
    print(f"  ⏳  Ranking {len(image_paths)} image(s) with ImageReward …", flush=True)

    if config.INDIVIDUAL_IMAGE_PROMPTS and n > 1:
        # Score each image against the prompt it was generated from.
        scored = [(score_image(prompt, path), prompt, path) for prompt, path in prompt_image_pairs]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_prompt, chosen = scored[0]
        idx = image_paths.index(chosen) + 1
        rank_summary = "  ".join(f"#{i+1} {s:.3f}" for i, (s, _, _) in enumerate(scored))
        ok(f"Best image: #{idx}/{len(image_paths)} (score {best_score:.3f}) → {os.path.basename(chosen)}")
        logger.info("ImageReward ranking (individual prompts): %s  →  best: %s (%.3f)", rank_summary, chosen, best_score)
        # Use the winning image's prompt as the canonical prompt for downstream steps.
        image_prompt = best_prompt
    else:
        chosen = pick_best_image(image_prompt, image_paths)
        idx = image_paths.index(chosen) + 1
        ok(f"Best image: #{idx}/{len(image_paths)} → {os.path.basename(chosen)}")
        logger.info("Best image selected: %s (from %d options)", chosen, len(image_paths))

    return {
        **state,
        "midjourney_prompt": image_prompt,   # key kept for backwards compatibility
        "image_path": chosen,
    }
