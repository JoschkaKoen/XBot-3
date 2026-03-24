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

import io
import math
import os
import time
import logging
import requests
from typing import List
from datetime import datetime

import config
from config import TT_API_KEY, IMAGES_DIR, resolve_image_style
from services.ai_client import get_ai_response
from services.image_ranker import pick_best_image
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


# ── lazy client init (only the active provider is instantiated) ───────────────

def _make_client():
    if config.IMAGE_PROVIDER == "grok":
        return GrokImagineClient()
    if config.IMAGE_PROVIDER == "z-image-turbo":
        from services.zit_image import ZITImageClient
        return ZITImageClient()
    return MidjourneyClient()


# ── flag overlay (PIL, applied after image download) ──────────────────────────

_FLAGCDN_URL      = "https://flagcdn.com/w{w}/{code}.png"
_FLAGCDN_WIDTHS   = [20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 320]


def _flag_emoji_to_country_code(emoji: str) -> str:
    """Extract the ISO 3166-1 alpha-2 country code from a flag emoji (e.g. 🇩🇪 → 'de')."""
    chars = [
        chr(ord(c) - 0x1F1E6 + ord("A"))
        for c in emoji
        if 0x1F1E6 <= ord(c) <= 0x1F1FF
    ]
    return "".join(chars).lower()


_flag_cache: dict = {}   # (code, fetch_w) → PIL Image, avoids re-downloading each cycle


@with_retry(max_attempts=3, base_delay=0.1, backoff=1.0, label="fetch_flag")
def _fetch_flag(code: str, desired_width: int) -> "Image":
    """Download a flag PNG from flagcdn.com at the nearest supported width (cached)."""
    from PIL import Image
    fetch_w = next((w for w in _FLAGCDN_WIDTHS if w >= desired_width), 320)
    key = (code.lower(), fetch_w)
    if key not in _flag_cache:
        url = _FLAGCDN_URL.format(w=fetch_w, code=code.lower())
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        _flag_cache[key] = Image.open(io.BytesIO(resp.content)).convert("RGBA")
    return _flag_cache[key].copy()   # copy so callers can mutate without poisoning cache


def _fit_flag(flag: "Image", w: int, h: int) -> "Image":
    """Scale-to-fill: zoom until both dimensions are covered, then centre-crop."""
    from PIL import Image
    ratio = max(w / flag.width, h / flag.height)
    new_w = int(flag.width  * ratio)
    new_h = int(flag.height * ratio)
    flag = flag.resize((new_w, new_h), Image.LANCZOS)
    x0 = (new_w - w) // 2
    y0 = (new_h - h) // 2
    return flag.crop((x0, y0, x0 + w, y0 + h))


def _create_flag_badge(badge_w: int, badge_h: int) -> "Image":
    """
    Gradient-blended badge using real flag images from flagcdn.com.
    Target flag (learner's language) on the left, source flag (taught language) on the right.
    Cosine ease-in/out gradient for a smooth, seamless blend.
    """
    from PIL import Image
    src_code = _flag_emoji_to_country_code(config.SOURCE_FLAG)
    tgt_code = _flag_emoji_to_country_code(config.TARGET_FLAG)

    src_img = _fit_flag(_fetch_flag(src_code, badge_w * 2), badge_w, badge_h).convert("RGB")
    tgt_img = _fit_flag(_fetch_flag(tgt_code, badge_w * 2), badge_w, badge_h).convert("RGB")

    # Cosine gradient: left=255 (target fully visible) → right=0 (source fully visible)
    gradient = bytes(
        [int(128 + 127 * math.cos(math.pi * x / max(badge_w - 1, 1)))
         for x in range(badge_w)] * badge_h
    )
    mask = Image.frombytes("L", (badge_w, badge_h), gradient)
    return Image.composite(tgt_img, src_img, mask)


def _apply_rounded_corners(img: "Image", radius: int) -> "Image":
    """Return img with rounded corners via a rounded_rectangle alpha mask."""
    from PIL import Image, ImageDraw
    img = img.convert("RGBA")
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    img.putalpha(mask)
    return img


def _overlay_flags(image_path: str) -> str:
    """Composite a target→source flag badge onto the top-right corner of the image."""
    from PIL import Image, ImageDraw
    img = Image.open(image_path).convert("RGBA")
    iw, ih = img.size

    badge_w = max(int(iw * 0.10), 100)
    badge_h = int(badge_w * 0.60)
    padding = max(int(iw * 0.015), 12)
    radius  = max(int(badge_h * 0.20), 4)

    badge = _create_flag_badge(badge_w, badge_h)

    # Draw border BEFORE rounding so it is clipped cleanly with the corners
    ImageDraw.Draw(badge).rounded_rectangle(
        [0, 0, badge_w - 1, badge_h - 1], radius=radius,
        outline=(255, 255, 255), width=2,
    )

    # Rounded corners — clips both badge content and border in one pass
    badge = _apply_rounded_corners(badge, radius)

    # 90 % opacity
    r, g, b, a = badge.split()
    a = a.point(lambda v: int(v * 0.90))
    badge = Image.merge("RGBA", (r, g, b, a))

    img.paste(badge, (iw - badge_w - padding, padding), badge)
    img.convert("RGB").save(image_path, format="PNG")
    logger.info("Flag overlay applied → %s", os.path.basename(image_path))
    return image_path


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

    # 1. Generate image prompt via LLM
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
    )

    # ── Disney / Pixar style prompts ──────────────────────────────────────────
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

    # ── Photographic style prompts (default) ──────────────────────────────────
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
            img_req = (
                f"A {config.SOURCE_LANGUAGE} learning tweet contains a joke. Your job is to create an image generation prompt that is "
                "BOTH visually stunning AND makes the punchline of the joke instantly obvious.\n\n"
                f"{tweet_context}"
                f"{config.SOURCE_LANGUAGE} sentence: \"{example_de}\"\n"
                f"{config.TARGET_LANGUAGE} sentence: \"{example_en}\"\n\n"
                "Step 1 — Identify the punchline: find the ironic twist, the subverted expectation, or the absurd contrast.\n"
                "Step 2 — Stage it visually: design a scene that shows the punchline in action with exaggerated expressions "
                "or body language. The comedy must land from the image alone — the viewer should laugh before reading the tweet.\n"
                "Step 3 — Make it beautiful: apply deliberate aesthetic choices — golden-hour light, rich colours, "
                "shallow depth of field, a composition worth sharing for its looks alone. "
                "Beauty and humour must coexist: a stunning image that is also funny.\n"
                "Step 4 — Keep it clean and readable: ONE subject, ONE joke, uncluttered frame.\n"
                "Step 5 — Keep it positive: warm, light-hearted, family-friendly. "
                "The viewer should feel amused and uplifted — never unsettled.\n"
                "IMPORTANT: If the scene is absurd or impossible in real life (e.g. a walking cake, "
                "a talking animal, an object behaving like a person), do NOT render it photorealistically — "
                "that would look disturbing or uncanny. Instead, describe it as a charming 3D render in "
                "a Pixar/Disney style: soft rounded shapes, pastel colours, big expressive eyes, warm lighting. "
                "Cute and whimsical always beats realistic for impossible subjects.\n\n"
                f"{_IMMERSIVE}"
                f"{_CLEAN_AESTHETIC}"
                f"{_AESTHETIC}"
                "Photorealistic photography, NOT illustration or cartoon."
                f"{_RULES}"
            )
            system_prompt = (
                "You are an expert image generation prompt engineer who creates images that are both visually stunning "
                "and instantly funny. Your prompts always combine two things: (1) a clear visual punchline that "
                "lands from the image alone, and (2) deliberately beautiful aesthetics — perfect light, rich colours, "
                "shallow depth of field, editorial composition. "
                "You never sacrifice beauty for the joke or the joke for beauty — the best prompt delivers both. "
                "Humour is always warm and family-friendly. "
                "EXCEPTION — absurd or impossible subjects: if the scene involves something physically impossible "
                "(e.g. a walking food item, a talking object, an animal in a human role), do NOT render it "
                "photorealistically — that looks uncanny and disturbing. Instead use a charming Pixar/Disney 3D "
                "render style: soft rounded shapes, pastel tones, big expressive eyes, warm lighting. "
                "For all other (realistic) scenes: always include specific camera model, lens, and lighting "
                "descriptors (e.g. 'shot on Sony A7IV, 50mm f/1.4, golden hour backlight'). "
                "Never use words like 'painting', 'illustration', 'artistic', 'rendered', 'digital art'. "
                "No parameter flags. No double hyphens. Output only the description."
            )
        else:
            _aspect_hint = "square 1:1" if config.IMAGE_PROVIDER == "z-image-turbo" else "16:9"
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

    image_prompt: str = retry_call(
        get_ai_response,
        img_req,
        system_prompt,
        max_tokens=400,
        temperature=0.8,
        label="img_prompt",
    ).strip()

    # Clean up smart/curly quotes that can cause API parsing issues.
    import re
    image_prompt = image_prompt.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", "").replace("\u201d", "")

    if config.IMAGE_PROVIDER == "midjourney":
        # Strip any --parameter flags the AI may have included despite instructions.
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

    logger.debug("Image prompt (%s): %s", config.IMAGE_PROVIDER, image_prompt)
    print(f"  Prompt: {image_prompt}", flush=True)

    # 2. Generate images via the configured provider
    if config.IMAGE_PROVIDER == "grok":
        image_paths = retry_call(
            _image_client.generate,
            image_prompt,
            n=config.GROK_IMAGE_COUNT,
            aspect_ratio="16:9",
            max_attempts=3,
            base_delay=5.0,
            label="grok_generate",
        )
    elif config.IMAGE_PROVIDER == "z-image-turbo":
        image_paths = retry_call(
            _image_client.generate,
            image_prompt,
            max_attempts=3,
            base_delay=10.0,
            label="z_image_turbo_generate",
        )
    else:
        image_paths = retry_call(
            _image_client.generate,
            image_prompt,
            mode="fast",
            aspect_ratio="16:9",
            max_attempts=3,
            base_delay=5.0,
            label="mj_generate",
        )

    # 3. Rank images with ImageReward and pick the best one
    # Use the image prompt as the scoring reference — closest natural-language
    # description of what we wanted the image to depict.
    print(f"  ⏳  Ranking {len(image_paths)} images with ImageReward …", flush=True)
    chosen = pick_best_image(image_prompt, image_paths)
    idx = image_paths.index(chosen) + 1
    ok(f"Best image: #{idx}/{len(image_paths)} → {os.path.basename(chosen)}")
    logger.info("Best image selected: %s (from %d options)", chosen, len(image_paths))

    if config.FLAG_OVERLAY:
        try:
            _overlay_flags(chosen)
        except Exception as exc:
            logger.warning("Flag overlay skipped — could not fetch flag images: %s", exc)
            ui_warn(f"Flag overlay skipped (could not fetch flag images: {exc}) — image posted without flags.")

    return {
        **state,
        "midjourney_prompt": image_prompt,   # key kept for backwards compatibility
        "image_path": chosen,
    }
