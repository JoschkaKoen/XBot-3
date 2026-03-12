"""
Node: generate_image

Generates an image prompt via LLM, then submits to the configured image
provider (Midjourney via TTAPI, or xAI Grok Imagine), polls until done,
downloads all images, and picks the best one using ImageReward-v1.0.
Falls back to the first image if the ranker model is unavailable.

IMAGE_PROVIDER setting controls which backend is used:
  "midjourney" (default) — requires TT_API_KEY
  "grok"                 — requires XAI_API_KEY
"""

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
    return MidjourneyClient()


# ── flag overlay (PIL, applied after image download) ──────────────────────────

def _draw_tricolor(badge_w: int, badge_h: int, colors: list):
    """Draw a flag as three equal horizontal bands using the given RGB color list."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (badge_w, badge_h))
    d = ImageDraw.Draw(img)
    bh = badge_h // 3
    d.rectangle([0, 0,       badge_w, bh],       fill=tuple(colors[0]))
    d.rectangle([0, bh,      badge_w, bh * 2],   fill=tuple(colors[1]))
    d.rectangle([0, bh * 2,  badge_w, badge_h],  fill=tuple(colors[2]))
    return img


def _create_flag_badge(badge_w: int, badge_h: int):
    """Return a PIL Image of a target→source blended flag badge."""
    from PIL import Image

    target_colors = config._parse_flag_colors(
        config.TARGET_FLAG_COLORS,
        default=[(178, 34, 52), (255, 255, 255), (60, 59, 110)],
    )
    source_colors = config._parse_flag_colors(
        config.SOURCE_FLAG_COLORS,
        default=[(0, 0, 0), (221, 0, 0), (255, 206, 0)],
    )

    target_img = _draw_tricolor(badge_w, badge_h, target_colors)
    source_img = _draw_tricolor(badge_w, badge_h, source_colors)

    # Gradient mask: 255 (left) = target visible, 0 (right) = source visible
    gradient = bytes(
        [255 - int(255 * x / max(badge_w - 1, 1)) for x in range(badge_w)] * badge_h
    )
    mask = Image.frombytes("L", (badge_w, badge_h), gradient)

    return Image.composite(target_img, source_img, mask)


def _apply_rounded_corners(img, radius: int):
    """Return img with rounded corners (requires RGBA)."""
    from PIL import Image, ImageDraw
    img = img.convert("RGBA")
    circle = Image.new("L", (radius * 2, radius * 2), 0)
    ImageDraw.Draw(circle).ellipse((0, 0, radius * 2, radius * 2), fill=255)
    w, h = img.size
    alpha = Image.new("L", (w, h), 255)
    for (x, y) in [(0, 0), (w - radius * 2, 0), (0, h - radius * 2), (w - radius * 2, h - radius * 2)]:
        alpha.paste(circle.crop((
            0 if x == 0 else radius,
            0 if y == 0 else radius,
            radius if x == 0 else radius * 2,
            radius if y == 0 else radius * 2,
        )), (x + (0 if x == 0 else radius), y + (0 if y == 0 else radius)))
    img.putalpha(alpha)
    return img


def _overlay_flags(image_path: str) -> str:
    """Composite a target→source flag badge onto the top-right corner of the image in-place."""
    from PIL import Image
    img = Image.open(image_path).convert("RGBA")
    iw, ih = img.size

    badge_w = max(int(iw * 0.09), 90)        # ~9 % of image width
    badge_h = int(badge_w * 0.60)
    padding = max(int(iw * 0.015), 12)
    radius  = max(badge_h // 5, 4)

    badge = _create_flag_badge(badge_w, badge_h)
    badge = _apply_rounded_corners(badge, radius)

    # Thin white border (2 px) for legibility
    from PIL import ImageDraw
    border_draw = ImageDraw.Draw(badge)
    border_draw.rounded_rectangle(
        [0, 0, badge_w - 1, badge_h - 1], radius=radius, outline=(255, 255, 255), width=2
    )

    # 85 % opacity
    r, g, b, a = badge.split()
    a = a.point(lambda v: int(v * 0.85))
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
        "- Do NOT use quotation marks in the output"
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
        if config.FUNNY_MODE and example_de:
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

        if config.FUNNY_MODE and example_de:
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
            img_req = (
                "Generate an image generation prompt for a photorealistic, aesthetically stunning 16:9 photograph.\n\n"
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
                "positive joyful atmosphere, warm and welcoming, bright uplifting mood"
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
        _overlay_flags(chosen)

    return {
        **state,
        "midjourney_prompt": image_prompt,   # key kept for backwards compatibility
        "image_path": chosen,
    }
