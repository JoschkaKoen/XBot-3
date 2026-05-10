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
  styles/<name>.py        — per-style prompt blocks, midjourney_suffix, motion
                            guidance. To add a new style, drop a new file in
                            styles/ — it is auto-discovered each cycle.
  Flag overlay: _overlay_flags(), _create_flag_badge() — see _FLAGCDN_* constants
================================================================================

IMAGE_PROVIDER (settings.env): "midjourney" (TT_API_KEY), "grok" (XAI_API_KEY), or "z-image-turbo" (ComfyUI local).

================================================================================
 STATE CONTRACT
================================================================================
  Reads from state:   source_word, full_tweet, example_sentence_source,
                      example_sentence_target, cycle
  Writes to state:    midjourney_prompt, image_path, image_subject_gender,
                      comfyui_unavailable (set when ComfyUI is unreachable)
  Side effects:       writes PNG to Images/, may pre-warm ImageReward model
================================================================================
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
from services.image_clients import (
    MidjourneyClient,
    GrokImagineClient,
    ComfyUIUnavailableError,
)
from styles import get_style, PromptContext
from utils.errors import FatalProviderError
from utils.retry import retry_call, with_retry
from utils.ui import stage_banner, ok, info, warn as ui_warn

logger = logging.getLogger("xbot.generate_image")


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

    ctx = PromptContext(
        example_en=example_en,
        example_de=example_de,
        full_tweet=full_tweet,
        aspect_hint=_aspect_hint,
        rules=_RULES,
        source_language=config.SOURCE_LANGUAGE,
        target_language=config.TARGET_LANGUAGE,
    )
    img_req, system_prompt = get_style(image_style).build_image_prompt(
        funny=funny, is_zit=is_zit, ctx=ctx,
    )

    max_tokens = 700 if is_zit else 400

    image_prompt: str = get_ai_response(
        config.IMAGE_PROMPT_MODEL,
        img_req,
        system_prompt,
        max_tokens=max_tokens,
        temperature=0.8,
        retry_label="img_prompt",
    ).strip()

    image_prompt = image_prompt.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", "").replace("\u201d", "")

    if config.IMAGE_PROVIDER == "midjourney":
        image_prompt = re.sub(r"\s*--\w[\w\d]*.*$", "", image_prompt).strip()
        image_prompt = image_prompt.rstrip(".") + get_style(image_style).midjourney_suffix

    if is_zit:
        # Strip any accidental --flags the LLM may have included.
        # Z_IMAGE_PROMPT_SUFFIX below is style-agnostic — it's a global
        # fine-tuning hint applied to every Z-Image run regardless of style.
        # Per-style suffixes live in styles/<name>.py:midjourney_suffix.
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
            logger.info("Prompt %d/%d: %.100s%s", i + 1, n, p, "…" if len(p) > 100 else "")
        # Representative prompt for backwards-compatible state key: use the first one.
        image_prompt = prompts[0]
    else:
        image_prompt = _build_image_prompt(example_en, example_de, full_tweet, image_style, funny)
        prompts = [image_prompt] * n
        logger.debug("Image prompt (%s): %s", config.IMAGE_PROVIDER, image_prompt)
        logger.info("Prompt: %s", image_prompt)

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
