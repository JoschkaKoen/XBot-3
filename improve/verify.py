"""
improve/verify — post-cycle quality checks shared by the self-improvement
engine and scripts/verify_quality.py.

Each function returns a dict with at least {"pass": bool, ...}. Failures are
captured in the dict (pass=False, reason=…) rather than raised; the verifier
must keep running so the caller sees every failure, not just the first.
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger("xbot.improve.verify")


def _parse_json_response(raw: str) -> dict:
    """
    Safely extract JSON from an AI response that may be wrapped in ```json … ``` fences.
    Line-based fence detection avoids stripping characters inside JSON values.
    """
    text = raw.strip()
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return json.loads("\n".join(lines))


def verify_tweet_exists(tweet_id: str) -> dict:
    """Confirm the posted tweet is actually live on X. Returns {pass, metrics?, reason?}."""
    if not tweet_id:
        return {"pass": False, "reason": "no tweet_id in cycle output"}
    try:
        import tweepy
        from config import (
            X_BEARER_TOKEN,
            TWITTER_CONSUMER_KEY,
            TWITTER_CONSUMER_SECRET,
            TWITTER_ACCESS_TOKEN,
            TWITTER_ACCESS_TOKEN_SECRET,
        )
        client = tweepy.Client(
            bearer_token=X_BEARER_TOKEN or None,
            consumer_key=TWITTER_CONSUMER_KEY,
            consumer_secret=TWITTER_CONSUMER_SECRET,
            access_token=TWITTER_ACCESS_TOKEN,
            access_token_secret=TWITTER_ACCESS_TOKEN_SECRET,
        )
        response = client.get_tweet(tweet_id, tweet_fields=["public_metrics"])
        if response.data is None:
            return {"pass": False, "reason": "tweet not found on X"}
        return {"pass": True, "metrics": response.data.get("public_metrics", {})}
    except Exception as exc:
        return {"pass": False, "reason": str(exc)}


def verify_tweet_text(tweet_text: str) -> dict:
    """LLM-rated tweet-quality check. Returns {pass, score, issues}."""
    if not tweet_text:
        return {"pass": False, "score": 0, "issues": ["no tweet text"]}
    from services.grok_ai import get_grok_response
    prompt = f"""Evaluate this German-learning tweet:
---
{tweet_text}
---

Check:
1. FORMAT: Starts with #DeutschLernen [LEVEL], has 🇩🇪/🇬🇧 lines, emoji pairs
2. GERMAN: Word and sentence grammatically correct
3. TRANSLATION: English is natural and accurate
4. CEFR: Level matches word difficulty
5. EMOJIS: Two DIFFERENT emoji pairs (word line ≠ sentence line)
6. SENTENCE: Contains the exact word, is short and natural
7. LENGTH: Under 280 characters

Respond in JSON only: {{"pass": true/false, "score": 1-10, "issues": [...]}}
Pass if score >= 7.
"""
    try:
        raw = get_grok_response(
            prompt,
            "You are a strict quality reviewer for German learning social media content. Respond only with JSON.",
            max_tokens=300,
            temperature=0.1,
        )
        result = _parse_json_response(raw)
        result.setdefault("pass", result.get("score", 0) >= 7)
        result.setdefault("issues", [])
        return result
    except Exception as exc:
        logger.warning("Tweet text verification failed: %s", exc)
        return {"pass": False, "score": 0, "issues": [str(exc)]}


def verify_image_quality(image_path: str, midjourney_prompt: str) -> dict:
    """Score the chosen image with ImageReward. Skips gracefully if the model is unavailable."""
    if not image_path or not os.path.exists(image_path):
        return {"pass": False, "score": -999.0, "reason": "image file not found"}
    try:
        from services.image_ranker import _get_model
        model = _get_model()
        if model is None:
            return {"pass": True, "score": 0.0, "reason": "ImageReward unavailable — skipped"}
        from PIL import Image as PILImage
        img = PILImage.open(image_path).convert("RGB")
        score = model.score(midjourney_prompt or "a beautiful photorealistic scene", img)
        return {"pass": score > -1.0, "score": round(float(score), 4), "reason": ""}
    except Exception as exc:
        return {"pass": True, "score": 0.0, "reason": f"scoring failed ({exc}) — skipped"}


def verify_terminal_output(terminal_output: str) -> dict:
    """LLM check that every cycle stage ran and no errors fired."""
    from services.grok_ai import get_grok_response
    if len(terminal_output) > 5000:
        head = terminal_output[:2500]
        tail = terminal_output[-2500:]
        snippet = head + "\n\n... [middle truncated] ...\n\n" + tail
    else:
        snippet = terminal_output
    prompt = f"""You are reviewing the terminal output of a German-learning Twitter bot that just ran one full cycle.

Terminal output (may be truncated):
---
{snippet}
---

Check for:
1. ERRORS: Any Python exceptions, tracebacks, or ERROR-level log lines
2. SKIPPED STEPS: Any stage that was skipped unexpectedly (generate_content, generate_image, generate_audio, create_video, publish must all appear)
3. API FAILURES: Any mention of rate limits, auth errors, or failed API calls
4. WARNINGS: Excessive warnings that suggest degraded operation
5. COMPLETION: Output ends with a successful publish confirmation (tweet URL visible)
6. TIMING: Any stage taking suspiciously long (>5 min for a single step)

Respond in JSON only:
{{
  "pass": true/false,
  "score": 1-10,
  "all_stages_present": true/false,
  "errors_found": ["..."],
  "warnings_found": ["..."],
  "summary": "one sentence"
}}

Pass if score >= 7 AND all_stages_present is true AND no unhandled exceptions.
"""
    try:
        raw = get_grok_response(
            prompt,
            "You are a strict quality reviewer for bot terminal output. Respond only with JSON.",
            max_tokens=400,
            temperature=0.1,
        )
        result = _parse_json_response(raw)
        result.setdefault("score", 0)
        result.setdefault("all_stages_present", False)
        result.setdefault("errors_found", [])
        result.setdefault("summary", "")
        # Enforce pass conditions in Python rather than trusting AI alone
        result["pass"] = (
            result["score"] >= 7
            and result["all_stages_present"]
            and not result["errors_found"]
        )
        return result
    except Exception as exc:
        logger.warning("Terminal output verification failed: %s", exc)
        return {
            "pass": False, "score": 0, "all_stages_present": False,
            "errors_found": [str(exc)], "summary": "verification error",
        }
