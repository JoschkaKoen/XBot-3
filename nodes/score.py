"""
Node: score_and_store

Computes a weighted engagement score and appends the full record
(tweet text, word, metrics, score, timestamps) to a persistent JSON file.
"""

import json
import logging
import os
from datetime import datetime, timezone

from config import HISTORY_FILE
from utils.ui import stage_banner, ok

logger = logging.getLogger("german_bot.score")


def _compute_score(metrics: dict) -> float:
    """
    Weighted engagement score:
        likes + 3×reposts + 5×replies + 2×quotes + impressions/100
    """
    likes       = metrics.get("like_count", 0)
    reposts     = metrics.get("retweet_count", 0)
    replies     = metrics.get("reply_count", 0)
    quotes      = metrics.get("quote_count", 0)
    impressions = metrics.get("impression_count", 0)

    score = likes + 3 * reposts + 5 * replies + 2 * quotes + impressions / 100
    return round(score, 2)


def _load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as exc:
        logger.warning("Could not read history file: %s", exc)
        return []


def _save_history(history: list) -> None:
    os.makedirs(os.path.dirname(HISTORY_FILE) or ".", exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# ── node ──────────────────────────────────────────────────────────────────────

def score_and_store(state: dict) -> dict:
    stage_banner(8)
    logger.info("Node: record_post")

    metrics: dict = state.get("metrics", {})
    score: float = _compute_score(metrics)
    ok(f"Engagement score: {score:.2f}")
    logger.info("Engagement score: %.2f | metrics: %s", score, metrics)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tweet_id": state.get("tweet_id", ""),
        "tweet_url": state.get("tweet_url", ""),
        "full_tweet": state.get("full_tweet", ""),
        "german_word": state.get("german_word", ""),
        "article": state.get("article", ""),
        "cefr_level": state.get("cefr_level", ""),
        "example_sentence_de": state.get("example_sentence_de", ""),
        "example_sentence_en": state.get("example_sentence_en", ""),
        "metrics": metrics,
        "engagement_score": score,
        "cycle": state.get("cycle", 0),
    }

    history = _load_history()
    history.append(record)
    _save_history(history)
    ok(f"Saved to history ({len(history)} records total)")
    logger.info("History saved (%d records total).", len(history))

    return {**state, "engagement_score": score}
