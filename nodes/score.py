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

_MIN_AGE_HOURS = 6   # avoid extreme inflation for very fresh tweets


def tweet_age_hours(record: dict) -> float:
    """Return how many hours ago this tweet was posted. Minimum 6 hours to avoid division inflation."""
    ts = record.get("timestamp", "")
    if not ts:
        return 24.0
    try:
        posted = datetime.fromisoformat(ts)
        if posted.tzinfo is None:
            posted = posted.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - posted).total_seconds() / 3600
        return round(max(age_hours, _MIN_AGE_HOURS), 2)
    except (ValueError, TypeError):
        return 24.0


def normalized_score(record: dict) -> float:
    """Engagement score divided by age in hours — a fair per-hour rate for comparison."""
    age_hours = tweet_age_hours(record)
    return round(record.get("engagement_score", 0.0) / age_hours, 4)


def get_top_tweets(history: list, n: int = 3) -> list:
    """Return the N highest age-normalized-scoring tweets from history, excluding score 0.0."""
    qualifying = [
        r for r in history
        if r.get("engagement_score", 0.0) > 0.0 and r.get("full_tweet")
    ]
    qualifying.sort(key=normalized_score, reverse=True)
    return qualifying[:n]


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
        "source_word": state.get("source_word", ""),
        "article": state.get("article", ""),
        "cefr_level": state.get("cefr_level", ""),
        "example_sentence_source": state.get("example_sentence_source", ""),
        "example_sentence_target": state.get("example_sentence_target", ""),
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
