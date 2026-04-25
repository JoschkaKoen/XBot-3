"""
services/history — load, save, and score post history.

These helpers were previously private (underscore-prefixed) inside
nodes/score.py but were imported across nodes (analyze, fetch_metrics,
generate_content) and the improve script. Moved here so node modules don't
depend on each other and the contract is explicit.

Persistence:
  - load_history()         → list of past posts (or [])
  - save_history(records)  → atomic write to data/post_history.json

Scoring:
  - compute_score(metrics) → weighted engagement score for a single post
  - tweet_age_hours(record)→ hours since posting (clamped to ≥6h)
  - normalized_score(record)→ engagement_score / age_hours
  - get_top_tweets(history, n)→ N best tweets by normalized score
"""

import logging
from datetime import datetime, timezone

from config import HISTORY_FILE
from utils.io import atomic_json_write, safe_json_read

logger = logging.getLogger("xbot.history")

_MIN_AGE_HOURS = 6   # avoid extreme normalized-score inflation for very fresh tweets


def compute_score(metrics: dict) -> float:
    """
    Weighted engagement score:
        likes + 3×reposts + 5×replies + 2×quotes + impressions/100
    """
    likes       = metrics.get("like_count", 0)
    reposts     = metrics.get("retweet_count", 0)
    replies     = metrics.get("reply_count", 0)
    quotes      = metrics.get("quote_count", 0)
    impressions = metrics.get("impression_count", 0)
    return round(likes + 3 * reposts + 5 * replies + 2 * quotes + impressions / 100, 2)


def tweet_age_hours(record: dict) -> float:
    """Hours since this tweet was posted; clamped to a minimum of 6 to avoid division inflation."""
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
    """Engagement score divided by age in hours — a fair per-hour rate."""
    return round(record.get("engagement_score", 0.0) / tweet_age_hours(record), 4)


def get_top_tweets(history: list, n: int = 3) -> list:
    """Return the N best tweets by normalized score, excluding score 0.0 and empty tweets."""
    qualifying = [
        r for r in history
        if r.get("engagement_score", 0.0) > 0.0 and r.get("full_tweet")
    ]
    qualifying.sort(key=normalized_score, reverse=True)
    return qualifying[:n]


def load_history() -> list:
    """Return the post-history list, or [] if the file is missing/corrupt."""
    return safe_json_read(HISTORY_FILE, default=[], logger=logger)


def save_history(history: list) -> None:
    """Atomically replace the post-history file."""
    atomic_json_write(HISTORY_FILE, history, ensure_ascii=False, indent=2)
