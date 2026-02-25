"""
Node: fetch_all_metrics

Runs at the START of every cycle, before content generation.

For every tweet in post_history.json:
  - Fetch current public metrics from the X API
  - If the tweet no longer exists (deleted / not found): remove it from history
  - If the tweet still exists: update its metrics and recompute its score

This means engagement scores grow over time as posts keep accumulating
likes / impressions, giving the strategy analyser ever-improving signal.
"""

import logging
import tweepy

from config import (
    X_BEARER_TOKEN,
    TWITTER_CONSUMER_KEY,
    TWITTER_CONSUMER_SECRET,
    TWITTER_ACCESS_TOKEN,
    TWITTER_ACCESS_TOKEN_SECRET,
)
from nodes.score import _load_history, _save_history, _compute_score
from utils.retry import with_retry
from utils.ui import stage_banner, ok, info as ui_info, warn as ui_warn

logger = logging.getLogger("german_bot.fetch_metrics")

_NOT_FOUND_CODES = {144, 34}


def _client() -> tweepy.Client:
    return tweepy.Client(
        bearer_token=X_BEARER_TOKEN or None,
        consumer_key=TWITTER_CONSUMER_KEY,
        consumer_secret=TWITTER_CONSUMER_SECRET,
        access_token=TWITTER_ACCESS_TOKEN,
        access_token_secret=TWITTER_ACCESS_TOKEN_SECRET,
    )


def _tweet_is_gone(exc: Exception) -> bool:
    msg = str(exc).lower()
    if "no data" in msg or "not found" in msg or "404" in msg:
        return True
    if hasattr(exc, "api_codes") and any(c in _NOT_FOUND_CODES for c in exc.api_codes):
        return True
    if hasattr(exc, "response") and getattr(exc.response, "status_code", None) == 404:
        return True
    return False


class _TweetGoneError(Exception):
    pass


@with_retry(max_attempts=3, base_delay=0.0, label="fetch_metrics")
def _fetch_one(client: tweepy.Client, tweet_id: str) -> dict:
    """Fetch public metrics for a single tweet. Raises _TweetGoneError if deleted."""
    response = client.get_tweet(
        id=tweet_id,
        tweet_fields=["public_metrics"],
    )
    if response.data is None:
        raise _TweetGoneError(f"Tweet {tweet_id} returned no data (possibly deleted).")
    return response.data.get("public_metrics", {})


# ── node ──────────────────────────────────────────────────────────────────────

def fetch_all_metrics(state: dict) -> dict:
    """
    Refresh metrics for every tweet in history.
    Deleted tweets are removed; existing ones get updated scores.
    """
    stage_banner(1)
    logger.info("Node: fetch_all_metrics")

    history = _load_history()
    if not history:
        ui_info("No posts in history yet — skipping metrics refresh.")
        logger.info("No post history found — nothing to refresh.")
        return state

    client = _client()
    total   = len(history)
    updated = 0
    deleted = 0
    kept_unchanged = 0

    n = "tweet" if total == 1 else "tweets"
    ui_info(f"Refreshing metrics for {total} {n} …")
    logger.info("Refreshing metrics for %d %s.", total, n)

    new_history = []
    for record in history:
        tweet_id = record.get("tweet_id", "")
        if not tweet_id:
            new_history.append(record)
            kept_unchanged += 1
            continue

        try:
            metrics = _fetch_one(client, tweet_id)
            score   = _compute_score(metrics)
            new_history.append({**record, "metrics": metrics, "engagement_score": score})
            updated += 1
            logger.debug(
                "Updated tweet %s: score=%.2f metrics=%s", tweet_id, score, metrics
            )

        except _TweetGoneError as exc:
            logger.info("Tweet %s no longer exists — removing from history. (%s)", tweet_id, exc)
            deleted += 1

        except Exception as exc:
            if _tweet_is_gone(exc):
                logger.info("Tweet %s appears deleted — removing. (%s)", tweet_id, exc)
                deleted += 1
            else:
                # API error (rate limit, network, etc.) — keep record, don't update
                logger.warning(
                    "Could not fetch metrics for tweet %s (%s) — keeping as-is.", tweet_id, exc
                )
                new_history.append(record)
                kept_unchanged += 1

    _save_history(new_history)

    parts = []
    if updated:       parts.append(f"{updated} updated")
    if deleted:       parts.append(f"{deleted} deleted")
    if kept_unchanged: parts.append(f"{kept_unchanged} unchanged")
    n = "tweet" if total == 1 else "tweets"
    ok(f"{total} {n} refreshed — {',  '.join(parts) if parts else 'nothing to do'}")
    logger.info(
        "Metrics refresh done: %d updated, %d deleted, %d unchanged (%d remaining).",
        updated, deleted, kept_unchanged, len(new_history),
    )

    return state
