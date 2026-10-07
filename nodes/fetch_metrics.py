"""
Node: fetch_all_metrics

Runs at the START of every cycle, before content generation.

For every tweet in post_history.json (only when a strategy update can run — see below):
  - Fetch current public metrics from the X API
  - If the tweet no longer exists (deleted / not found): retain its learned-word history
  - If the tweet still exists: update its metrics and recompute its score

Metrics are **not** fetched unless both (1) the throttle interval has elapsed and
(2) there are at least **2** posts in history — the same precondition as LLM
strategy analysis. No standalone “metrics-only” refresh.

A timestamp sidecar (data/metrics_refresh.json) is used to throttle refreshes
to at most once every STRATEGY_UPDATE_INTERVAL_HOURS hours.

If STRATEGY_UPDATE_INTERVAL_HOURS is false/off/never/disabled in settings, metrics
refresh and strategy update are both skipped every cycle (no X API metric calls).

================================================================================
 STATE CONTRACT
================================================================================
  Reads from state:   (none — works from history file + config)
  Writes to state:    metrics_refreshed (bool)
  Side effects:       updates data/post_history.json, writes data/metrics_refresh.json
================================================================================
"""

import json
import logging
import os
from datetime import datetime, timezone

import tweepy

from config import (
    X_BEARER_TOKEN,
    TWITTER_CONSUMER_KEY,
    TWITTER_CONSUMER_SECRET,
    TWITTER_ACCESS_TOKEN,
    TWITTER_ACCESS_TOKEN_SECRET,
    STRATEGY_METRICS_UPDATES_ENABLED,
    STRATEGY_UPDATE_INTERVAL_HOURS,
    METRICS_FETCH_MAX_TWEETS,
    METRICS_REFRESH_FILE,
)
from services.history import load_history, save_history, compute_score
from utils.retry import with_retry
from utils.ui import stage_banner, ok, info as ui_info

logger = logging.getLogger("xbot.fetch_metrics")

_NOT_FOUND_CODES = {144, 34}

_REFRESH_STATE_PATH = METRICS_REFRESH_FILE


def _last_refresh_hours_ago() -> float:
    """Return how many hours ago metrics were last refreshed. Returns inf if never."""
    try:
        with open(_REFRESH_STATE_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        ts = datetime.fromisoformat(data["last_refresh"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - ts
        return delta.total_seconds() / 3600
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return float("inf")


def _record_refresh_timestamp() -> None:
    os.makedirs(os.path.dirname(_REFRESH_STATE_PATH), exist_ok=True)
    with open(_REFRESH_STATE_PATH, "w", encoding="utf-8") as fh:
        json.dump({"last_refresh": datetime.now(timezone.utc).isoformat()}, fh)


def _client() -> tweepy.Client:
    return tweepy.Client(
        bearer_token=X_BEARER_TOKEN or None,
        consumer_key=TWITTER_CONSUMER_KEY,
        consumer_secret=TWITTER_CONSUMER_SECRET,
        access_token=TWITTER_ACCESS_TOKEN,
        access_token_secret=TWITTER_ACCESS_TOKEN_SECRET,
    )


def _tweet_is_gone(exc: Exception) -> bool:
    """
    True only when *exc* is a reliable "tweet no longer exists" signal.

    Uses tweepy's typed 404 exception and structured API codes — NOT a substring
    search for "404"/"not found" in the message, which would falsely prune a
    *live* tweet from history whenever a rate-limit or network error happened to
    mention those tokens (e.g. a "404" in a URL). Deletion via a 200-with-no-data
    response is handled separately by _TweetGoneError.
    """
    if isinstance(exc, tweepy.NotFound):
        return True
    api_codes = getattr(exc, "api_codes", None)
    if api_codes and any(c in _NOT_FOUND_CODES for c in api_codes):
        return True
    if getattr(getattr(exc, "response", None), "status_code", None) == 404:
        return True
    return False


class _TweetGoneError(Exception):
    pass


# Retry only real API errors (rate limits, transient 5xx, network — all of which
# tweepy wraps as TweepyException). The _TweetGoneError sentinel is deliberately
# excluded so a deleted tweet propagates on the first call instead of burning two
# extra get_tweet calls per dead row.
@with_retry(max_attempts=3, base_delay=0.0, exceptions=(tweepy.TweepyException,), label="fetch_metrics")
def _fetch_one(client: tweepy.Client, tweet_id: str) -> dict:
    """Fetch public metrics for a single tweet. Raises _TweetGoneError if deleted."""
    try:
        response = client.get_tweet(
            id=tweet_id,
            tweet_fields=["public_metrics"],
        )
    except tweepy.NotFound as exc:
        # A 404 is a definitive deletion — convert to the sentinel so it (like the
        # 200-with-no-data case below) propagates immediately instead of being
        # retried as a TweepyException.
        raise _TweetGoneError(f"Tweet {tweet_id} returned 404 (deleted).") from exc
    if response.data is None:
        raise _TweetGoneError(f"Tweet {tweet_id} returned no data (possibly deleted).")
    return response.data.get("public_metrics", {})


# ── per-cycle lightweight refresh ─────────────────────────────────────────────

def _refresh_age_records(client, history, n):
    from services.age_metrics import due_records, snapshot
    selected = {str(r.get("tweet_id")) for r in due_records(history, n)}
    out = []
    for row in history:
        if str(row.get("tweet_id")) in selected:
            try:
                metrics = _fetch_one(client, row["tweet_id"])
                row = snapshot(row, metrics)
                row["engagement_score"] = compute_score(metrics)
            except Exception as exc:
                if _tweet_is_gone(exc):
                    row = {**row, "deleted_on_x": True}
                # Preserve the word ledger even when a tweet is gone. Deletion
                # never turns a taught word into a fresh word again.
                logger.warning("Age metrics unavailable for %s: %s", row.get("tweet_id"), type(exc).__name__)
        out.append(row)
    return out


def _fetch_cycle_metrics(n):
    history = load_history()
    if history:
        save_history(_refresh_age_records(_client(), history, n))


# ── secondary targets ─────────────────────────────────────────────────────────

def _target_client(spec) -> tweepy.Client:
    """A tweepy client for a secondary target's own X app."""
    creds = spec.account
    return tweepy.Client(
        bearer_token=creds.bearer_token or None,
        consumer_key=creds.consumer_key,
        consumer_secret=creds.consumer_secret,
        access_token=creds.access_token,
        access_token_secret=creds.access_token_secret,
    )


def _fetch_target_metrics(spec, n):
    from utils.io import safe_json_read, atomic_json_write
    history = safe_json_read(spec.history_file, default=[])
    if history and spec.account.is_complete():
        atomic_json_write(spec.history_file,
                          _refresh_age_records(_target_client(spec), history, n),
                          ensure_ascii=False, indent=2)


def _refresh_secondary_metrics(n: int) -> None:
    """Refresh recent metrics for every enabled secondary target.

    Isolated per target so one account's API trouble cannot stop the others or
    the primary cycle.
    """
    try:
        from services.targets import load_targets
        specs = load_targets()
    except Exception as exc:
        logger.warning("Metrics: could not load secondary targets (%s).", exc)
        return
    for spec in specs:
        try:
            _fetch_target_metrics(spec, n)
        except Exception as exc:
            logger.warning("Metrics [%s] failed (non-fatal): %s", spec.id, exc)


# ── node ──────────────────────────────────────────────────────────────────────

def fetch_all_metrics(state: dict) -> dict:
    """
    Refresh metrics for every tweet in history.
    Deleted tweets are removed; existing ones get updated scores.
    Skips the refresh if it was run within the last STRATEGY_UPDATE_INTERVAL_HOURS hours.
    If STRATEGY_METRICS_UPDATES_ENABLED is False (STRATEGY_UPDATE_INTERVAL_HOURS=false),
    never refreshes metrics or triggers strategy analysis.
    """
    stage_banner(1)
    logger.info("Node: fetch_all_metrics")

    # Per-cycle lightweight refresh — runs every cycle regardless of strategy gates.
    import config as _cfg
    if _cfg.METRICS_FETCH_PER_CYCLE > 0:
        try:
            _fetch_cycle_metrics(_cfg.METRICS_FETCH_PER_CYCLE)
        except Exception as exc:
            logger.warning("Per-cycle metrics refresh failed (non-fatal): %s", exc)
        # Secondary accounts get the same treatment on their own credentials —
        # without this they accumulate posts with no engagement data at all.
        _refresh_secondary_metrics(_cfg.METRICS_FETCH_PER_CYCLE)

    if not STRATEGY_METRICS_UPDATES_ENABLED:
        ui_info(
            "Strategy + metrics updates disabled (STRATEGY_UPDATE_INTERVAL_HOURS=false) — "
            "skipping X API metrics refresh and strategy re-analysis."
        )
        logger.info(
            "Metrics refresh disabled via STRATEGY_UPDATE_INTERVAL_HOURS — metrics_refreshed=False."
        )
        return {**state, "metrics_refreshed": False}

    hours_ago = _last_refresh_hours_ago()
    if hours_ago < STRATEGY_UPDATE_INTERVAL_HOURS:
        hours_left = STRATEGY_UPDATE_INTERVAL_HOURS - hours_ago
        ui_info(
            f"Metrics were refreshed {hours_ago:.1f}h ago — "
            f"skipping metrics + strategy update (next in {hours_left:.1f}h)."
        )
        logger.info(
            "Metrics refresh skipped: last run %.1fh ago, interval is %dh.",
            hours_ago,
            STRATEGY_UPDATE_INTERVAL_HOURS,
        )
        return {**state, "metrics_refreshed": False}

    history = load_history()
    if not history:
        ui_info("No posts in history yet — skipping metrics refresh.")
        logger.info("No post history found — nothing to refresh.")
        return {**state, "metrics_refreshed": False}

    # Strategy analysis only runs with ≥2 posts (see analyze_and_improve). Avoid
    # X API calls until then — metrics are only needed when a strategy update can run.
    if len(history) < 2:
        n = len(history)
        ui_info(
            f"{n} post{'s' if n != 1 else ''} in history — skipping metrics fetch until at least "
            "2 posts exist (metrics run only when a strategy update can run)."
        )
        logger.info(
            "Metrics refresh skipped: need >= 2 posts for strategy analysis; not calling X API."
        )
        return {**state, "metrics_refreshed": False}

    # Only the most recent posts need fresh metrics for strategy (see ANALYZE_LAST_N).
    # Default cap: max(ANALYZE_LAST_N, 30). Set METRICS_FETCH_MAX_TWEETS=0 for no cap.
    cap = METRICS_FETCH_MAX_TWEETS
    total_in_file = len(history)
    skip_prefix = 0
    if cap > 0 and total_in_file > cap:
        skip_prefix = total_in_file - cap
        ui_info(
            f"Refreshing metrics for the last {cap} of {total_in_file} posts "
            f"(METRICS_FETCH_MAX_TWEETS; older rows keep stored scores)."
        )
        logger.info(
            "Metrics cap: fetching %d newest of %d total.", cap, total_in_file
        )

    client = _client()
    total   = total_in_file
    updated = 0
    deleted = 0
    kept_unchanged = 0
    skipped_old = 0

    n = "tweet" if total == 1 else "tweets"
    if skip_prefix:
        ui_info(f"Calling X API for up to {cap} {n} …")
    else:
        ui_info(f"Refreshing metrics for {total} {n} …")
    logger.info("Refreshing metrics for %d %s.", total, n)

    new_history = []
    for idx, record in enumerate(history):
        if idx < skip_prefix:
            new_history.append(record)
            skipped_old += 1
            continue
        tweet_id = record.get("tweet_id", "")
        if not tweet_id:
            new_history.append(record)
            kept_unchanged += 1
            continue

        try:
            metrics = _fetch_one(client, tweet_id)
            score   = compute_score(metrics)
            new_history.append({**record, "metrics": metrics, "engagement_score": score})
            updated += 1
            logger.debug(
                "Updated tweet %s: score=%.2f metrics=%s", tweet_id, score, metrics
            )

        except _TweetGoneError as exc:
            logger.info("Tweet %s no longer exists — retaining word history. (%s)", tweet_id, exc)
            deleted += 1
            new_history.append({**record, "deleted_on_x": True})

        except Exception as exc:
            if _tweet_is_gone(exc):
                logger.info("Tweet %s appears deleted — retaining word history. (%s)", tweet_id, exc)
                deleted += 1
                new_history.append({**record, "deleted_on_x": True})
            else:
                # API error (rate limit, network, etc.) — keep record, don't update
                logger.warning(
                    "Could not fetch metrics for tweet %s (%s) — keeping as-is.", tweet_id, exc
                )
                new_history.append(record)
                kept_unchanged += 1

    save_history(new_history)
    _record_refresh_timestamp()

    parts = []
    if updated:       parts.append(f"{updated} updated")
    if deleted:       parts.append(f"{deleted} deleted")
    if kept_unchanged: parts.append(f"{kept_unchanged} unchanged")
    if skipped_old:    parts.append(f"{skipped_old} older rows not re-fetched")
    n = "tweet" if total == 1 else "tweets"
    ok(f"{total} {n} in file — {',  '.join(parts) if parts else 'nothing to do'}")
    logger.info(
        "Metrics refresh done: %d updated, %d deleted, %d unchanged, %d skipped (cap) (%d remaining).",
        updated, deleted, kept_unchanged, skipped_old, len(new_history),
    )

    return {**state, "metrics_refreshed": True}
