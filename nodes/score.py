"""
Node: score_and_store

Computes a weighted engagement score and appends the full record
(tweet text, word, metrics, score, timestamps) to data/post_history.json.

The actual load/save/scoring helpers live in services/history.py — this
module is just the LangGraph node wrapper.

================================================================================
 ENGAGEMENT SCORE FORMULA  (edit services/history.py:compute_score to change weights)
================================================================================
  score = likes + 3×reposts + 5×replies + 2×quotes + impressions/100

  Rationale for weights:
    - replies are valued most (active engagement, signals discussion)
    - reposts indicate shareability
    - quotes show enough interest to add commentary
    - impressions give a small baseline signal even with no interactions

  The age-normalised score (see normalized_score) divides by hours since
  posting and is used by the strategy analyser to fairly compare tweets
  of different ages.
================================================================================

================================================================================
 STATE CONTRACT
================================================================================
  Reads from state:   metrics, tweet_id, tweet_url, full_tweet, source_word,
                      article, cefr_level, example_sentence_source,
                      example_sentence_target, used_trend, pool_theme, cycle
  Writes to state:    engagement_score
  Side effects:       appends record to data/post_history.json
================================================================================
"""

import logging
from datetime import datetime, timezone

from services.history import compute_score, load_history, save_history
from utils.ui import stage_banner, ok

logger = logging.getLogger("xbot.score")


# ── node ──────────────────────────────────────────────────────────────────────

def score_and_store(state: dict) -> dict:
    stage_banner(9)
    logger.info("Node: score_and_store")

    metrics: dict = state.get("metrics", {})
    score: float = compute_score(metrics)

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
        "used_trend": state.get("used_trend", ""),
        "pool_theme": state.get("pool_theme", ""),
        "metrics": metrics,
        "engagement_score": score,
        "cycle": state.get("cycle", 0),
    }

    history = load_history()
    history.append(record)
    save_history(history)
    ok(f"Saved to history ({len(history)} records total)")
    logger.info("History saved (%d records total).", len(history))

    return {**state, "engagement_score": score}
