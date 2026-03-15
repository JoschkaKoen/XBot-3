"""
XBot-3 Language Learning Bot — Main Entry Point

================================================================================
 QUICK START
================================================================================
    cd /path/to/XBot-3
    source venv/bin/activate
    python main.py

Run one cycle only (for testing or improvement engine):
    python main.py --single-cycle

================================================================================
 BOT FLOW
================================================================================
The bot runs an endless loop, each cycle:
    1. fetch_all_metrics    → refresh tweet engagement metrics
    2. analyze_and_improve → update strategy based on past performance
    3. generate_content    → pick word + write tweet(s)
    4. generate_image      → create visual for the tweet
    5. generate_audio      → text-to-speech for the sentence
    6. create_video        → combine image + audio → MP4
    7. publish             → post to X/Twitter
    8. score_and_store    → record engagement score
    9. wait                → sleep until next cycle (or run self-improvement)

================================================================================
"""

import json
import sys
import os
import time
import signal
import logging

# Ensure print() flushes immediately even when output is redirected
os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.stdout.reconfigure(line_buffering=True)

import config as _config
from config import setup_logging, reload_settings
from utils.ui import startup_banner, cycle_banner, cycle_summary, err, warn


def _model_lines() -> list:
    """Build (label, model-name) pairs for the startup banner (reads live config)."""
    if _config.AI_PROVIDER == "grok":
        _model_names = {
            "flagship":      "grok-4  (flagship)",
            "reasoning":     "grok-4-1-fast  (reasoning)",
            "non-reasoning": "grok-4-1-fast-non-reasoning",
        }
        tweet_model    = _model_names.get(_config.TWEET_MODEL, _config.TWEET_MODEL)
        strategy_model = _model_names.get(_config.STRATEGY_MODEL, _config.STRATEGY_MODEL)
        trend_model    = "grok-4-1-fast  (reasoning)"
        word_model     = "grok-4-1-fast-non-reasoning"
    else:
        tweet_model = strategy_model = trend_model = word_model = f"{_config.AI_PROVIDER} (default)"

    lines = [
        # ── Language pair ──────────────────────────────────────────────────────
        (
            "Source lang:",
            f"{_config.SOURCE_FLAG}  {_config.SOURCE_LANGUAGE} ({_config.SOURCE_LANGUAGE_CODE})",
            "🌐",
        ),
        (
            "Target lang:",
            f"{_config.TARGET_FLAG}  {_config.TARGET_LANGUAGE} ({_config.TARGET_LANGUAGE_CODE})",
            "🌐",
        ),
        (
            "Trends country:",
            _config.TRENDS_COUNTRY,
            "📈",
        ),
        (
            "Source flag colors:",
            _config.SOURCE_FLAG_COLORS,
            "🎨",
        ),
        (
            "Target flag colors:",
            _config.TARGET_FLAG_COLORS,
            "🎨",
        ),
        ("─" * 22, "─" * 30),
        # ── AI models ──────────────────────────────────────────────────────────
        ("Tweet generation:",  tweet_model),
        ("Strategy analysis:", strategy_model),
        ("Word selection:",    trend_model if _config.USE_TRENDS else word_model),
    ]
    if _config.USE_TRENDS:
        lines.append(("  (trend filtering):", trend_model))

    from collections import Counter

    def _cycle_label(cycle_list: list) -> str:
        if len(cycle_list) == 1:
            return cycle_list[0]
        if len(cycle_list) <= 4:
            return "  ↺  ".join(cycle_list) + "  (cycle)"
        counts = Counter(cycle_list)
        parts = [f"{s} ({n}×)" for s, n in counts.most_common()]
        return "  ↺  ".join(parts) + f"  ({len(cycle_list)}-step cycle)"

    image_style_label = _cycle_label(_config.IMAGE_STYLE_CYCLE)
    tweet_style_label = _cycle_label(_config.TWEET_STYLE_CYCLE)

    lines.append(("─" * 22, "─" * 30))   # visual separator
    lines.append(("Use trends:",          "ON" if _config.USE_TRENDS else "off"))
    if _config.USE_TRENDS:
        lines.append(("  Candidate limit:", f"{_config.TREND_CANDIDATE_LIMIT}  (top-{_config.TREND_CANDIDATE_LIMIT}, then AI fallback)"))
    lines.append(("Image style:",         image_style_label))
    lines.append(("Tweet style:",         tweet_style_label))
    _engine = _config.ENABLE_VIDEO
    if _engine in ("grok", "wan"):
        freq_label = "every tweet" if _config.VIDEO_FREQUENCY <= 1 else f"every {_config.VIDEO_FREQUENCY} tweets"
        engine_display = "Grok Imagine" if _engine == "grok" else "Wan2.1 (local)"
        lines.append(("Video (I2V):", f"ON  ({freq_label} via {engine_display})"))
    else:
        lines.append(("Video (I2V):", "off"))
    return lines

setup_logging()
logger = logging.getLogger("lang_bot.main")

_shutdown = False


def _handle_signal(sig, frame):
    global _shutdown
    warn(f"Signal {sig} received — will stop after current cycle.")
    logger.info("Signal %s received — stopping after current cycle.", sig)
    _shutdown = True


signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def _initial_state() -> dict:
    """
    Build the initial state for the first cycle.
    Load strategy from data/strategy.json if it exists so the bot
    resumes with its last-known strategy after a restart.
    """
    from nodes.analyze import load_strategy
    strategy = load_strategy()
    logger.info("Loaded strategy:")
    logger.info("  topic : %s", strategy.get("next_topic", "—"))
    logger.info("  style : %s", strategy.get("style", "—"))
    logger.info("  CEFR  : %s", strategy.get("preferred_cefr", "—"))
    return {
        "cycle":    0,
        "strategy": strategy,
        "error":    None,
    }


def main():
    _config.resolve_language_config()

    from graph import get_graph

    startup_banner(_model_lines())
    logger.info("%s for %s — Language Learning X Bot starting …", _config.SOURCE_LANGUAGE, _config.TARGET_LANGUAGE)

    graph = get_graph()

    thread_id = "german_bot_main"
    config = {"configurable": {"thread_id": thread_id}}

    state = _initial_state()
    cycle = 0

    while not _shutdown:
        reload_settings()
        cycle += 1
        cycle_banner(cycle)
        logger.info("Starting cycle %d …", cycle)

        try:
            result = graph.invoke(state, config=config)

            state = {
                "cycle":    cycle,
                "strategy": result.get("strategy", state["strategy"]),
                "error":    None,
            }

            tweet_url = result.get("tweet_url", "n/a")
            score     = result.get("engagement_score", 0.0)
            cycle_summary(cycle, tweet_url, score)
            logger.info("Cycle %d complete. url=%s score=%.2f", cycle, tweet_url, score)

        except KeyboardInterrupt:
            warn("Interrupted — stopping.")
            logger.info("KeyboardInterrupt — stopping.")
            break
        except Exception as exc:
            err(f"Cycle {cycle} failed: {exc}")
            logger.exception("Cycle %d failed: %s", cycle, exc)
            warn("Waiting 60s before retrying …")
            time.sleep(60)
            state["error"] = str(exc)

    logger.info("Bot shut down after %d cycle(s).", cycle)
    print("\nBot stopped.\n")


def _single_cycle() -> None:
    """
    Run exactly ONE full cycle (including posting a real tweet), write results
    to data/test_cycle_output.json, then exit.

    Used by the self-improvement engine to verify that improved code works.
    Uses a fresh UUID thread ID each time to avoid resuming a previous run's
    checkpointed state.
    """
    import uuid
    from graph import get_graph

    setup_logging()
    _config.resolve_language_config()

    graph = get_graph()
    thread_id = f"single_cycle_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    state = _initial_state()

    output_path = os.path.join(os.path.dirname(__file__), "data", "test_cycle_output.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        result = graph.invoke(state, config=config)
        output = {
            "success":             True,
            "tweet_id":            result.get("tweet_id", ""),
            "tweet_url":           result.get("tweet_url", ""),
            "tweet_text":              result.get("full_tweet", ""),
            "source_word":             result.get("source_word", ""),
            "cefr_level":              result.get("cefr_level", ""),
            "example_sentence_source": result.get("example_sentence_source", ""),
            "image_path":          result.get("image_path", ""),
            "midjourney_prompt":   result.get("midjourney_prompt", ""),
            "video_path":          result.get("video_path", ""),
            "errors":              [],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        sys.exit(0)

    except Exception as exc:
        output = {"success": False, "errors": [str(exc)]}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    if "--single-cycle" in sys.argv:
        _single_cycle()
    else:
        main()
