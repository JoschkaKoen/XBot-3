"""
XBot-3 Language Learning Bot — Main Entry Point

================================================================================
 QUICK START
================================================================================
    cd /path/to/XBot-3
    source venv/bin/activate
    python main.py

    source venv/bin/activate && python main.py

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
from utils.errors import FatalProviderError
from utils.ui import (
    startup_banner,
    cycle_banner,
    err,
    format_elapsed,
    warn,
)


def _model_lines() -> list:
    """Build (label, model-name) pairs for the startup banner (reads live config)."""
    tweet_model    = _config.TWEET_MODEL
    strategy_model = _config.STRATEGY_MODEL
    trend_model    = _config.TREND_FILTER_MODEL
    word_model     = _config.WORD_PICK_MODEL

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
    ]
    _any_trends = any(m == "trends" for m in _config.USE_TRENDS_MODE_CYCLE)
    _all_trends = all(m == "trends" for m in _config.USE_TRENDS_MODE_CYCLE)
    if _all_trends:
        _word_sel = trend_model
    elif not _any_trends:
        _word_sel = word_model
    else:
        _word_sel = f"{word_model}  /  {trend_model}  (by cycle)"
    lines.append(("Word selection:", _word_sel))
    if _any_trends:
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
    def _word_source_label(m: str) -> str:
        return {"trends": "trends", "pool": "pool", "strategy": "off"}.get(m, m)

    use_trends_label = _cycle_label([_word_source_label(m) for m in _config.USE_TRENDS_MODE_CYCLE])
    lines.append(("Word source:", use_trends_label))
    if _any_trends:
        lines.append(("  Candidate limit:", f"{_config.TREND_CANDIDATE_LIMIT}  (top-{_config.TREND_CANDIDATE_LIMIT}, then AI fallback)"))
    lines.append(("Image style:",         image_style_label))
    lines.append(("Tweet style:",         tweet_style_label))
    _engine = _config.ENABLE_VIDEO
    if _engine in ("grok", "wan2.1"):
        freq_label = "every tweet" if _config.VIDEO_FREQUENCY <= 1 else f"every {_config.VIDEO_FREQUENCY} tweets"
        engine_display = "Grok Imagine" if _engine == "grok" else "Wan2.1 (local)"
        lines.append(("Video (I2V):", f"ON  ({freq_label} via {engine_display})"))
    else:
        lines.append(("Video (I2V):", "off"))

    # ── Output each cycle: which pairs, and exactly what happens to each tweet ──
    _secondary = list(getattr(_config, "SECONDARY_TARGETS", []) or [])
    lines.append(("─" * 22, "─" * 30))
    lines.append(("Output each cycle:", f"{1 + len(_secondary)} tweet(s)", "📤"))
    lines.append((
        f"  {_config.SOURCE_FLAG} {_config.SOURCE_LANGUAGE}→{_config.TARGET_LANGUAGE}:",
        "→ posted to X (primary account)",
        "🐦",
    ))
    for _s in _secondary:
        if _config.FANOUT_DRY_RUN:
            _dest = "→ DRY-RUN: built locally, NOT posted"
        elif not _s.account.is_complete():
            _dest = "→ ⚠ NOT posted (missing API keys)"
        else:
            _dest = f"→ posted to {getattr(_s, 'handle', '') or ('2nd account [' + _s.id + ']')}"
        if getattr(_s, "content_policy", ""):
            _dest += "  (China-compliance checked)"
        lines.append((
            f"  {_s.source_flag} {_s.source_language}→{_s.target_language}:",
            _dest,
            "🌐",
        ))
    if _secondary:
        lines.append(("Transcreation model:", _config.TRANSCREATION_MODEL))
        if any(getattr(s, "content_policy", "") for s in _secondary):
            lines.append(("Compliance model:", _config.CONTENT_SAFETY_MODEL))
        _ingest = (getattr(_config, "EXERCISE_INGEST_URL", "") or "").strip()
        if not _ingest:
            _web = "→ disabled (EXERCISE_INGEST_URL unset)"
        else:
            from urllib.parse import urlparse
            _host = urlparse(_ingest).netloc or _ingest
            _web = (f"→ DRY-RUN: not mirrored ({_host})" if _config.FANOUT_DRY_RUN
                    else f"→ mirrored to {_host}")
        lines.append(("eXercise website:", _web, "🌐"))
    return lines

setup_logging()
logger = logging.getLogger("xbot.main")

_shutdown = False


def _handle_signal(sig, frame):
    global _shutdown
    warn(f"Signal {sig} received — will stop after current cycle.")
    logger.info("Signal %s received — stopping after current cycle.", sig)
    _shutdown = True


signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def _last_posted_cycle() -> int:
    """Return the highest cycle number recorded in data/post_history.json, or 0."""
    from utils.io import safe_json_read
    hist = safe_json_read("data/post_history.json", default=[])
    if not isinstance(hist, list) or not hist:
        return 0
    try:
        return max(int(r.get("cycle", 0)) for r in hist if isinstance(r, dict))
    except (ValueError, TypeError):
        return 0


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

    # Discover styles from styles/ and validate the configured IMAGE_STYLE cycle
    # before doing anything else — fail-fast on a typo so the operator sees the
    # error immediately rather than mid-cycle.
    import styles
    styles.reload_styles()
    styles.validate_cycle(_config.IMAGE_STYLE_CYCLE)

    from graph import get_graph

    startup_banner(_model_lines())
    logger.info("%s for %s — Language Learning X Bot starting …", _config.SOURCE_LANGUAGE, _config.TARGET_LANGUAGE)

    graph = get_graph()

    thread_id = "german_bot_main"
    config = {"configurable": {"thread_id": thread_id}}

    state = _initial_state()
    cycle = _last_posted_cycle()
    if cycle:
        logger.info("Resuming from last posted cycle %d — next cycle will be %d.", cycle, cycle + 1)
    consecutive_failures = 0

    while not _shutdown:
        reload_settings()
        # Re-discover styles after settings reload so files added/edited between
        # cycles are picked up, then validate the (possibly updated) IMAGE_STYLE
        # cycle against the (possibly updated) registry.
        styles.reload_styles()
        styles.validate_cycle(_config.IMAGE_STYLE_CYCLE)
        max_consecutive = _config.MAX_CONSECUTIVE_FAILURES

        # If IMAGE_PROVIDER=z-image-turbo, make sure ComfyUI is running.
        # This spawns it in the background (non-blocking) so it has time to
        # warm up before the image-generation step arrives later in the cycle.
        if _config.IMAGE_PROVIDER == "z-image-turbo":
            from services.zit_image import ensure_comfyui_running
            ensure_comfyui_running()

        cycle += 1
        cycle_banner(cycle)
        logger.info("Starting cycle %d …", cycle)
        from services.usage import reset_run_usage
        reset_run_usage()
        t_cycle = time.perf_counter()

        try:
            # Pass cycle start time in state so wait_node can compute elapsed.
            # Reset secondary_results so stale checkpoint data from a prior
            # dry-run or a different cycle never blocks the fan-out.
            state["_cycle_start_time"] = t_cycle
            state["cycle"] = cycle
            state["secondary_results"] = []
            result = graph.invoke(state, config=config)

            # A cycle that produced no posted tweet (e.g. the image provider was
            # unavailable so publish was skipped) still runs LLM/provider calls and
            # burns credits. Count it toward the consecutive-failure watchdog so a
            # persistent outage stops the bot instead of looping forever posting
            # nothing — but never crash, since the pipeline handled it gracefully.
            if result.get("tweet_id"):
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                warn(
                    f"Cycle {cycle} posted no tweet "
                    f"({consecutive_failures}/{max_consecutive}) — provider may be unavailable."
                )
                logger.warning(
                    "Cycle %d completed without posting a tweet (%d/%d consecutive).",
                    cycle, consecutive_failures, max_consecutive,
                )
                if consecutive_failures >= max_consecutive:
                    err(
                        f"Stopping bot — {consecutive_failures} cycles in a row posted nothing. "
                        "Fix the image/video provider before restarting."
                    )
                    logger.critical(
                        "Aborting after %d consecutive no-post cycles.", consecutive_failures
                    )
                    state["error"] = "no tweet posted"
                    break

            state = {
                "cycle":    cycle,
                "strategy": result.get("strategy", state["strategy"]),
                "error":    None,
            }
            # cycle_summary and cycle_cost_summary are printed inside wait_node
            # so they survive os.execv() restarts triggered by _check_for_update().

        except KeyboardInterrupt:
            warn("Interrupted — stopping.")
            logger.info("KeyboardInterrupt — stopping.")
            break
        except FatalProviderError as exc:
            elapsed = time.perf_counter() - t_cycle
            err(f"FATAL provider error after {format_elapsed(elapsed)} — stopping bot: {exc}")
            logger.critical("Cycle %d — FatalProviderError: %s", cycle, exc)
            state["error"] = str(exc)
            break
        except Exception as exc:
            elapsed = time.perf_counter() - t_cycle
            consecutive_failures += 1
            err(
                f"Cycle {cycle} failed after {format_elapsed(elapsed)} "
                f"({consecutive_failures}/{max_consecutive}): {exc}"
            )
            logger.exception(
                "Cycle %d failed after %.1fs: %s", cycle, elapsed, exc
            )
            if consecutive_failures >= max_consecutive:
                err(
                    f"Stopping bot — {consecutive_failures} consecutive failures. "
                    "Fix the underlying issue before restarting."
                )
                logger.critical(
                    "Aborting after %d consecutive failures.", consecutive_failures
                )
                state["error"] = str(exc)
                break
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

    import styles
    styles.reload_styles()
    styles.validate_cycle(_config.IMAGE_STYLE_CYCLE)

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
            # Secondary fan-out posts (e.g. English→Chinese) are REAL tweets on
            # other accounts. The self-improvement engine reads these so a failed
            # verification can delete them too — otherwise they leak (stay live).
            "secondary_results":   result.get("secondary_results", []),
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
