"""
Entry point for the German Learning X Bot.

Run with:
    source venv/bin/activate
    python main.py

    git add -A
    git commit -m "Improvements to the bot"
    git push
"""

import sys
import os
import time
import signal
import logging

# Ensure print() flushes immediately even when output is redirected
os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.stdout.reconfigure(line_buffering=True)

from config import setup_logging, AI_PROVIDER, SENTENCE_MODEL, STRATEGY_MODEL, USE_TRENDS, FUNNY_MODE
from utils.ui import startup_banner, cycle_banner, cycle_summary, err, warn
from services.image_ranker import warmup as _warmup_image_ranker


def _model_lines() -> list:
    """Build (label, model-name) pairs for the startup banner."""
    if AI_PROVIDER == "grok":
        _model_names = {
            "flagship":      "grok-4  (flagship)",
            "reasoning":     "grok-4-1-fast  (reasoning)",
            "non-reasoning": "grok-4-1-fast-non-reasoning",
        }
        tweet_model    = _model_names.get(SENTENCE_MODEL, SENTENCE_MODEL)
        strategy_model = _model_names.get(STRATEGY_MODEL, STRATEGY_MODEL)
        trend_model    = "grok-4-1-fast  (reasoning)"
        word_model     = "grok-4-1-fast-non-reasoning"
    else:
        tweet_model = strategy_model = trend_model = word_model = f"{AI_PROVIDER} (default)"

    lines = [
        ("Tweet generation:",  tweet_model),
        ("Strategy analysis:", strategy_model),
        ("Word selection:",    trend_model if USE_TRENDS else word_model),
    ]
    if USE_TRENDS:
        lines.append(("  (trend filtering):", trend_model))
    lines.append(("─" * 22, "─" * 30))   # visual separator
    lines.append(("Use trends:",          "ON" if USE_TRENDS else "off"))
    lines.append(("Funny mode:",          "ON 😄" if FUNNY_MODE else "off"))
    return lines

setup_logging()
logger = logging.getLogger("german_bot.main")

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
    logger.info("Loaded initial strategy: %s", {k: v for k, v in strategy.items() if k != "avoid_words"})
    return {
        "cycle":    0,
        "strategy": strategy,
        "error":    None,
    }


def main():
    from graph import get_graph

    startup_banner(_model_lines())
    logger.info("German Learning X Bot starting …")

    # Start loading ImageReward in the background immediately so it is ready
    # by the time the first image ranking call happens (~5+ minutes into cycle 1).
    _warmup_image_ranker()

    graph = get_graph()

    thread_id = "german_bot_main"
    config = {"configurable": {"thread_id": thread_id}}

    state = _initial_state()
    cycle = 0

    while not _shutdown:
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


if __name__ == "__main__":
    main()
