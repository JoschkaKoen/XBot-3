"""
LangGraph graph definition for the German Learning Bot.

New cycle order (metrics-first):
  fetch_all_metrics → analyze_and_improve → generate_content → generate_image
  → generate_audio → create_video → publish → score_and_store → wait → END

main.py runs this graph in a while-True loop, carrying strategy + cycle counter
forward between iterations via the state dict. SqliteSaver checkpointing means
each individual node is checkpointed; if the process crashes mid-cycle it
resumes from the last completed node on the next run.
"""

import os
import sqlite3
import sys
import logging
import subprocess
import time
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from state import BotState
import config
from config import CHECKPOINT_DB, logger as root_logger
from nodes import (
    fetch_all_metrics,
    analyze_and_improve,
    generate_content,
    generate_image,
    generate_audio,
    create_video,
    publish,
    score_and_store,
)

logger = logging.getLogger("german_bot.graph")

_PROJECT_DIR = Path(__file__).parent.resolve()


# ── auto-update from GitHub ───────────────────────────────────────────────────

def _check_for_update() -> None:
    """
    Fetch origin/main and, if new commits exist, pull and restart the process.

    Uses os.execv to replace the current process in-place (same args, no
    orphan processes).  Any error (network failure, git not available, etc.)
    is caught and logged so the bot continues normally.
    """
    if os.getenv("AUTO_UPDATE", "true").lower().strip() != "true":
        return

    try:
        from utils.ui import ok, warn as ui_warn

        # Fetch without merging
        subprocess.run(
            ["git", "fetch", "origin", "main"],
            cwd=str(_PROJECT_DIR),
            capture_output=True,
            timeout=30,
        )

        local = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()

        remote = subprocess.run(
            ["git", "rev-parse", "origin/main"],
            cwd=str(_PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()

        if local == remote:
            logger.info("Auto-update: already up to date (%s).", local[:7])
            return

        logger.info(
            "Auto-update: new commits on origin/main (%s → %s). Pulling …",
            local[:7], remote[:7],
        )
        ok(f"Update found ({local[:7]} → {remote[:7]}) — pulling and restarting …")

        pull = subprocess.run(
            ["git", "pull", "origin", "main"],
            cwd=str(_PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if pull.returncode != 0:
            logger.warning("Auto-update: git pull failed:\n%s", pull.stderr)
            ui_warn("git pull failed — continuing without update.")
            return

        logger.info("Auto-update: pull successful. Restarting …")
        os.execv(sys.executable, [sys.executable] + sys.argv)
        # os.execv replaces the process — code below is unreachable

    except Exception as exc:
        logger.warning("Auto-update check failed: %s — continuing.", exc)


# ── wait node ─────────────────────────────────────────────────────────────────

def wait_node(state: dict) -> dict:
    """
    Wait before the next cycle.  Runs (in order):
      1. Self-improvement (if ENABLE_SELF_IMPROVEMENT and cycle % N == 0)
      2. Countdown sleep for POST_INTERVAL_SECONDS (minus improvement time)
      3. _check_for_update() — if origin/main has new commits, pull and restart
    """
    from utils.ui import stage_banner, wait_countdown

    # Single-cycle mode: skip wait and auto-update
    if "--single-cycle" in sys.argv:
        logger.info("Single-cycle mode — skipping wait.")
        return state

    stage_banner(9)

    improvement_duration = 0

    if config.ENABLE_SELF_IMPROVEMENT:
        cycle = state.get("cycle", 0)
        if cycle > 0 and cycle % config.IMPROVEMENT_INTERVAL_CYCLES == 0:
            logger.info(
                "Cycle %d: triggering self-improvement (every %d cycles) …",
                cycle, config.IMPROVEMENT_INTERVAL_CYCLES,
            )
            try:
                improvement_env = os.environ.copy()
                improvement_env["OLD_BOT_PID"] = str(os.getpid())
                t0 = time.time()
                subprocess.run(
                    [sys.executable, "improve_with_claude_code.py"],
                    cwd=str(_PROJECT_DIR),
                    env=improvement_env,
                    timeout=3600,  # up to 1 hour for 3 attempts
                )
                improvement_duration = int(time.time() - t0)
                logger.info("Self-improvement run completed in %ds.", improvement_duration)
            except subprocess.TimeoutExpired:
                logger.warning("Self-improvement timed out after 3600s — continuing.")
            except Exception as exc:
                logger.warning("Self-improvement failed: %s", exc)

    remaining = max(config.POST_INTERVAL_SECONDS - improvement_duration, 60)
    logger.info("Waiting %ds before next cycle …", remaining)
    wait_countdown(remaining)
    logger.info("Wait complete.")

    _check_for_update()

    return state


# ── graph builder ─────────────────────────────────────────────────────────────
# Node order is fixed; each node reads/writes BotState.  To add a node or change
# order, update the edges below and the docstring at the top of this file.

def build_graph(checkpointer=None):
    builder = StateGraph(BotState)

    builder.add_node("fetch_all_metrics",   fetch_all_metrics)
    builder.add_node("analyze_and_improve", analyze_and_improve)
    builder.add_node("generate_content",    generate_content)
    builder.add_node("generate_image",      generate_image)
    builder.add_node("generate_audio",      generate_audio)
    builder.add_node("create_video",        create_video)
    builder.add_node("publish",             publish)
    builder.add_node("score_and_store",     score_and_store)
    builder.add_node("wait",                wait_node)

    builder.set_entry_point("fetch_all_metrics")
    builder.add_edge("fetch_all_metrics",   "analyze_and_improve")
    builder.add_edge("analyze_and_improve", "generate_content")
    builder.add_edge("generate_content",    "generate_image")
    builder.add_edge("generate_image",      "generate_audio")
    builder.add_edge("generate_audio",      "create_video")
    builder.add_edge("create_video",        "publish")
    builder.add_edge("publish",             "score_and_store")
    builder.add_edge("score_and_store",     "wait")
    builder.add_edge("wait",                END)

    kwargs = {"checkpointer": checkpointer} if checkpointer else {}
    return builder.compile(**kwargs)


def get_graph():
    """
    Return a compiled graph with SqliteSaver checkpointing.

    SqliteSaver requires a sqlite3 connection (not from_conn_string, which
    returns a context manager). We create a persistent connection here and
    keep it open for the lifetime of the process.
    """
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = build_graph(checkpointer=checkpointer)
    logger.info("LangGraph compiled | checkpoint DB: %s", CHECKPOINT_DB)
    return graph
