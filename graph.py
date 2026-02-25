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

import sqlite3
import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from state import BotState
from config import POST_INTERVAL_SECONDS, CHECKPOINT_DB, logger as root_logger
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


# ── wait node ─────────────────────────────────────────────────────────────────

def wait_node(state: dict) -> dict:
    """Sleep for POST_INTERVAL_SECONDS before the next cycle."""
    from utils.ui import stage_banner, wait_countdown
    stage_banner(9)
    logger.info("Waiting %ds before next cycle …", POST_INTERVAL_SECONDS)
    wait_countdown(POST_INTERVAL_SECONDS)
    logger.info("Wait complete.")
    return state


# ── graph builder ─────────────────────────────────────────────────────────────

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
