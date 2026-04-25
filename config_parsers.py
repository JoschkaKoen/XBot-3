"""
config_parsers — pure parsing helpers for settings.env values.

Extracted from config.py so each helper can be tested in isolation. They never
read or mutate global config state — they take raw strings, return parsed
values, and (via the supplied logger) report invalid input.
"""

import logging
import re
from typing import Optional

_LOG = logging.getLogger(__name__)


def parse_strategy_update_interval(raw: Optional[str]) -> tuple[bool, int]:
    """
    Parse STRATEGY_UPDATE_INTERVAL_HOURS.

    Returns (metrics_and_strategy_updates_enabled, interval_hours).
    When enabled is False, the bot never calls the X API to refresh metrics and
    never re-runs LLM strategy analysis (same as leaving metrics_refreshed=False).

    Accepts: false / off / no / never / disabled (case-insensitive) → disabled.
    Accepts: plain integers, or simple expressions like 24*7 or 24 * 7 → hours.
    """
    if raw is None or not str(raw).strip():
        return True, 24
    s = str(raw).strip()
    low = s.lower()
    if low in ("false", "off", "no", "never", "disabled", "none"):
        return False, 24
    mul = re.fullmatch(r"^\s*(\d+)\s*\*\s*(\d+)\s*$", s)
    if mul:
        try:
            return True, int(mul.group(1)) * int(mul.group(2))
        except ValueError:
            pass
    try:
        return True, int(s)
    except ValueError:
        _LOG.warning(
            "Invalid STRATEGY_UPDATE_INTERVAL_HOURS=%r — using 24h. "
            "Use an integer, e.g. 24, 168, or false to disable metrics + strategy updates.",
            raw,
        )
        return True, 24


def parse_metrics_fetch_max(raw: Optional[str], analyze_last_n: int) -> int:
    """
    Max tweets to refresh per metrics run (X API calls). Empty/unset → max(analyze_last_n, 30).
    Set to 0 or 'all' / 'unlimited' for no cap (every row in post_history).
    """
    if raw is None or not str(raw).strip():
        return max(analyze_last_n, 30)
    s = str(raw).strip().lower()
    if s in ("0", "all", "unlimited", "none"):
        return 0
    return max(1, int(s))


def parse_use_trends_mode_cycle(raw: Optional[str]) -> list[str]:
    """
    Parse USE_TRENDS into a cycle of word-source modes.

    Tokens (case-insensitive):
      trends, true, 1, yes, on  → pick word from X trending topics
      pool                       → AI word pick + ephemeral random theme (German→English bank)
      false, 0, no, off, strategy → AI word pick + strategy next_topic / style

    Examples:
      "false"                         → ["strategy"]
      "true"                          → ["trends"]
      "trends,trends,pool,pool,trends" → five-step cycle
      "true,false,false,false"        → trends then three strategy steps (backward compatible)
    """
    if raw is None or not str(raw).strip():
        return ["strategy"]
    parts = [p.strip().lower() for p in str(raw).split(",") if p.strip()]
    if not parts:
        return ["strategy"]
    out: list[str] = []
    for p in parts:
        if p in ("trends", "true", "1", "yes", "on"):
            out.append("trends")
        elif p == "pool":
            out.append("pool")
        elif p in ("false", "0", "no", "off", "strategy"):
            out.append("strategy")
        else:
            _LOG.warning("Invalid USE_TRENDS token %r — treating as strategy.", p)
            out.append("strategy")
    return out


def parse_on_off_env(raw: Optional[str], default: bool = False) -> bool:
    """
    Parse a boolean env var value as true/false, yes/no, on/off, or 1/0.
    Unknown values fall back to *default*.
    """
    if raw is None or not str(raw).strip():
        return default
    s = str(raw).strip().lower()
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off"):
        return False
    return default


def parse_ktv_font_size(raw: Optional[str]) -> int:
    """
    KTV karaoke subtitle font size in px at a 720p output frame (standard HD reference).
    At other resolutions the size is scaled proportionally (e.g. ~53px at 480p Wan when
    set to 80). Clamped to a safe range (12–200).
    """
    if raw is None or not str(raw).strip():
        v = 80
    else:
        try:
            v = int(str(raw).strip())
        except ValueError:
            _LOG.warning("Invalid KTV_FONT_SIZE=%r — using 80.", raw)
            v = 80
    return max(12, min(200, v))


def parse_flag_colors(env_val: str, default: list) -> list:
    """Parse a comma-separated hex color string into a list of (R, G, B) tuples."""
    try:
        parts = [p.strip() for p in env_val.split(",") if p.strip()]
        result = [tuple(int(c[i:i+2], 16) for i in (0, 2, 4)) for c in parts]
        if len(result) >= 3:
            return result
    except (ValueError, IndexError):
        pass
    return default
