"""
Console UI helpers — pretty output for the terminal.
All functions use print() directly so they never appear in the log file.
"""

import sys
import time
import shutil
import config

# ── ANSI colours ──────────────────────────────────────────────────────────────
_R      = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_BLUE   = "\033[94m"
_GRAY   = "\033[90m"
_WHITE  = "\033[97m"

def _w() -> int:
    return min(shutil.get_terminal_size((72, 24)).columns, 72)


# ── Stage metadata ─────────────────────────────────────────────────────────────
def _stages() -> dict:
    src = config.SOURCE_LANGUAGE
    return {
        1: ("📊", "Refreshing metrics for all past tweets"),
        2: ("🧠", "Analysing & improving strategy"),
        3: ("✍️ ", f"Crafting {src} vocabulary content"),
        4: ("🎨", "Generating image"),
        5: ("🎙️ ", f"Generating {src} TTS audio"),
        6: ("🎬", "Rendering video"),
        7: ("🎞️ ", "Frame interpolation (optional)"),
        8: ("📤", "Posting to X"),
        9: ("🏅", "Recording new post"),
        10: ("⏳", "Waiting before next cycle"),
    }

_TOTAL = 10


# ── Public helpers ─────────────────────────────────────────────────────────────

def startup_banner(model_lines: list = None):
    """
    model_lines: list of (label, value) or (label, value, icon) tuples.
    A label starting with "─" renders as a full-width divider line.
    The icon defaults to "🤖" when not supplied.
    """
    w = _w()
    src = config.SOURCE_LANGUAGE.upper()
    tgt = config.TARGET_LANGUAGE.upper()
    flag = config.SOURCE_FLAG
    print()
    print(f"{_CYAN}{_BOLD}{'═' * w}{_R}")
    print(f"{_CYAN}{_BOLD}  {flag}  {src} FOR {tgt} — LANGUAGE LEARNING X BOT{_R}")
    if model_lines:
        print(f"{_CYAN}{'─' * w}{_R}")
        for row in model_lines:
            label = row[0]
            if label.startswith("─"):
                print(f"{_CYAN}{'─' * w}{_R}")
                continue
            value = row[1]
            icon  = row[2] if len(row) > 2 else "🤖"
            print(f"{_CYAN}  {icon}  {_BOLD}{label:<22}{_R}{_CYAN}{value}{_R}")
    print(f"{_CYAN}{_BOLD}{'═' * w}{_R}")
    print()


def cycle_banner(cycle: int):
    w = _w()
    print()
    print(f"{_BOLD}{'━' * w}{_R}")
    print(f"{_BOLD}  🔄  CYCLE {cycle}{_R}")
    print(f"{_BOLD}{'━' * w}{_R}")
    print()


def stage_banner(step: int):
    icon, name = _stages().get(step, ("▶", f"Step {step}"))
    w = _w()
    label = f"  {icon}  [{step}/{_TOTAL}]  {name.upper()}"
    print()
    print(f"{_CYAN}{'─' * w}{_R}")
    print(f"{_CYAN}{_BOLD}{label}{_R}")
    print(f"{_CYAN}{'─' * w}{_R}")


def tweet_box(tweet_text: str):
    """Print the assembled tweet inside a bordered box, with no added indentation
    so spacing and emojis appear exactly as they will be posted."""
    w = _w()
    print()
    print(f"{_GREEN}{_BOLD}┌─ TWEET READY {'─' * max(w - 16, 2)}┐{_R}")
    for line in tweet_text.split("\n"):
        print(f"{_GREEN}│{_R} {line}")
    print(f"{_GREEN}{_BOLD}└{'─' * (w - 2)}┘{_R}")
    print()


def wait_countdown(seconds: int):
    """Live countdown bar during the wait period."""
    h_total = seconds // 3600
    m_total = (seconds % 3600) // 60
    bar_width = 28
    print()
    print(f"{_YELLOW}{_BOLD}⏳  Waiting {h_total}h {m_total:02d}m before next cycle …{_R}")
    print()

    end = time.time() + seconds
    try:
        while True:
            remaining = end - time.time()
            if remaining <= 0:
                break
            elapsed = seconds - remaining
            done = int(elapsed / seconds * bar_width)
            bar = f"{'█' * done}{'░' * (bar_width - done)}"
            h = int(remaining // 3600)
            m = int((remaining % 3600) // 60)
            s = int(remaining % 60)
            print(
                f"\r  {_CYAN}[{bar}]{_R}  {h:02d}:{m:02d}:{s:02d} remaining  ",
                end="", flush=True
            )
            time.sleep(1)
    except KeyboardInterrupt:
        print()
        raise

    print(f"\r  {_CYAN}[{'█' * bar_width}]{_R}  00:00:00  {_GREEN}Done!{_R}          ")
    print()


def ok(msg: str):
    print(f"  {_GREEN}✅  {msg}{_R}")


def info(msg: str):
    print(f"  {_GRAY}ℹ   {msg}{_R}")


def warn(msg: str):
    print(f"  {_YELLOW}⚠   {msg}{_R}")


def err(msg: str):
    print(f"  {_RED}✖   {msg}{_R}", file=sys.stderr)


def cycle_summary(cycle: int, tweet_url: str, score: float):
    w = _w()
    print()
    print(f"{_GREEN}{_BOLD}{'═' * w}{_R}")
    print(f"{_GREEN}{_BOLD}  ✅  CYCLE {cycle} COMPLETE{_R}")
    print(f"{_GREEN}  🔗  {tweet_url}{_R}")
    print(f"{_GREEN}  📈  Engagement score: {score:.2f}{_R}")
    print(f"{_GREEN}{_BOLD}{'═' * w}{_R}")
    print()
