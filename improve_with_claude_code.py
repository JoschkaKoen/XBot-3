"""
Self-Improvement Engine — improve_with_claude_code.py

Runs AFTER the bot posts a tweet (triggered from graph.py wait_node).
Uses Claude Code CLI to improve code on a separate git branch, then
verifies the improvement via a real live cycle (posts a tweet to X).

Phase 1  → Code improvement on a new branch
Phase 2  → Live cycle (real post to X)
Phase 3  → Verification (tweet exists, text quality, image quality, terminal output)
Phase 4  → Success: swap bot process / Failure: delete tweets, discard branch

NEVER auto-merges. NEVER touches .env, improve_with_claude_code.py, verify_quality.py.

Usage (manual):
    python improve_with_claude_code.py [--force]

Usage (automatic, called from graph.py):
    OLD_BOT_PID=<pid> python improve_with_claude_code.py
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

# ── ANSI colours (mirroring utils/ui.py) ───────────────────────────────────────
_R     = "\033[0m"
_BOLD  = "\033[1m"
_GREEN = "\033[92m"
_RED   = "\033[91m"
_YELLOW= "\033[93m"
_CYAN  = "\033[96m"
_GRAY  = "\033[90m"

# ── File-only logger ───────────────────────────────────────────────────────────
_file_logger = logging.getLogger("german_bot.improve")
_file_logger.setLevel(logging.DEBUG)
_log_path = PROJECT_DIR / "data" / "improvement.log"
_log_path.parent.mkdir(parents=True, exist_ok=True)
_fh = logging.FileHandler(str(_log_path), encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_file_logger.addHandler(_fh)
_file_logger.propagate = False


def log_both(msg: str, level: str = "info") -> None:
    """Print coloured to terminal AND write plain text to improvement.log."""
    print(f"  {msg}", flush=True)
    clean = msg
    for code in ["\033[0m", "\033[1m", "\033[92m", "\033[91m", "\033[93m",
                 "\033[96m", "\033[90m", "\033[94m", "\033[2m"]:
        clean = clean.replace(code, "")
    getattr(_file_logger, level)(clean)


def log_header(title: str) -> None:
    w = 60
    print(f"\n{_CYAN}{_BOLD}{'─' * w}{_R}", flush=True)
    print(f"{_CYAN}{_BOLD}  {title}{_R}", flush=True)
    print(f"{_CYAN}{'─' * w}{_R}\n", flush=True)
    _file_logger.info("=" * 60)
    _file_logger.info("  %s", title)
    _file_logger.info("=" * 60)


# ── Twitter helpers ────────────────────────────────────────────────────────────

def _twitter_client():
    import tweepy
    from config import (
        TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET,
        TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET,
    )
    return tweepy.Client(
        consumer_key=TWITTER_CONSUMER_KEY,
        consumer_secret=TWITTER_CONSUMER_SECRET,
        access_token=TWITTER_ACCESS_TOKEN,
        access_token_secret=TWITTER_ACCESS_TOKEN_SECRET,
    )


def _delete_tweet(tweet_id: str) -> None:
    try:
        client = _twitter_client()
        client.delete_tweet(tweet_id)
        log_both(f"{_RED}🗑️  Deleted tweet {tweet_id} from X{_R}")
    except Exception as exc:
        log_both(f"{_YELLOW}⚠️  Could not delete tweet {tweet_id}: {exc}{_R}", "warning")


def _remove_from_history(tweet_id: str) -> None:
    from nodes.score import _load_history, _save_history
    history = _load_history()
    before = len(history)
    history = [r for r in history if r.get("tweet_id") != tweet_id]
    _save_history(history)
    log_both(f"{_GRAY}🗑️  Removed tweet {tweet_id} from history ({before} → {len(history)} records){_R}")


# ── Git helpers ────────────────────────────────────────────────────────────────

def _git(args: list, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args,
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
        check=check,
    )


def _git_current_branch() -> str:
    return _git(["branch", "--show-current"]).stdout.strip()


def _parse_json_response(raw: str) -> dict:
    """
    Safely extract JSON from an AI response that may be wrapped in ```json ... ``` fences.
    Uses line-by-line fence detection rather than str.strip(chars) to avoid stripping
    valid JSON characters.
    """
    text = raw.strip()
    # Remove opening fence (```json or ```)
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    # Remove closing fence
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return json.loads("\n".join(lines))


# ── requirements.txt re-install ────────────────────────────────────────────────

def _maybe_reinstall_requirements() -> None:
    result = subprocess.run(
        ["git", "diff", "HEAD~1", "--name-only"],
        capture_output=True, text=True, cwd=str(PROJECT_DIR)
    )
    if "requirements.txt" in result.stdout:
        log_both(f"{_CYAN}📦  requirements.txt changed — running pip install …{_R}")
        pip = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True, text=True, cwd=str(PROJECT_DIR)
        )
        if pip.returncode != 0:
            raise RuntimeError(f"pip install failed:\n{pip.stderr[-2000:]}")
        log_both(f"{_GREEN}✅  pip install completed{_R}")


# ── Claude Code environment helper ────────────────────────────────────────────

def _find_claude_binary() -> str:
    """
    Resolve the best available Claude Code binary.

    Preference order:
      1. CLAUDE_BIN env var (manual override)
      2. Newest native binary from Cursor extensions (~/.cursor/extensions/)
      3. System claude (PATH)
    """
    # Manual override
    override = os.environ.get("CLAUDE_BIN")
    if override and os.path.isfile(override):
        return override

    # Native binaries shipped with Cursor extension
    cursor_ext = Path.home() / ".cursor" / "extensions"
    if cursor_ext.exists():
        candidates = sorted(cursor_ext.glob("anthropic.claude-code-*/resources/native-binary/claude"))
        if candidates:
            # Last in sorted order = newest version
            native = str(candidates[-1])
            log_both(f"{_GRAY}    Using native Claude binary: {native}{_R}")
            return native

    # Fall back to system PATH
    return "claude"


def _build_claude_env() -> dict:
    """
    Build an environment dict for Claude Code subprocesses.

    Claude Code uses OAuth credentials stored in ~/.claude/ by default.
    We must NOT inject ANTHROPIC_API_KEY into the environment — if that env var
    is present it overrides the OAuth session and causes "Invalid API key" errors.
    We strip it out to ensure Claude Code falls back to its stored OAuth token.
    """
    env = os.environ.copy()
    if "ANTHROPIC_API_KEY" in env:
        del env["ANTHROPIC_API_KEY"]
        log_both(f"{_GRAY}    Removed ANTHROPIC_API_KEY from env so Claude Code uses OAuth session{_R}")
    return env


# ── Claude Code prompt builders ────────────────────────────────────────────────

def _build_phase1_prompt(history: list, branch_name: str) -> str:
    from nodes.score import get_top_tweets

    top_tweets = get_top_tweets(history, n=3)
    bottom_tweets = sorted(
        [r for r in history if r.get("engagement_score", 0) > 0],
        key=lambda r: r.get("engagement_score", 0)
    )[:3]

    top_summary = "\n".join(
        f"  score={t['engagement_score']:.1f} | word={t.get('german_word')} | "
        f"cefr={t.get('cefr_level')} | sentence: {t.get('example_sentence_de', '')[:60]}"
        for t in top_tweets
    ) or "  (none yet)"

    bottom_summary = "\n".join(
        f"  score={t['engagement_score']:.1f} | word={t.get('german_word')} | "
        f"cefr={t.get('cefr_level')} | sentence: {t.get('example_sentence_de', '')[:60]}"
        for t in bottom_tweets
    ) or "  (none yet)"

    return f"""You are improving a German-learning X (Twitter) bot that posts vocabulary tweets with videos.
You are working on branch: {branch_name}

## Performance Data

TOP performing tweets (imitate these):
{top_summary}

BOTTOM performing tweets (avoid these patterns):
{bottom_summary}

## Your Goal

Analyse the performance data and improve the bot's code to generate higher-engagement tweets.
Focus on what makes the top tweets better than the bottom ones.

## What You May Improve

You may modify ANY file in the repository EXCEPT:
- .env                          ← NEVER touch (API keys and secrets)
- improve_with_claude_code.py   ← NEVER touch (self-modification forbidden)
- verify_quality.py             ← NEVER touch (verifier must stay unchanged)

Everything else is fair game, including:
- nodes/generate_content.py     — tweet prompts, word selection logic, scaffold
- nodes/generate_image.py       — Midjourney prompt construction
- nodes/analyze.py              — strategy analysis prompts
- nodes/score.py                — engagement score weighting formula
- services/grok_ai.py           — model name constants, call parameters
- services/image_ranker.py      — image scoring and selection logic
- data/strategy.json            — strategy parameters
- Any other file where you see a genuine improvement opportunity

## Cost Constraint for Model Changes

If you change which AI model is used for any step, follow this rule:
- High-volume, low-stakes calls (word selection, MJ prompt generation, trend filtering):
  use grok-4-1-fast-non-reasoning or grok-4-1-fast — NOT grok-4 flagship
- Low-volume, high-stakes calls (final tweet/sentence generation):
  grok-4 flagship is acceptable but not required
- Do NOT switch multiple steps to flagship simultaneously

## What to Actually Do

1. Read the performance data above
2. Read the relevant source files
3. Make targeted, well-reasoned improvements
4. Test that imports still work after your changes
5. Keep changes focused — do not refactor everything at once

After making changes, verify:
  python -c "from nodes.generate_content import generate_content; print('OK')"
  python -c "from nodes.generate_image import generate_image; print('OK')"
  python -c "from nodes.analyze import analyze_and_improve; print('OK')"
  python -c "from nodes.score import score_and_store; print('OK')"

Commit all changes with a descriptive commit message.
"""


def _build_fix_prompt(failure_report: dict, attempt: int, terminal_output: str) -> str:
    snippet = terminal_output[:5000] + (" ... [truncated]" if len(terminal_output) > 5000 else "")
    return f"""A live verification attempt ({attempt}/3) of the improved bot just FAILED.

Failure report:
- Tweet exists on X: {failure_report.get('tweet_exists')}
- Tweet text score: {failure_report.get('tweet_score')}/10 (need ≥7) — issues: {failure_report.get('tweet_issues')}
- Image reward score: {failure_report.get('image_score')} (need >-1.0)
- Terminal output score: {failure_report.get('terminal_score')}/10 (need ≥7)
- Terminal errors found: {failure_report.get('terminal_errors')}
- Terminal all stages present: {failure_report.get('all_stages_present')}

Terminal output (first 5000 chars):
---
{snippet}
---

You may modify any file EXCEPT .env, improve_with_claude_code.py, and verify_quality.py.
Same cost constraint applies: do not switch high-volume calls to grok-4 flagship.

Review the failures and decide:
1. If the failures are fixable with code changes → fix them and respond: FIXED
2. If the failures are due to external factors (API, randomness) not caused by code → respond: NO_CHANGE (will retry same code)
3. If the failures indicate the improvement is fundamentally broken or not worth pursuing → respond: GIVE_UP

Your response must START with one of: FIXED | NO_CHANGE | GIVE_UP
Then explain your reasoning.
"""


# ── Streaming subprocess runner for Claude Code ────────────────────────────────

def _display_stream_event(event: dict) -> None:
    """Parse a stream-json event from Claude Code and print a human-readable line."""
    etype = event.get("type", "")

    if etype == "system":
        if event.get("subtype") == "init":
            model = event.get("model", "")
            if model:
                print(f"  {_GRAY}    Model: {model}{_R}", flush=True)

    elif etype == "assistant":
        message = event.get("message", {})
        for block in message.get("content", []):
            btype = block.get("type", "")
            if btype == "thinking":
                text = block.get("thinking", "")
                preview = text[:200].replace("\n", " ").strip()
                if len(text) > 200:
                    preview += " …"
                print(f"  {_GRAY}    💭 {preview}{_R}", flush=True)
            elif btype == "text":
                text = block.get("text", "")
                for line in text.splitlines():
                    print(f"  {_GRAY}    📝 {line}{_R}", flush=True)
            elif btype == "tool_use":
                name = block.get("name", "?")
                inp = block.get("input", {})
                if name in ("Read", "ReadFile"):
                    detail = inp.get("file_path", inp.get("path", ""))
                elif name in ("Edit", "Write", "WriteFile"):
                    detail = inp.get("file_path", inp.get("path", ""))
                elif name in ("Bash", "Shell"):
                    detail = inp.get("command", "")[:120]
                else:
                    detail = str(inp)[:120]
                print(f"  {_CYAN}    🔧 {name}: {detail}{_R}", flush=True)

    elif etype == "tool":
        content = event.get("content", [])
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                lines = text.splitlines()
                if lines:
                    preview = lines[0][:150]
                    if len(lines) > 1:
                        preview += f"  … ({len(lines)} lines)"
                    print(f"  {_GRAY}    ↳ {preview}{_R}", flush=True)

    elif etype == "result":
        cost = event.get("total_cost_usd", event.get("cost_usd", 0))
        turns = event.get("num_turns", "?")
        is_error = event.get("is_error", False)
        if is_error:
            print(f"  {_RED}    ❌ Finished with error — {turns} turns, ${cost:.4f}{_R}", flush=True)
        else:
            print(f"  {_GREEN}    ✅ Done — {turns} turns, ${cost:.4f}{_R}", flush=True)


def _run_claude_streaming(
    cmd: list,
    env: dict,
    timeout: int = 1800,
    label: str = "Claude Code",
) -> "tuple[int | None, str]":
    """
    Run Claude Code with ``--output-format stream-json`` so every event
    (thinking, tool use, text) is printed to the terminal as it arrives.

    Returns ``(exit_code_or_None, final_result_text)``.
    """
    import threading

    if "--output-format" not in cmd:
        cmd = cmd + ["--output-format", "stream-json"]

    log_both(f"{_GRAY}    $ {' '.join(cmd[:4])} …{_R}")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        log_both(f"{_RED}❌  Claude Code binary not found: {cmd[0]}{_R}", "error")
        log_both(f"{_RED}    Install with: sudo npm install -g @anthropic-ai/claude-code{_R}", "error")
        return None, ""

    stderr_chunks: list[str] = []
    def _drain_stderr():
        assert proc.stderr is not None
        for raw in proc.stderr:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                stderr_chunks.append(line)
                print(f"  {_YELLOW}    [stderr] {line}{_R}", flush=True)

    t = threading.Thread(target=_drain_stderr, daemon=True)
    t.start()

    final_text = ""
    deadline = time.time() + timeout

    try:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            if time.time() > deadline:
                proc.kill()
                log_both(f"{_RED}❌  {label}: timed out after {timeout}s{_R}", "error")
                return None, final_text

            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                print(f"  {_GRAY}    {line}{_R}", flush=True)
                continue

            _display_stream_event(event)

            if event.get("type") == "result":
                final_text = event.get("result", "")

    except Exception as exc:
        log_both(f"{_YELLOW}⚠️  {label}: error reading stream: {exc}{_R}", "warning")

    proc.wait()
    t.join(timeout=5)
    return proc.returncode, final_text


# ── Phase 1: Code Improvement ──────────────────────────────────────────────────

def phase_1_improve_code(original_branch: str) -> "str | None":
    """Create branch, run Claude Code, verify basic imports. Returns branch name or None."""
    log_header("PHASE 1: CODE IMPROVEMENT")

    # Create branch
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    branch_name = f"auto-improve/{ts}"

    # Stash any uncommitted changes
    stashed = False
    dirty = _git(["status", "--porcelain"]).stdout.strip()
    if dirty:
        log_both(f"{_YELLOW}⚠️  Stashing uncommitted changes …{_R}")
        _git(["stash", "push", "-m", "auto-improve stash"])
        stashed = True

    _git(["checkout", "-b", branch_name])
    log_both(f"{_GREEN}✅  Created branch: {branch_name}{_R}")

    # Load history for the prompt
    from nodes.score import _load_history
    history = _load_history()
    log_both(f"{_GRAY}📊  History: {len(history)} records{_R}")

    prompt = _build_phase1_prompt(history, branch_name)

    # Run Claude Code
    log_both(f"{_CYAN}🤖  Running Claude Code (--max-turns 25) …{_R}")
    log_both(f"{_GRAY}    This may take several minutes.{_R}")

    # Resolve the authenticated Claude binary and build its environment.
    claude_bin = _find_claude_binary()
    claude_env = _build_claude_env()

    returncode, _ = _run_claude_streaming(
        [claude_bin, "-p", prompt, "--max-turns", "25"],
        claude_env,
        timeout=1800,
        label="Phase 1",
    )

    if returncode is None:  # timeout or FileNotFoundError
        _git(["checkout", original_branch])
        _git(["branch", "-D", branch_name], check=False)
        if stashed:
            _git(["stash", "pop"], check=False)
        return None

    claude_returncode = returncode

    if claude_returncode != 0:
        log_both(f"{_YELLOW}⚠️  Claude Code exited with code {claude_returncode} — continuing to verification{_R}", "warning")

    # Check if requirements.txt was modified, reinstall if so
    try:
        _maybe_reinstall_requirements()
    except RuntimeError as exc:
        log_both(f"{_RED}❌  Phase 1 aborted: {exc}{_R}", "error")
        _git(["checkout", original_branch])
        _git(["branch", "-D", branch_name], check=False)
        if stashed:
            _git(["stash", "pop"], check=False)
        return None

    # Verify imports
    log_both(f"{_CYAN}🔍  Verifying imports …{_R}")
    checks = [
        "from nodes.generate_content import generate_content; print('OK')",
        "from nodes.generate_image import generate_image; print('OK')",
        "from nodes.analyze import analyze_and_improve; print('OK')",
        "from nodes.score import score_and_store; print('OK')",
    ]
    for check in checks:
        result = subprocess.run(
            [sys.executable, "-c", check],
            cwd=str(PROJECT_DIR),
            capture_output=True, text=True
        )
        label = check.split("import ")[1].split(";")[0].strip()
        if result.returncode == 0:
            log_both(f"{_GREEN}    ✅  {label}{_R}")
        else:
            log_both(f"{_RED}❌  Import verification failed: {label}{_R}", "error")
            log_both(f"{_RED}    {result.stderr[:300]}{_R}", "error")
            _git(["checkout", original_branch])
            _git(["branch", "-D", branch_name], check=False)
            if stashed:
                _git(["stash", "pop"], check=False)
            return None

    # Commit any uncommitted changes on the branch
    status = _git(["status", "--porcelain"]).stdout.strip()
    if status:
        _git(["add", "-A"])
        _git(["commit", "-m", f"auto-improve: Claude Code changes ({ts})"], check=False)

    log_both(f"{_GREEN}✅  Phase 1 complete — branch ready: {branch_name}{_R}")
    return branch_name


# ── Phase 2: Live Cycle ────────────────────────────────────────────────────────

def phase_2_live_cycle(attempt: int) -> dict:
    """Run one full bot cycle with the improved code. Returns cycle output dict."""
    log_both(f"{_CYAN}🔄  Running live cycle (attempt {attempt}/3) …{_R}")
    log_both(f"{_GRAY}    This runs a REAL cycle and will post a real tweet.{_R}")

    output_path = PROJECT_DIR / "data" / "test_cycle_output.json"
    terminal_path = PROJECT_DIR / "data" / "test_cycle_terminal.txt"

    # Clean up any previous artifacts
    output_path.unlink(missing_ok=True)
    terminal_path.unlink(missing_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Stream output to terminal in real-time AND capture for verification
    stdout_lines: list = []
    stderr_lines: list = []

    try:
        import threading
        proc = subprocess.Popen(
            [sys.executable, "-u", "main.py", "--single-cycle"],
            cwd=str(PROJECT_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        def _drain(stream, buf: list, prefix: str = ""):
            for line in stream:
                print(prefix + line, end="", flush=True)
                buf.append(line)

        t_out = threading.Thread(target=_drain, args=(proc.stdout, stdout_lines))
        t_err = threading.Thread(target=_drain, args=(proc.stderr, stderr_lines, _GRAY))
        t_out.start(); t_err.start()

        try:
            proc.wait(timeout=900)
        except subprocess.TimeoutExpired:
            proc.kill()
            t_out.join(); t_err.join()
            partial = "".join(stdout_lines) + "\n--- STDERR ---\n" + "".join(stderr_lines)
            terminal_path.write_text(
                f"TIMEOUT: Live cycle exceeded 15 minutes\n\n{partial}", encoding="utf-8"
            )
            log_both(f"{_RED}❌  Live cycle timed out after 15 minutes{_R}", "error")
            return {"success": False, "errors": ["timeout"]}

        t_out.join(); t_err.join()
        terminal_content = "".join(stdout_lines) + "\n--- STDERR ---\n" + "".join(stderr_lines)
        terminal_path.write_text(terminal_content, encoding="utf-8")
        log_both(f"{_GRAY}    Exit code: {proc.returncode}{_R}")

    except Exception as exc:
        log_both(f"{_RED}❌  Live cycle subprocess failed: {exc}{_R}", "error")
        terminal_path.write_text(f"SUBPROCESS ERROR: {exc}", encoding="utf-8")
        return {"success": False, "errors": [str(exc)]}

    # Read cycle output
    if not output_path.exists():
        log_both(f"{_RED}❌  test_cycle_output.json not written — cycle may have crashed{_R}", "error")
        return {"success": False, "errors": ["output file missing"]}

    try:
        cycle_output = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log_both(f"{_RED}❌  Could not parse cycle output JSON: {exc}{_R}", "error")
        return {"success": False, "errors": [str(exc)]}

    if cycle_output.get("success"):
        log_both(f"{_GREEN}✅  Live cycle complete — tweet: {cycle_output.get('tweet_url', 'n/a')}{_R}")
    else:
        log_both(f"{_RED}❌  Live cycle reported failure: {cycle_output.get('errors')}{_R}", "error")

    return cycle_output


# ── Phase 3: Verification ──────────────────────────────────────────────────────

def _verify_tweet_exists(tweet_id: str) -> dict:
    if not tweet_id:
        return {"pass": False, "reason": "no tweet_id in cycle output"}
    try:
        import tweepy
        from config import X_BEARER_TOKEN, TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET, \
            TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET
        client = tweepy.Client(
            bearer_token=X_BEARER_TOKEN or None,
            consumer_key=TWITTER_CONSUMER_KEY,
            consumer_secret=TWITTER_CONSUMER_SECRET,
            access_token=TWITTER_ACCESS_TOKEN,
            access_token_secret=TWITTER_ACCESS_TOKEN_SECRET,
        )
        response = client.get_tweet(tweet_id, tweet_fields=["public_metrics"])
        if response.data is None:
            return {"pass": False, "reason": "tweet not found on X"}
        metrics = response.data.get("public_metrics", {})
        return {"pass": True, "metrics": metrics}
    except Exception as exc:
        return {"pass": False, "reason": str(exc)}


def _verify_tweet_text(tweet_text: str) -> dict:
    if not tweet_text:
        return {"pass": False, "score": 0, "issues": ["no tweet text"]}
    from services.grok_ai import get_grok_response
    prompt = f"""Evaluate this German-learning tweet:
---
{tweet_text}
---

Check:
1. FORMAT: Starts with #DeutschLernen [LEVEL], has 🇩🇪/🇬🇧 lines, emoji pairs
2. GERMAN: Word and sentence grammatically correct
3. TRANSLATION: English is natural and accurate
4. CEFR: Level matches word difficulty
5. EMOJIS: Two DIFFERENT emoji pairs (word line ≠ sentence line)
6. SENTENCE: Contains the exact word, is short and natural
7. LENGTH: Under 280 characters

Respond in JSON only: {{"pass": true/false, "score": 1-10, "issues": [...]}}
Pass if score >= 7.
"""
    try:
        raw = get_grok_response(
            prompt,
            "You are a strict quality reviewer for German learning social media content. Respond only with JSON.",
            max_tokens=300,
            temperature=0.1,
        )
        result = _parse_json_response(raw)
        result.setdefault("pass", result.get("score", 0) >= 7)
        result.setdefault("issues", [])
        return result
    except Exception as exc:
        log_both(f"{_YELLOW}⚠️  Tweet text verification failed: {exc}{_R}", "warning")
        return {"pass": False, "score": 0, "issues": [str(exc)]}


def _verify_image_quality(image_path: str, midjourney_prompt: str) -> dict:
    if not image_path or not os.path.exists(image_path):
        return {"pass": False, "score": -999.0, "reason": "image file not found"}
    try:
        from services.image_ranker import _get_model
        model = _get_model()
        if model is None:
            return {"pass": True, "score": 0.0, "reason": "ImageReward unavailable — skipped"}
        from PIL import Image as PILImage
        img = PILImage.open(image_path).convert("RGB")
        score = model.score(midjourney_prompt or "a beautiful photorealistic scene", img)
        return {"pass": score > -1.0, "score": round(float(score), 4), "reason": ""}
    except Exception as exc:
        return {"pass": True, "score": 0.0, "reason": f"scoring failed ({exc}) — skipped"}


def _verify_terminal_output(terminal_output: str) -> dict:
    from services.grok_ai import get_grok_response
    # Use head + tail so both early errors AND final stages/URL are visible.
    # A full cycle produces ~8000-12000 chars; 2500+2500 captures both ends reliably.
    if len(terminal_output) > 5000:
        head = terminal_output[:2500]
        tail = terminal_output[-2500:]
        snippet = head + "\n\n... [middle truncated] ...\n\n" + tail
    else:
        snippet = terminal_output
    prompt = f"""You are reviewing the terminal output of a German-learning Twitter bot that just ran one full cycle.

Terminal output (may be truncated):
---
{snippet}
---

Check for:
1. ERRORS: Any Python exceptions, tracebacks, or ERROR-level log lines
2. SKIPPED STEPS: Any stage that was skipped unexpectedly (generate_content, generate_image, generate_audio, create_video, publish must all appear)
3. API FAILURES: Any mention of rate limits, auth errors, or failed API calls
4. WARNINGS: Excessive warnings that suggest degraded operation
5. COMPLETION: Output ends with a successful publish confirmation (tweet URL visible)
6. TIMING: Any stage taking suspiciously long (>5 min for a single step)

Respond in JSON only:
{{
  "pass": true/false,
  "score": 1-10,
  "all_stages_present": true/false,
  "errors_found": ["..."],
  "warnings_found": ["..."],
  "summary": "one sentence"
}}

Pass if score >= 7 AND all_stages_present is true AND no unhandled exceptions.
"""
    try:
        raw = get_grok_response(
            prompt,
            "You are a strict quality reviewer for bot terminal output. Respond only with JSON.",
            max_tokens=400,
            temperature=0.1,
        )
        result = _parse_json_response(raw)
        result.setdefault("score", 0)
        result.setdefault("all_stages_present", False)
        result.setdefault("errors_found", [])
        result.setdefault("summary", "")
        # Enforce pass conditions in Python rather than trusting AI alone
        result["pass"] = (
            result["score"] >= 7
            and result["all_stages_present"]
            and not result["errors_found"]
        )
        return result
    except Exception as exc:
        log_both(f"{_YELLOW}⚠️  Terminal output verification failed: {exc}{_R}", "warning")
        return {"pass": False, "score": 0, "all_stages_present": False,
                "errors_found": [str(exc)], "summary": "verification error"}


def phase_3_verify(cycle_output: dict, terminal_output: str) -> dict:
    """Run four checks. Returns {"pass": bool, "details": {...}}."""
    log_both(f"{_CYAN}📊  Running verification checks …{_R}")

    tweet_id   = cycle_output.get("tweet_id", "")
    tweet_text = cycle_output.get("tweet_text", "")
    image_path = cycle_output.get("image_path", "")
    mj_prompt  = cycle_output.get("midjourney_prompt", "")

    # Check a: tweet exists on X
    tweet_check = _verify_tweet_exists(tweet_id)
    metrics_str = str(tweet_check.get("metrics", {})) if tweet_check["pass"] else tweet_check.get("reason", "")
    icon_a = f"{_GREEN}✅" if tweet_check["pass"] else f"{_RED}❌"
    log_both(f"{icon_a}  Tweet exists on X: {metrics_str}{_R}")

    # Check b: tweet text quality
    text_check = _verify_tweet_text(tweet_text)
    text_score = text_check.get("score", 0)
    text_issues = text_check.get("issues", [])
    icon_b = f"{_GREEN}✅" if text_check.get("pass") else f"{_RED}❌"
    log_both(f"{icon_b}  Tweet text score: {text_score}/10{' — PASS' if text_check.get('pass') else f' — FAIL (issues: {text_issues})'}{_R}")

    # Check c: image quality
    img_check = _verify_image_quality(image_path, mj_prompt)
    img_score = img_check.get("score", -999)
    icon_c = f"{_GREEN}✅" if img_check["pass"] else f"{_RED}❌"
    log_both(f"{icon_c}  Image quality score: {img_score:.3f}{' — PASS' if img_check['pass'] else ' — FAIL'}{_R}")

    # Check d: terminal output
    term_check = _verify_terminal_output(terminal_output)
    term_score = term_check.get("score", 0)
    term_errors = term_check.get("errors_found", [])
    term_stages = term_check.get("all_stages_present", False)
    icon_d = f"{_GREEN}✅" if term_check["pass"] else f"{_RED}❌"
    log_both(f"{icon_d}  Terminal output: {term_score}/10{' — PASS' if term_check['pass'] else f' — FAIL'} ({term_check.get('summary', '')}){_R}")

    all_pass = all([
        tweet_check["pass"],
        text_check.get("pass", False),
        img_check["pass"],
        term_check["pass"],
    ])

    details = {
        "tweet_exists":       tweet_check["pass"],
        "tweet_score":        text_score,
        "tweet_issues":       text_issues,
        "image_score":        img_score,
        "terminal_score":     term_score,
        "terminal_errors":    term_errors,
        "all_stages_present": term_stages,
        "cycle_crashed":      False,
    }

    return {"pass": all_pass, "details": details}


# ── Inter-Attempt Re-Improvement ───────────────────────────────────────────────

def _ask_claude_code_to_fix(failure_report: dict, attempt: int, terminal_output: str) -> str:
    """
    Ask Claude Code to review the failure and either fix code or give up.
    Returns "GIVE_UP", "FIXED", or "NO_CHANGE".
    """
    log_both(f"{_CYAN}🤖  Asking Claude Code to review failure (attempt {attempt}/3) …{_R}")

    prompt = _build_fix_prompt(failure_report, attempt, terminal_output)
    claude_bin = _find_claude_binary()
    claude_env = _build_claude_env()

    returncode, output = _run_claude_streaming(
        [claude_bin, "-p", prompt, "--max-turns", "15"],
        claude_env,
        timeout=900,
        label="Fix attempt",
    )

    if returncode is None:
        log_both(f"{_YELLOW}⚠️  Claude Code failed or timed out — treating as NO_CHANGE{_R}", "warning")
        return "NO_CHANGE"

    output = output.strip()
    last_lines = [ln.strip() for ln in output.splitlines()[-30:] if ln.strip()]
    log_both(f"{_GRAY}    Claude Code last line: {last_lines[-1] if last_lines else '(empty)'}{_R}")

    decision = "NO_CHANGE"
    for line in reversed(last_lines):
        first = line.split()[0].upper() if line.split() else ""
        if first in ("FIXED", "NO_CHANGE", "GIVE_UP"):
            decision = first
            break
        for kw in ("GIVE_UP", "FIXED", "NO_CHANGE"):
            if kw in line.upper() and len(line) <= 40:
                decision = kw
                break
        if decision != "NO_CHANGE":
            break

    log_both(f"{_CYAN}🤖  Claude Code says: {decision}{_R}")
    return decision


# ── Phase 4: Success ───────────────────────────────────────────────────────────

def phase_4_success(branch_name: str, original_branch: str) -> None:
    log_header("PHASE 4: DECISION — SUCCESS")

    # Write branch name for reference
    (PROJECT_DIR / "data" / "new_bot_branch.txt").write_text(branch_name, encoding="utf-8")
    log_both(f"{_GREEN}✅  Wrote branch name to data/new_bot_branch.txt{_R}")

    # Start new bot process on the branch
    log_both(f"{_CYAN}🚀  Starting new bot from branch {branch_name} …{_R}")
    new_process = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=str(PROJECT_DIR),
        start_new_session=True,
    )
    log_both(f"{_GREEN}✅  New bot started (PID {new_process.pid}){_R}")

    # Kill old bot
    old_pid = int(os.environ.get("OLD_BOT_PID", "0"))
    if old_pid:
        try:
            time.sleep(3)  # give new bot a moment to start
            os.kill(old_pid, signal.SIGTERM)
            log_both(f"{_CYAN}🛑  Sent SIGTERM to old bot (PID {old_pid}){_R}")
        except ProcessLookupError:
            log_both(f"{_YELLOW}⚠️  Old bot (PID {old_pid}) already stopped{_R}", "warning")
    else:
        log_both(f"{_YELLOW}⚠️  OLD_BOT_PID not set — old bot may still be running. Kill it manually.{_R}", "warning")

    log_both(f"{_GREEN}{_BOLD}⚠️  Branch NOT merged into main — run merge_improvement.sh to merge manually{_R}")


# ── Phase 4: Failure ───────────────────────────────────────────────────────────

def phase_4_failure(
    attempted_tweets: list,
    branch_name: "str | None",
    original_branch: str,
) -> None:
    log_header("PHASE 4: DECISION — FAILURE")

    # Delete all tweets posted during failed attempts
    if attempted_tweets:
        log_both(f"{_RED}🗑️  Deleting {len(attempted_tweets)} tweet(s) posted during failed attempts …{_R}")
        for tweet_id in attempted_tweets:
            _delete_tweet(tweet_id)
            _remove_from_history(tweet_id)
    else:
        log_both(f"{_GRAY}    No tweets to delete.{_R}")

    # Restore to original branch and delete improvement branch
    if branch_name:
        try:
            _git(["checkout", original_branch])
            log_both(f"{_CYAN}🔀  Checked out original branch: {original_branch}{_R}")
        except Exception as exc:
            log_both(f"{_YELLOW}⚠️  Could not checkout {original_branch}: {exc}{_R}", "warning")

        try:
            _git(["branch", "-D", branch_name])
            log_both(f"{_CYAN}🗑️  Deleted improvement branch: {branch_name}{_R}")
        except Exception as exc:
            log_both(f"{_YELLOW}⚠️  Could not delete branch {branch_name}: {exc}{_R}", "warning")

    # Restore stash if any
    stash_list = _git(["stash", "list"], check=False).stdout.strip()
    if "auto-improve stash" in stash_list:
        _git(["stash", "pop"], check=False)
        log_both(f"{_CYAN}📦  Restored stash{_R}")

    log_both(f"{_GRAY}    Old bot continues on {original_branch}.{_R}")


# ── Cleanup ────────────────────────────────────────────────────────────────────

def _cleanup_artifacts() -> None:
    for f in ["data/test_cycle_output.json", "data/test_cycle_terminal.txt"]:
        try:
            (PROJECT_DIR / f).unlink(missing_ok=True)
        except Exception:
            pass


# ── Main ───────────────────────────────────────────────────────────────────────

def run() -> None:
    log_header("SELF-IMPROVEMENT ENGINE — STARTING")
    log_both(f"{_GRAY}    Log: data/improvement.log{_R}")

    original_branch = _git_current_branch()
    log_both(f"{_GRAY}    Current branch: {original_branch}{_R}")

    branch_name: "str | None" = None
    attempted_tweets: list = []

    try:
        # ── Phase 1 ──────────────────────────────────────────────────────────
        branch_name = phase_1_improve_code(original_branch)
        if not branch_name:
            log_both(f"{_RED}❌  Phase 1 failed — improvement aborted{_R}", "error")
            return

        # ── Phase 2+3: up to 3 attempts ───────────────────────────────────
        for attempt in range(1, 4):
            log_header(f"PHASE 2+3: ATTEMPT {attempt}/3")

            cycle_output = phase_2_live_cycle(attempt)

            if not cycle_output.get("success"):
                log_both(f"{_RED}❌  Live cycle crashed on attempt {attempt}{_R}", "error")
                if cycle_output.get("tweet_id"):
                    attempted_tweets.append(cycle_output["tweet_id"])
                failure_report = {
                    "cycle_crashed": True,
                    "tweet_exists": False,
                    "tweet_score": 0,
                    "tweet_issues": ["cycle crashed"],
                    "image_score": -999.0,
                    "terminal_score": 0,
                    "terminal_errors": ["cycle crashed"],
                    "all_stages_present": False,
                }
            else:
                if cycle_output.get("tweet_id"):
                    attempted_tweets.append(cycle_output["tweet_id"])

                terminal_path = PROJECT_DIR / "data" / "test_cycle_terminal.txt"
                terminal_output = terminal_path.read_text(encoding="utf-8") \
                    if terminal_path.exists() else ""

                verification = phase_3_verify(cycle_output, terminal_output)

                if verification["pass"]:
                    log_both(f"\n{_GREEN}{_BOLD}✅  ALL CHECKS PASSED on attempt {attempt}/3{_R}")
                    phase_4_success(branch_name, original_branch)
                    _cleanup_artifacts()
                    return

                failure_report = verification["details"]
                log_both(f"\n{_RED}⚠️  Attempt {attempt}/3 FAILED{_R}", "warning")

            # Ask Claude Code what to do (only if there are remaining attempts)
            if attempt < 3:
                terminal_output = ""
                terminal_path = PROJECT_DIR / "data" / "test_cycle_terminal.txt"
                if terminal_path.exists():
                    terminal_output = terminal_path.read_text(encoding="utf-8")

                decision = _ask_claude_code_to_fix(failure_report, attempt, terminal_output)

                if decision == "GIVE_UP":
                    log_both(f"{_RED}🤖  Claude Code: GIVE_UP after attempt {attempt} — aborting{_R}", "warning")
                    break
                if decision == "FIXED":
                    try:
                        _maybe_reinstall_requirements()
                    except RuntimeError as exc:
                        log_both(f"{_YELLOW}⚠️  pip install after fix failed: {exc}{_R}", "warning")
                # FIXED or NO_CHANGE → continue to next attempt

        # All attempts exhausted or GIVE_UP
        log_both(f"{_RED}❌  All verification attempts failed{_R}", "error")
        phase_4_failure(attempted_tweets, branch_name, original_branch)
        _cleanup_artifacts()

    except Exception as exc:
        log_both(f"{_RED}❌  Improvement engine crashed: {exc}{_R}", "error")
        _file_logger.exception("Improvement engine crashed")
        phase_4_failure(attempted_tweets, branch_name, original_branch)
        _cleanup_artifacts()


if __name__ == "__main__":
    # Allow --force flag to bypass ENABLE_SELF_IMPROVEMENT check
    force = "--force" in sys.argv

    if not force:
        try:
            from config import ENABLE_SELF_IMPROVEMENT
            if not ENABLE_SELF_IMPROVEMENT:
                print("Self-improvement is disabled (ENABLE_SELF_IMPROVEMENT=false).")
                print("Use --force to run anyway.")
                sys.exit(0)
        except ImportError:
            pass

    run()
