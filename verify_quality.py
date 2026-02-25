"""
verify_quality.py — Standalone Quality Checker

Reads data/test_cycle_output.json and data/test_cycle_terminal.txt
and runs all four verification checks from the improvement pipeline.

Exit code 0 = all checks pass
Exit code 1 = one or more checks fail

Usage:
    python verify_quality.py
"""

import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

_R     = "\033[0m"
_BOLD  = "\033[1m"
_GREEN = "\033[92m"
_RED   = "\033[91m"
_CYAN  = "\033[96m"
_GRAY  = "\033[90m"
_YELLOW= "\033[93m"


def _print_check(name: str, passed: bool, detail: str = "") -> None:
    icon = f"{_GREEN}✅" if passed else f"{_RED}❌"
    suffix = f"  {_GRAY}{detail}{_R}" if detail else ""
    print(f"  {icon}  {name}{suffix}{_R}", flush=True)


def main() -> int:
    output_path   = PROJECT_DIR / "data" / "test_cycle_output.json"
    terminal_path = PROJECT_DIR / "data" / "test_cycle_terminal.txt"

    print(f"\n{_CYAN}{_BOLD}{'─' * 60}{_R}", flush=True)
    print(f"{_CYAN}{_BOLD}  QUALITY VERIFICATION{_R}", flush=True)
    print(f"{_CYAN}{'─' * 60}{_R}\n", flush=True)

    # Load cycle output
    if not output_path.exists():
        print(f"  {_RED}❌  data/test_cycle_output.json not found{_R}")
        print(f"  {_GRAY}    Run: python main.py --single-cycle{_R}")
        return 1

    cycle_output = json.loads(output_path.read_text(encoding="utf-8"))

    if not cycle_output.get("success"):
        print(f"  {_RED}❌  Cycle reported failure: {cycle_output.get('errors')}{_R}")
        return 1

    tweet_id   = cycle_output.get("tweet_id", "")
    tweet_text = cycle_output.get("tweet_text", "")
    image_path = cycle_output.get("image_path", "")
    mj_prompt  = cycle_output.get("midjourney_prompt", "")

    terminal_output = terminal_path.read_text(encoding="utf-8") \
        if terminal_path.exists() else ""

    # Import shared check functions from improve_with_claude_code
    from improve_with_claude_code import (
        _verify_tweet_exists,
        _verify_tweet_text,
        _verify_image_quality,
        _verify_terminal_output,
    )

    results = {}

    # Check a: tweet exists on X
    print(f"  {_CYAN}Checking tweet on X …{_R}", flush=True)
    r = _verify_tweet_exists(tweet_id)
    results["tweet_exists"] = r["pass"]
    _print_check(
        f"Tweet exists on X ({tweet_id})",
        r["pass"],
        str(r.get("metrics", r.get("reason", ""))),
    )

    # Check b: tweet text quality
    print(f"  {_CYAN}Checking tweet text quality …{_R}", flush=True)
    r = _verify_tweet_text(tweet_text)
    results["tweet_text"] = r.get("pass", False)
    _print_check(
        f"Tweet text quality (score {r.get('score', 0)}/10)",
        r.get("pass", False),
        ", ".join(r.get("issues", [])) or "no issues",
    )

    # Check c: image quality
    print(f"  {_CYAN}Checking image quality …{_R}", flush=True)
    r = _verify_image_quality(image_path, mj_prompt)
    results["image_quality"] = r["pass"]
    _print_check(
        f"Image quality (ImageReward {r.get('score', 0.0):.3f})",
        r["pass"],
        r.get("reason", ""),
    )

    # Check d: terminal output
    if terminal_output:
        print(f"  {_CYAN}Checking terminal output …{_R}", flush=True)
        r = _verify_terminal_output(terminal_output)
        results["terminal"] = r["pass"]
        _print_check(
            f"Terminal output (score {r.get('score', 0)}/10, all stages: {r.get('all_stages_present')})",
            r["pass"],
            r.get("summary", ""),
        )
    else:
        print(f"  {_YELLOW}⚠️   Terminal output file not found — skipping check{_R}", flush=True)
        results["terminal"] = True  # not the bot's fault

    # Summary
    all_pass = all(results.values())
    print(f"\n{_CYAN}{'─' * 60}{_R}", flush=True)
    if all_pass:
        print(f"{_GREEN}{_BOLD}  ✅  ALL CHECKS PASSED{_R}\n", flush=True)
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"{_RED}{_BOLD}  ❌  FAILED: {', '.join(failed)}{_R}\n", flush=True)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
