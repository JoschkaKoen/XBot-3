#!/usr/bin/env bash
# check.sh — fast local verification before committing.
#
# Runs (in order):
#   1. byte-compile every core module (catches syntax errors)
#   2. import the full LangGraph pipeline (catches import/wiring errors)
#   3. the regression test suite (tests/)
#   4. pyflakes, if installed (unused names / undefined refs)
#
# Uses the project venv. No network, no GPU, no X API, no real tweets.

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY="venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "❌  venv not found at $PY — create it: python3 -m venv venv && pip install -r requirements.txt"
    exit 1
fi

fail=0

echo "━━ 1/4  byte-compile ━━"
if "$PY" -m compileall -q \
    config.py graph.py state.py main.py scaffolds.py improve_with_claude_code.py \
    config_parsers.py config_logging.py nodes services utils styles improve tests \
    scripts/verify_quality.py; then
    echo "   ✅  compile OK"
else
    echo "   ❌  compile FAILED"; fail=1
fi

echo "━━ 2/4  pipeline import ━━"
if "$PY" -c "from graph import build_graph; build_graph(); import improve_with_claude_code" 2>/tmp/xbot_check_import.log; then
    echo "   ✅  graph builds + engine imports"
else
    echo "   ❌  import FAILED:"; sed 's/^/      /' /tmp/xbot_check_import.log; fail=1
fi

echo "━━ 3/4  tests ━━"
if "$PY" -m unittest discover -s tests >/tmp/xbot_check_tests.log 2>&1; then
    tail -1 /tmp/xbot_check_tests.log | sed 's/^/   /'
    echo "   ✅  tests OK"
else
    echo "   ❌  tests FAILED:"; tail -20 /tmp/xbot_check_tests.log | sed 's/^/      /'; fail=1
fi

echo "━━ 4/4  pyflakes (optional) ━━"
if "$PY" -c "import pyflakes" 2>/dev/null; then
    if "$PY" -m pyflakes config.py graph.py state.py main.py scaffolds.py nodes services utils styles improve tests 2>/tmp/xbot_check_flakes.log; then
        echo "   ✅  pyflakes clean"
    else
        echo "   ⚠️   pyflakes warnings:"; sed 's/^/      /' /tmp/xbot_check_flakes.log
    fi
else
    echo "   ⏭   pyflakes not installed — skipped (pip install pyflakes to enable)"
fi

echo ""
if [ "$fail" -eq 0 ]; then
    echo "✅  ALL CHECKS PASSED"
else
    echo "❌  CHECKS FAILED — see above"
fi
exit "$fail"
