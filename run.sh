#!/usr/bin/env bash
# run.sh — pull latest from GitHub and start the bot

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BRANCH="$(git branch --show-current)"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🤖  XBot 3 — starting on branch: $BRANCH"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── 1. Pull latest from GitHub ──────────────────────────────────
echo "📥  Pulling latest from GitHub ($BRANCH) …"

# Stash any uncommitted runtime data so the pull doesn't get blocked
STASHED=false
if ! git diff --quiet; then
    git stash push -m "run.sh: stash before pull" --include-untracked
    STASHED=true
fi

# Only pull if this branch actually exists on origin. auto-improve/* branches
# (from the self-improvement engine) are local-only, and `git pull origin <branch>`
# would fail with "couldn't find remote ref" — aborting the whole script under `set -e`.
if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
    git pull origin "$BRANCH"
else
    echo "ℹ️   Branch '$BRANCH' is local-only (not on origin) — skipping pull."
fi

# Restore stashed runtime data (merge favours the freshly-pulled version on conflict)
if [ "$STASHED" = true ]; then
    git stash pop || true
fi

echo ""

# ── 2. Activate Python environment ─────────────────────────────
echo "🐍  Activating Python environment …"
source "$SCRIPT_DIR/venv/bin/activate"
echo ""

# ── 3. Run the bot ──────────────────────────────────────────────
echo "🚀  Starting main.py …"
echo ""
python -u main.py
