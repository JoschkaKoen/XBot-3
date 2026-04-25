#!/bin/bash
# merge_improvement.sh — Manual review & merge helper for auto-improve branches
#
# Usage: bash scripts/merge_improvement.sh

set -e

CYAN="\033[96m"
GREEN="\033[92m"
YELLOW="\033[93m"
RED="\033[91m"
BOLD="\033[1m"
R="\033[0m"

echo -e "\n${CYAN}${BOLD}════════════════════════════════════════════════════════${R}"
echo -e "${CYAN}${BOLD}  IMPROVEMENT BRANCH REVIEW${R}"
echo -e "${CYAN}════════════════════════════════════════════════════════${R}\n"

CURRENT=$(git branch --show-current)
echo -e "  Current branch: ${BOLD}${CURRENT}${R}"

# Show improvement branches
echo -e "\n  ${CYAN}Available improvement branches:${R}"
BRANCHES=$(git branch | grep "auto-improve" || true)
if [ -z "$BRANCHES" ]; then
    echo -e "  ${YELLOW}  (none found)${R}"
    echo ""
    exit 0
fi
echo "$BRANCHES"

# Show latest branch hint
LATEST=$(git branch | grep "auto-improve" | tail -1 | xargs)
echo ""
read -rp "  Branch to review (default: $LATEST, or q to quit): " BRANCH
BRANCH="${BRANCH:-$LATEST}"

[ "$BRANCH" = "q" ] && echo -e "  ${YELLOW}Exited.${R}" && exit 0

# Validate branch exists
if ! git branch | grep -q "$BRANCH"; then
    echo -e "  ${RED}Branch '$BRANCH' not found.${R}"
    exit 1
fi

echo -e "\n${CYAN}${BOLD}── Stat diff: $BRANCH vs main ─────────────────────────${R}"
git diff "main..$BRANCH" --stat
echo ""

echo -e "${CYAN}${BOLD}── Full diff ──────────────────────────────────────────${R}"
git diff "main..$BRANCH"
echo ""

echo -e "${CYAN}${BOLD}── Commits on $BRANCH not in main ─────────────────────${R}"
git log "main..$BRANCH" --oneline
echo ""

read -rp "  Merge ${BOLD}${BRANCH}${R} into main? (y/n): " CONFIRM
if [ "$CONFIRM" = "y" ]; then
    git checkout main
    git merge "$BRANCH" --no-ff -m "Merge improvement branch $BRANCH"
    echo -e "\n  ${GREEN}${BOLD}✅  Merged successfully.${R}"
    echo -e "  ${YELLOW}⚠️   Restart the bot to run on main: python main.py${R}"
    echo ""

    read -rp "  Delete branch $BRANCH? (y/n): " DEL
    if [ "$DEL" = "y" ]; then
        git branch -d "$BRANCH"
        echo -e "  ${GREEN}✅  Branch deleted.${R}"
    fi
else
    echo -e "\n  ${YELLOW}Merge skipped. To switch back to main manually:${R}"
    echo -e "  ${BOLD}  git checkout main && python main.py${R}"
fi

echo ""
