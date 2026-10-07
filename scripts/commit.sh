#!/usr/bin/env bash
#
# ONE serialized commit for a working tree shared by concurrent sessions.
#
#     scripts/commit.sh "Message"          # the normal call: lock → add -A → commit → unlock
#     scripts/commit.sh -m "…" --author …  # anything else is passed through to `git commit`
#
# WHY THIS EXISTS. This tree is edited continuously by several Claude/agent sessions at once,
# and the repo's rule is to stage the WHOLE tree (`git add -A`, CLAUDE.md). Two sessions doing
# that at the same moment collide in two different ways, and both were observed live:
#
#   1. `.git/index.lock` — git takes one global index lock, so the second session dies with
#      "fatal: Unable to create '…/.git/index.lock': File exists." Measured 2026-09-19: the
#      lock was held at 14 MB mid-`add` while 30 builder/checker processes ran.
#   2. Interleaving — session A's `add -A` sweeps up session B's files BETWEEN B's own `add`
#      and B's `commit`, so the two commits carry overlapping, half-written snapshots of each
#      other's work.
#
# A git hook cannot fix either: hooks run INSIDE `git commit`, after the index lock is already
# held, so a hook that waited would hold that lock for minutes and block every other session's
# git command. The lock therefore has to be taken BEFORE `git add`, which means here.
#
# Owner-checked mkdir locks use Python fcntl to serialize reclamation; macOS
# lacks flock(1), but provides the kernel primitive through Python.
#
# ⛔ This never passes --no-verify. The pre-commit hook still runs (it is fast now: it guards
#    what is IN the commit, not the consistency of the shared tree — see .githooks/pre-commit).
#
set -uo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "commit.sh: not a git repo" >&2; exit 1; }
cd "$ROOT" || exit 1
GITDIR="$(git rev-parse --git-dir)"

# Translation batches run separately: saving shared work must never wait for a
# provider job or another session's content. The clean-snapshot pre-push gate
# still requires translation freshness before any deployment.

LOCK="$GITDIR/esg-commit.lock"
WAIT_S="${COMMIT_LOCK_WAIT_S:-900}"      # how long to queue behind another session (15 min)

# ── acquire ─────────────────────────────────────────────────────────────────────
LOCKED=""
waited=0
while [ "$waited" -lt "$WAIT_S" ]; do
  if LOCK_TOKEN=$(python3 "$ROOT/scripts/process_lock.py" acquire "$LOCK" "$$"); then LOCKED=1; break; fi
  [ "$waited" = 0 ] && echo "commit.sh: another session is committing — queueing…" >&2
  sleep 3
  waited=$((waited + 3))
done
if [ -z "$LOCKED" ]; then
  echo "commit.sh: could not acquire $LOCK after ${WAIT_S}s." >&2
  echo "           Inspect the holder PID; dead owners are reclaimed automatically." >&2
  exit 1
fi
trap 'python3 "$ROOT/scripts/process_lock.py" release "$LOCK" "$$" "$LOCK_TOKEN" >/dev/null 2>&1 || :' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ── stage the whole tree, retrying git's own index lock ─────────────────────────
# Even holding our lock, a session running a bare `git add`/`git commit` (or any tool that
# writes the index — a checker, an editor's git integration) can hold .git/index.lock for a
# moment. That is transient by nature, so retry rather than fail the commit.
# A git process killed mid-write leaves .git/index.lock behind, and every later commit in
# every session then fails until someone deletes it. It happens when a session's tool timeout
# (often 2 minutes) kills a long `git add -A` on this tree: three orphaned locks in one hour on
# 2026-09-28, each blocking all sessions. Reclaim one only when it is old, no process has it
# open, and no git that writes the index is running.
INDEX_LOCK_STALE_MIN="${COMMIT_INDEX_LOCK_STALE_MIN:-5}"
reclaim_stale_index_lock() {
  local l="$GITDIR/index.lock"
  [ -f "$l" ] || return 1
  [ -n "$(find "$l" -maxdepth 0 -mmin "+$INDEX_LOCK_STALE_MIN" 2>/dev/null)" ] || return 1
  if command -v lsof >/dev/null 2>&1 && [ -n "$(lsof "$l" 2>/dev/null)" ]; then return 1; fi
  if pgrep -f 'git (add|commit|update-index|read-tree|checkout|merge|rebase|stash|reset|apply|am|mv|rm)( |$)' >/dev/null 2>&1; then return 1; fi
  rm -f "$l" && echo "commit.sh: removed a stale $l (older than ${INDEX_LOCK_STALE_MIN} min, held by no process)." >&2
}

git_retry() {
  local tries=0 out rc
  while :; do
    out="$("$@" 2>&1)"; rc=$?
    if [ "$rc" -eq 0 ] || ! printf '%s' "$out" | grep -q 'index\.lock'; then
      [ -n "$out" ] && printf '%s\n' "$out"
      return "$rc"
    fi
    reclaim_stale_index_lock && continue
    tries=$((tries + 1))
    if [ "$tries" -ge 20 ]; then
      printf '%s\n' "$out" >&2
      echo "commit.sh: .git/index.lock stayed held through 20 retries — is a build writing the index?" >&2
      return "$rc"
    fi
    sleep 3
  done
}

# ⚠ WHOLE TREE on purpose (CLAUDE.md): the shared tree's normal state is many sessions' edits
#   that already pass the validators, and leaving some out ships a committed tip that is
#   half-broken. Committing everything SAVES that in-flight work; it never discards anything.
git_retry git add -A || { echo "commit.sh: git add -A failed." >&2; exit 1; }

# A same-tree merge still joins histories and must create its two-parent commit.
record_sha() {
  if [ -n "${COMMIT_SHA_FILE:-}" ]; then
    git rev-parse HEAD > "$COMMIT_SHA_FILE" || return 1
  fi
}
if git diff --cached --quiet && ! git rev-parse --verify -q MERGE_HEAD >/dev/null; then
  echo "commit.sh: nothing staged — no commit made."
  record_sha || exit 1
  exit 0
fi

# ── commit ──────────────────────────────────────────────────────────────────────
if [ "$#" -eq 0 ]; then
  echo "commit.sh: no message. Usage: scripts/commit.sh \"Message\"" >&2
  exit 1
fi
case "$1" in
  -*) ARGS=("$@") ;;                 # already git-commit flags — pass straight through
  *)  ARGS=(-m "$1"); shift; ARGS+=("$@") ;;
esac

ESG_COMMIT_LOCK_HELD=1 git_retry git commit "${ARGS[@]}"
rc=$?
if [ "$rc" -eq 0 ]; then record_sha || exit 1; fi
if [ "$rc" -ne 0 ]; then
  echo "" >&2
  echo "commit.sh: the commit was refused (exit $rc). The hook prints the reason above." >&2
  echo "           The staged snapshot is intact — fix the reason and re-run this script." >&2
fi
exit "$rc"
