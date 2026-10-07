"""
utils/schedule — fixed daily posting slots.

A fixed interval that does not divide 24h drifts through the clock: at
POST_INTERVAL_SECONDS=23000 (6.39h) the posting time moves 4.83h later each day,
so over a week every post lands at every hour — measured on the live history,
~25% of posts went out between 00:00 and 05:00 in the audience's own timezone,
on both accounts. Anchoring to fixed UTC slots keeps the same daily volume while
never spending a post at 3am.

Pure and side-effect free so it can be unit-tested without sleeping.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("xbot.schedule")


def parse_slots(spec: str) -> list:
    """Parse "04:00,08:00" into sorted, de-duplicated (hour, minute) tuples.

    Malformed entries are skipped with a warning rather than raising: a typo in
    settings.env must not take the bot down mid-cycle.
    """
    slots = []
    for raw in (spec or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            hh, _, mm = raw.partition(":")
            hour, minute = int(hh), int(mm or 0)
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError(raw)
            slots.append((hour, minute))
        except (ValueError, TypeError):
            logger.warning("POST_TIMES: ignoring malformed slot %r", raw)
    return sorted(set(slots))


def seconds_until_next_slot(slots: list, now: datetime | None = None,
                            min_seconds: int = 60) -> int:
    """Seconds from *now* (UTC) until the next slot, wrapping past midnight.

    *min_seconds* keeps the bot from firing twice inside one slot when a cycle
    finishes a few seconds before the slot it was aimed at: a slot less than that
    far away is treated as already used, and the following one is chosen.
    """
    if not slots:
        raise ValueError("no valid slots")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    for day in (0, 1):
        base = (now + timedelta(days=day)).replace(second=0, microsecond=0)
        for hour, minute in slots:
            target = base.replace(hour=hour, minute=minute)
            delta = (target - now).total_seconds()
            if delta >= min_seconds:
                return int(delta)
    # Unreachable for a non-empty slot list, but never return a negative sleep.
    return min_seconds


def next_due_at(enabled, times, interval, now=None):
    """Persistable epoch for the next cycle, shared by startup and normal waits."""
    now = now or datetime.now(timezone.utc)
    slots = parse_slots(times) if enabled else []
    delay = seconds_until_next_slot(slots, now) if slots else max(60, interval)
    return now.timestamp() + delay


def startup_due_at(runtime, history, enabled, times, interval, now=None):
    now = now or datetime.now(timezone.utc)
    epoch = now.timestamp()
    stored = runtime.get('next_post_at')
    if enabled and parse_slots(times):
        if isinstance(stored, (int, float)) and stored > epoch:
            return stored
        # Reboots between slots never produce an extra immediate post. Allow a
        # persisted missed slot only within its one-minute dispatch grace period.
        if isinstance(stored, (int, float)) and 0 <= epoch - stored < 60:
            return epoch
        return next_due_at(enabled, times, interval, now)
    posted = []
    for row in history:
        if not row.get('tweet_id'):
            continue
        try:
            date = datetime.fromisoformat(row['timestamp'])
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            posted.append(date.timestamp())
        except (ValueError, TypeError, KeyError):
            continue
    last = max(posted + [runtime.get('last_success_at', 0) or 0])
    due = max(epoch, last + max(60, interval)) if last else epoch
    return max(due, stored) if isinstance(stored, (int, float)) else due
