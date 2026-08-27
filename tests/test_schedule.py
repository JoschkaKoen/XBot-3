"""
Tests for utils.schedule — the fixed daily posting slots that replaced the
drifting interval.

A 6.39h interval moves the posting time 4.83h later each day, which put ~25% of
live posts into 00:00-05:00 in the audience's own timezone. These cover the slot
maths that stops that, including the midnight wrap and the near-slot guard.
"""

import importlib.util
import unittest
from datetime import datetime, timezone

# Import the module directly: utils/__init__ pulls in ui -> config -> dotenv,
# which is not needed for this pure helper.
_spec = importlib.util.spec_from_file_location(
    "_sched", __file__.rsplit("/", 2)[0] + "/utils/schedule.py")
sched = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sched)


def _utc(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


class ParseSlotsTests(unittest.TestCase):
    def test_parses_sorted_and_deduped(self):
        self.assertEqual(sched.parse_slots("12:00,04:00,08:00,04:00"),
                         [(4, 0), (8, 0), (12, 0)])

    def test_bare_hour_and_whitespace(self):
        self.assertEqual(sched.parse_slots(" 7 , 19:30 "), [(7, 0), (19, 30)])

    def test_malformed_entries_skipped_not_raised(self):
        """A typo in settings.env must not take the bot down mid-cycle."""
        self.assertEqual(sched.parse_slots("04:00,oops,25:00,08:30,12:99"),
                         [(4, 0), (8, 30)])

    def test_empty(self):
        self.assertEqual(sched.parse_slots(""), [])
        self.assertEqual(sched.parse_slots(None), [])


class NextSlotTests(unittest.TestCase):
    SLOTS = [(4, 0), (8, 0), (12, 0), (15, 0)]

    def test_picks_next_slot_today(self):
        self.assertEqual(sched.seconds_until_next_slot(self.SLOTS, _utc("2026-08-27T03:00")), 3600)

    def test_picks_later_slot_same_day(self):
        secs = sched.seconds_until_next_slot(self.SLOTS, _utc("2026-08-27T12:05"))
        self.assertEqual(secs, int(2.9166 * 3600) + 1)   # -> 15:00

    def test_wraps_past_midnight(self):
        """After the last slot, the next is tomorrow's first."""
        secs = sched.seconds_until_next_slot(self.SLOTS, _utc("2026-08-27T15:30"))
        self.assertEqual(secs, 12 * 3600 + 30 * 60)      # -> next day 04:00

    def test_late_evening_wraps(self):
        secs = sched.seconds_until_next_slot(self.SLOTS, _utc("2026-08-27T23:59"))
        self.assertEqual(secs, 4 * 3600 + 60)            # -> next day 04:00

    def test_skips_a_slot_that_is_imminent(self):
        """A cycle finishing seconds before its own slot must not fire twice."""
        secs = sched.seconds_until_next_slot(self.SLOTS, _utc("2026-08-27T07:59:30"))
        self.assertGreater(secs, 60)
        self.assertEqual(secs, 4 * 3600 + 30)            # skipped 08:00 -> 12:00

    def test_never_returns_negative_or_tiny(self):
        for hour in range(24):
            secs = sched.seconds_until_next_slot(self.SLOTS, _utc(f"2026-08-27T{hour:02d}:00"))
            self.assertGreaterEqual(secs, 60)
            self.assertLessEqual(secs, 24 * 3600)

    def test_naive_datetime_treated_as_utc(self):
        naive = datetime.fromisoformat("2026-08-27T03:00")
        self.assertEqual(sched.seconds_until_next_slot(self.SLOTS, naive), 3600)

    def test_single_slot_wraps_to_next_day(self):
        secs = sched.seconds_until_next_slot([(9, 0)], _utc("2026-08-27T09:30"))
        self.assertEqual(secs, 23 * 3600 + 30 * 60)

    def test_empty_slots_raise(self):
        with self.assertRaises(ValueError):
            sched.seconds_until_next_slot([], _utc("2026-08-27T03:00"))

    def test_no_slot_lands_in_audience_dead_zone(self):
        """Default slots must avoid 00:00-05:00 for BOTH audiences
        (CN = UTC+8, DE = UTC+2)."""
        for hour, minute in sched.parse_slots("04:00,08:00,12:00,15:00"):
            for offset, name in ((8, "CN"), (2, "DE")):
                local = (hour + offset) % 24
                self.assertFalse(0 <= local <= 5,
                                 f"{hour:02d}:{minute:02d} UTC = {local}:00 {name} (dead zone)")


if __name__ == "__main__":
    unittest.main()
