"""Tests for utils.text.truncate_emoji_pairs (tweet trailing-emoji collapse)."""

import unittest

from utils.text import truncate_emoji_pairs as collapse


class TruncateEmojiPairsTests(unittest.TestCase):
    def test_single_emoji_doubled_collapses(self):
        self.assertEqual(collapse("🇩🇪  der Apfel  🍎🍎"), "🇩🇪  der Apfel  🍎")
        self.assertEqual(collapse("line  🚗🚗"), "line  🚗")

    def test_multi_codepoint_flag_doubled_collapses_to_one(self):
        # 🇩🇪 is two regional-indicator code points; a doubled flag must collapse
        # to a single whole flag, never split mid-emoji.
        self.assertEqual(collapse("flags  🇩🇪🇩🇪"), "flags  🇩🇪")

    def test_distinct_emojis_not_collapsed(self):
        self.assertEqual(collapse("ok  🚗🚙"), "ok  🚗🚙")

    def test_single_emoji_unchanged(self):
        self.assertEqual(collapse("end  🎉"), "end  🎉")

    def test_no_double_space_unchanged(self):
        self.assertEqual(collapse("no double here"), "no double here")

    def test_multiline_each_line_independent(self):
        src = "🇩🇪  der Apfel  🍎🍎\n🇺🇸  the apple  🍏🍏"
        self.assertEqual(collapse(src), "🇩🇪  der Apfel  🍎\n🇺🇸  the apple  🍏")


if __name__ == "__main__":
    unittest.main()
