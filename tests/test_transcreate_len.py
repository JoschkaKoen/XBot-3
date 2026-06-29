"""
Tests for transcreate.x_weighted_len — the CJK/emoji-aware length estimate that
guards China fan-out tweets against X's weighted-character limit.
"""

import unittest

try:
    from services.transcreate import x_weighted_len
    _IMPORT_ERR = None
except Exception as exc:
    x_weighted_len = None
    _IMPORT_ERR = exc


@unittest.skipIf(x_weighted_len is None, f"transcreate import unavailable: {_IMPORT_ERR}")
class XWeightedLenTests(unittest.TestCase):
    def test_ascii_counts_one_each(self):
        self.assertEqual(x_weighted_len("hello"), 5)
        self.assertEqual(x_weighted_len(""), 0)

    def test_cjk_counts_two_each(self):
        self.assertEqual(x_weighted_len("中文"), 4)          # 2 Han chars × 2
        self.assertEqual(x_weighted_len("日本語"), 6)        # 3 × 2

    def test_kana_and_hangul_wide(self):
        self.assertEqual(x_weighted_len("あ"), 2)            # Hiragana
        self.assertEqual(x_weighted_len("한"), 2)            # Hangul syllable

    def test_emoji_wide(self):
        self.assertEqual(x_weighted_len("🍎"), 2)

    def test_mixed(self):
        # "AB中" → 1 + 1 + 2 = 4
        self.assertEqual(x_weighted_len("AB中"), 4)


if __name__ == "__main__":
    unittest.main()
