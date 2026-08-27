"""
Tests for the deterministic duplicate-word rejection in nodes.generate_content.

The avoid-list held the whole history, but only the last 20 words reached the
word-pick prompt and the last 50 the LLM similarity gate — so a word simply aged
out of the window and could be taught again. Measured on the live history: 84 of
396 posts (21%) repeated a word, e.g. "sortieren" on 2026-03-15 and again on
2026-08-24, ~250 posts apart.
"""

import unittest

try:
    from nodes.generate_content import _is_word_too_similar, _history_words
    _IMPORT_ERR = None
except Exception as exc:  # pragma: no cover
    _is_word_too_similar = _history_words = None
    _IMPORT_ERR = exc


@unittest.skipIf(_is_word_too_similar is None, f"import unavailable: {_IMPORT_ERR}")
class ExactRepeatTests(unittest.TestCase):
    """These never reach the LLM: the exact check returns first, so no API call."""

    def test_exact_repeat_rejected_beyond_the_llm_window(self):
        avoid = ["sortieren"] + [f"wort{i}" for i in range(200)]
        similar, matched = _is_word_too_similar("sortieren", avoid)
        self.assertTrue(similar, "a word older than the 50-word window must still be caught")
        self.assertEqual(matched, "sortieren")

    def test_case_and_whitespace_insensitive(self):
        for probe in ("URLAUB", "  urlaub  ", "Urlaub"):
            similar, matched = _is_word_too_similar(probe, ["Urlaub"] + ["x"] * 100)
            self.assertTrue(similar, probe)
            self.assertEqual(matched, "Urlaub")

    def test_empty_avoid_list_is_not_a_repeat(self):
        self.assertEqual(_is_word_too_similar("Urlaub", []), (False, ""))

    def test_blank_word_is_not_matched_against_blank_entries(self):
        similar, _ = _is_word_too_similar("   ", ["", "  "])
        self.assertFalse(similar)


@unittest.skipIf(_history_words is None, f"import unavailable: {_IMPORT_ERR}")
class HistoryWordTests(unittest.TestCase):
    def test_reads_current_and_legacy_schema(self):
        """35 Feb-Mar records store the word under the pre-refactor key
        'german_word' and were invisible to the avoid-list."""
        history = [
            {"source_word": "Urlaub"},
            {"german_word": "Gemuetlichkeit"},          # legacy row
            {"source_word": "", "german_word": "Freund"},
            {"source_word": "  spaced  "},
            {"other": "ignored"},
        ]
        self.assertEqual(_history_words(history),
                         ["Urlaub", "Gemuetlichkeit", "Freund", "spaced"])

    def test_empty_history(self):
        self.assertEqual(_history_words([]), [])


if __name__ == "__main__":
    unittest.main()
