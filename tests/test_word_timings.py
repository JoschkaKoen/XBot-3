"""
Tests for nodes.generate_audio._character_alignment_to_word_timings.

Guards the karaoke-timing rewrite: one entry per word, monotonic start times,
robust to repeated/substring words and to words missing from the TTS alignment.
"""

import unittest

try:
    from nodes.generate_audio import _character_alignment_to_word_timings as wt
    _IMPORT_ERR = None
except Exception as exc:   # heavy deps (services/config) unavailable in this env
    wt = None
    _IMPORT_ERR = exc


def _align(text):
    """A character alignment with one start time per char, 0.1s apart."""
    return {
        "characters": list(text),
        "character_start_times_seconds": [round(0.1 * i, 2) for i in range(len(text))],
    }


@unittest.skipIf(wt is None, f"generate_audio import unavailable: {_IMPORT_ERR}")
class WordTimingTests(unittest.TestCase):
    def _assert_one_per_word_monotonic(self, text, out):
        self.assertEqual([w["word"] for w in out], text.split())
        starts = [w["start"] for w in out]
        for a, b in zip(starts, starts[1:]):
            self.assertLessEqual(a, b, f"starts not monotonic: {starts}")
        for w in out:
            self.assertLessEqual(w["start"], w["end"])

    def test_clean_sentence(self):
        text = "Der Hund laeuft"
        out = wt(text, _align(text))
        self._assert_one_per_word_monotonic(text, out)
        self.assertEqual(out[0]["start"], 0.0)

    def test_repeated_words_get_distinct_starts(self):
        text = "die die Katze"
        out = wt(text, _align(text))
        self._assert_one_per_word_monotonic(text, out)
        # the two "die" must map to different positions (cursor advances)
        self.assertNotEqual(out[0]["start"], out[1]["start"])

    def test_substring_word(self):
        text = "er ist hier"
        out = wt(text, _align(text))
        self._assert_one_per_word_monotonic(text, out)

    def test_missing_word_does_not_drop_later_words(self):
        # alignment's middle word differs from the sentence's ("Hund" vs "XXXX");
        # the old code overshot its cursor here and dropped every later word.
        out = wt("Der Hund laeuft", _align("Der XXXX laeuft"))
        self.assertEqual([w["word"] for w in out], ["Der", "Hund", "laeuft"])
        self._assert_one_per_word_monotonic("Der Hund laeuft", out)

    def test_empty_alignment_falls_back(self):
        out = wt("Der Hund", {"characters": [], "character_start_times_seconds": []})
        self.assertEqual([w["word"] for w in out], ["Der", "Hund"])

    def test_empty_text(self):
        self.assertEqual(wt("", _align("abc")), [])


if __name__ == "__main__":
    unittest.main()
