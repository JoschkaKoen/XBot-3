"""
Tests for scaffolds.fill_scaffold / scaffold_at — the deterministic tweet
assembly the secondary (English→Chinese) fan-out relies on. Pure, no network.

These guard the formatting bugs the fan-out rewrite fixed: stray spaces from an
empty [ARTICLE], leftover [PLACEHOLDER] text after a scaffold edit, and the
triple-newline gap the LLM used to emit when it assembled the tweet itself.
"""

import logging
import unittest

try:
    from scaffolds import fill_scaffold, scaffold_at, _load_pool
    _IMPORT_ERR = None
except Exception as exc:  # pragma: no cover
    fill_scaffold = scaffold_at = _load_pool = None
    _IMPORT_ERR = exc


_ZH_VALUES = {
    "SOURCE_FLAG": "🇬🇧", "TARGET_FLAG": "🇨🇳",
    "SOURCE_LANGUAGE": "English", "TARGET_LANGUAGE": "Chinese",
    "ARTICLE": "",
    "SOURCE_WORD": "procrastinate",
    "TARGET_TRANSLATION": "拖延",
    "SHORT_FUNNY_SOURCE_SENTENCE": "She keeps procrastinating instead of starting.",
    "TARGET_TRANSLATION_OF_SENTENCE": "她一直拖延，迟迟不肯开始。",
    "EMOJI1": "😅", "EMOJI2": "⏳", "LEVEL": "B1",
}


@unittest.skipIf(fill_scaffold is None, f"scaffolds import unavailable: {_IMPORT_ERR}")
class FillScaffoldTests(unittest.TestCase):
    def test_every_real_scaffold_fills_cleanly(self):
        """All 6 shipped scaffolds fill with no leftover tokens, no stray spaces."""
        for name, template in _load_pool():
            out = fill_scaffold(template, _ZH_VALUES)
            self.assertNotIn("[", out, f"leftover placeholder in {name}: {out!r}")
            self.assertIn("procrastinate", out)
            self.assertIn("拖延", out)
            for line in out.split("\n"):
                self.assertEqual(line, line.rstrip(), f"trailing space in {name}: {line!r}")
                self.assertFalse(line.startswith(" "), f"leading space in {name}: {line!r}")

    def test_empty_article_leaves_no_gap(self):
        out = fill_scaffold("[SOURCE_FLAG]  [ARTICLE] [SOURCE_WORD]", _ZH_VALUES)
        self.assertEqual(out, "🇬🇧  procrastinate")   # two-space flag gap kept, no third space

    def test_nonempty_article_kept(self):
        vals = {**_ZH_VALUES, "ARTICLE": "die", "SOURCE_WORD": "Katze"}
        out = fill_scaffold("[SOURCE_FLAG]  [ARTICLE] [SOURCE_WORD]", vals)
        self.assertEqual(out, "🇬🇧  die Katze")

    def test_two_space_gaps_preserved(self):
        out = fill_scaffold("[TARGET_FLAG]  [TARGET_TRANSLATION]  [EMOJI1]", _ZH_VALUES)
        self.assertEqual(out, "🇨🇳  拖延  😅")

    def test_dropped_emoji_leaves_no_trailing_space(self):
        vals = {**_ZH_VALUES, "EMOJI1": ""}
        out = fill_scaffold("[TARGET_FLAG]  [TARGET_TRANSLATION]  [EMOJI1]", vals)
        self.assertEqual(out, "🇨🇳  拖延")

    def test_triple_newline_collapsed(self):
        out = fill_scaffold("[SOURCE_WORD]\n\n\n[TARGET_TRANSLATION]", _ZH_VALUES)
        self.assertEqual(out, "procrastinate\n\n拖延")

    def test_translation_token_no_prefix_collision(self):
        """[TARGET_TRANSLATION] must not clobber [TARGET_TRANSLATION_OF_SENTENCE]."""
        tmpl = "[TARGET_TRANSLATION] | [TARGET_TRANSLATION_OF_SENTENCE]"
        out = fill_scaffold(tmpl, _ZH_VALUES)
        self.assertEqual(out, "拖延 | 她一直拖延，迟迟不肯开始。")

    def test_unfilled_placeholder_stripped_and_logged(self):
        with self.assertLogs("xbot.scaffolds", level="WARNING") as cm:
            out = fill_scaffold("[SOURCE_WORD] [SURPRISE_TOKEN]", _ZH_VALUES)
        self.assertNotIn("[SURPRISE_TOKEN]", out)
        self.assertTrue(any("SURPRISE_TOKEN" in m for m in cm.output))

    def test_hashtags_render_from_source_language(self):
        out = fill_scaffold("#Learn[SOURCE_LANGUAGE] #[SOURCE_LANGUAGE]Vocabulary", _ZH_VALUES)
        self.assertEqual(out, "#LearnEnglish #EnglishVocabulary")


@unittest.skipIf(scaffold_at is None, f"scaffolds import unavailable: {_IMPORT_ERR}")
class ScaffoldAtTests(unittest.TestCase):
    def test_wraps_modulo_pool_size(self):
        pool = _load_pool()
        n = len(pool)
        self.assertEqual(scaffold_at(0), pool[0])
        self.assertEqual(scaffold_at(n), pool[0])          # wraps
        self.assertEqual(scaffold_at(n + 2), pool[2])
        self.assertEqual(scaffold_at(2 * n - 1), pool[n - 1])

    def test_stateless(self):
        """Repeated calls at the same index return the same scaffold (no advance)."""
        self.assertEqual(scaffold_at(3), scaffold_at(3))


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    unittest.main()
