"""
Tests for services.transcreate — the English→Chinese fan-out generator.

First AI-mocking tests in the suite. The patch seam is
services.transcreate.get_ai_response (bound at import in transcreate.py), NOT
services.ai_client.get_ai_response. The fake is keyed on retry_label rather than
call order, because stage-1 candidates run in a ThreadPoolExecutor.
"""

import json
import logging
import os
import tempfile
import threading
import unittest

try:
    import config
    import services.transcreate as T
    from services.targets import _spec_from_dict
    _IMPORT_ERR = None
except Exception as exc:  # pragma: no cover
    T = config = _spec_from_dict = None
    _IMPORT_ERR = exc


# A realistic stored midjourney_prompt: real scene + the constant photographic suffix.
_STYLE_SUFFIX = (", shot on Canon EOS R5, 35mm lens, natural lighting, RAW photo, "
                 "ultra realistic, 8k UHD, positive joyful atmosphere, warm and welcoming, "
                 "bright uplifting mood, subjects with natural warm smiles, positive facial expressions")
_SCENE = "A woman daydreams at a cluttered desk"


class FakeAI:
    """Keyed on retry_label so it's deterministic under the candidate thread pool."""

    def __init__(self, handlers):
        self.handlers = handlers            # label-substring -> str | callable(self)
        self.calls = []                     # (label, user, system)
        self._lock = threading.Lock()

    def __call__(self, model, user, system, max_tokens=0, temperature=0, retry_label=""):
        with self._lock:
            self.calls.append((retry_label, user, system))
        for key, resp in self.handlers.items():
            if key in retry_label:
                return resp(self) if callable(resp) else resp
        raise AssertionError(f"no fake response for retry_label={retry_label!r}")

    def prompts_for(self, key):
        return [(u, s) for (label, u, s) in self.calls if key in label]

    def count(self, key):
        return sum(1 for (label, _u, _s) in self.calls if key in label)


def _word(words):
    return json.dumps({"words": words})


def _sentence(text):
    return lambda _self: json.dumps({"sentence": text})


def _audience(word_zh, sentence_zh, e1="😅", e2="⏳"):
    return json.dumps({"audience_word": word_zh, "audience_sentence": sentence_zh,
                       "emoji1": e1, "emoji2": e2})


@unittest.skipIf(T is None, f"transcreate import unavailable: {_IMPORT_ERR}")
class TranscreateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        self._tmp.write("[]")
        self._tmp.close()
        self.spec = _spec_from_dict({
            "id": "zh", "source_language": "English", "target_language": "Chinese",
            "source_flag": "🇬🇧", "target_flag": "🇨🇳", "script": "simplified",
            "max_tweet_length": 280, "history_file": self._tmp.name,
        })
        # Determinism: fixed candidate count, funny style, CEFR rotation off by default.
        self._saved = {k: getattr(config, k) for k in ("TRANSCREATION_CANDIDATES", "CEFR_ROTATION")}
        config.TRANSCREATION_CANDIDATES = 3
        config.CEFR_ROTATION = False
        self._saved_style = config.resolve_tweet_style
        config.resolve_tweet_style = lambda cycle: "funny"
        self._orig_ai = T.get_ai_response

    def tearDown(self):
        T.get_ai_response = self._orig_ai
        config.resolve_tweet_style = self._saved_style
        for k, v in self._saved.items():
            setattr(config, k, v)
        os.unlink(self._tmp.name)

    def _write_history(self, records):
        with open(self._tmp.name, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)

    def _standard_ai(self, **overrides):
        handlers = {
            "word": _word([{"word": "procrastinate", "cefr": "B1"}]),
            "_src_": _sentence("She keeps procrastinating instead of starting."),
            "pick": "1 — relatable",
            "audience": _audience("拖延", "她一直拖延，迟迟不肯开始。"),
        }
        handlers.update(overrides)
        return FakeAI(handlers)

    # ── the core fix: word evoked by scene, avoid-list enforced ────────────────

    def test_avoid_list_skips_repeat_word(self):
        self._write_history([{"source_word": "notebook", "cefr_level": "A1"}])
        ai = self._standard_ai(word=_word([
            {"word": "notebook", "cefr": "A1"},        # already taught → must skip
            {"word": "procrastinate", "cefr": "B1"},   # fresh → chosen
        ]))
        T.get_ai_response = ai
        res = T.transcreate(self.spec, {"midjourney_prompt": _SCENE + _STYLE_SUFFIX}, verbose=False)
        self.assertEqual(res["source_word"], "procrastinate")

    def test_word_prompt_has_scene_avoid_cefr_and_strips_style_suffix(self):
        config.CEFR_ROTATION = True   # last level A1 → rotation asks for A2
        self._write_history([{"source_word": "notebook", "cefr_level": "A1"}])
        ai = self._standard_ai()
        T.get_ai_response = ai
        T.transcreate(self.spec, {"midjourney_prompt": _SCENE + _STYLE_SUFFIX}, verbose=False)
        user, _system = ai.prompts_for("word")[0]
        self.assertIn("daydreams at a cluttered desk", user)     # real scene present
        self.assertNotIn("Canon EOS R5", user)                   # photographic suffix stripped
        self.assertIn("notebook", user)                          # avoid-list present
        self.assertIn("A2", user)                                # rotated CEFR hint present

    def test_word_prompt_rejects_mood_and_depiction_words(self):
        """The funniness fix: mood/atmosphere and 'describe the picture' words are
        what produced flat captions (peaceful/convivial/give), so the word pick
        must explicitly rule them out and ask for comic potential instead."""
        ai = self._standard_ai()
        T.get_ai_response = ai
        T.transcreate(self.spec, {"midjourney_prompt": _SCENE + _STYLE_SUFFIX}, verbose=False)
        user, _system = ai.prompts_for("word")[0]
        self.assertIn("do NOT pick a word that merely DESCRIBES the picture", user)
        self.assertIn("mood/atmosphere adjectives", user)
        self.assertIn("comic potential", user)
        self.assertIn("EVERYDAY LIFE", user)          # German-bot framing
        self.assertNotIn("EVOKED BY", user)           # the wording that caused the regression

    def test_word_prompt_uses_compressed_situation_not_lush_description(self):
        """Only the first sentence (setting + subjects + action) reaches the word
        pick; the atmospheric tail soaked the call in descriptive register."""
        lush = ("A woman daydreams at a cluttered desk. Warm lamplight glows softly across "
                "the wall, and her gentle smile conveys quiet kindness." + _STYLE_SUFFIX)
        ai = self._standard_ai()
        T.get_ai_response = ai
        T.transcreate(self.spec, {"midjourney_prompt": lush}, verbose=False)
        user, _ = ai.prompts_for("word")[0]
        self.assertIn("daydreams at a cluttered desk", user)
        self.assertNotIn("quiet kindness", user)      # atmospheric tail dropped
        # Stage 1 still gets the full scene for grounding.
        s1_user, _ = ai.prompts_for("_src_")[0]
        self.assertIn("quiet kindness", s1_user)

    def test_empty_scene_falls_back_to_free_pick(self):
        ai = self._standard_ai()
        T.get_ai_response = ai
        res = T.transcreate(self.spec, {"midjourney_prompt": ""}, verbose=False)
        user, _ = ai.prompts_for("word")[0]
        self.assertNotIn("set in this everyday situation", user)   # no scene clause
        self.assertNotIn("DESCRIBES the picture", user)            # no anchor clause either
        self.assertEqual(res["source_word"], "procrastinate")

    # ── audience + China-compliance steering in joke generation ────────────────

    def test_stage1_targets_the_chinese_audience(self):
        ai = self._standard_ai()
        T.get_ai_response = ai
        T.transcreate(self.spec, {"midjourney_prompt": _SCENE}, verbose=False)
        _user, system = ai.prompts_for("_src_")[0]
        self.assertIn("Chinese speakers learning English", system)
        self.assertIn("universally relatable", system)
        self.assertIn("faithful Chinese translation", system)

    def test_china_policy_steers_generation_only_when_declared(self):
        spec_cn = _spec_from_dict({
            "id": "zh", "source_language": "English", "target_language": "Chinese",
            "source_flag": "🇬🇧", "target_flag": "🇨🇳", "script": "simplified",
            "max_tweet_length": 280, "history_file": self._tmp.name,
            "content_policy": "china",
        })
        ai = self._standard_ai()
        T.get_ai_response = ai
        T.transcreate(spec_cn, {"midjourney_prompt": _SCENE}, verbose=False)
        _user, system = ai.prompts_for("_src_")[0]
        self.assertIn("mainland China", system)               # steer present
        self.assertIn("ONLY restriction", system)             # …and explicitly bounded
        self.assertIn("do not water the joke down", system)

        # The default spec (no content_policy) must get NO compliance text at all.
        ai2 = self._standard_ai()
        T.get_ai_response = ai2
        T.transcreate(self.spec, {"midjourney_prompt": _SCENE}, verbose=False)
        _user2, system2 = ai2.prompts_for("_src_")[0]
        self.assertNotIn("mainland China", system2)

    # ── faithful Chinese, not a reinvented joke ────────────────────────────────

    def test_stage2_demands_faithful_translation(self):
        ai = self._standard_ai()
        T.get_ai_response = ai
        T.transcreate(self.spec, {"midjourney_prompt": _SCENE}, verbose=False)
        user, system = ai.prompts_for("audience")[0]
        blob = (user + " " + system).lower()
        self.assertIn("faithful", blob)
        self.assertIn("map it back", user.lower())
        self.assertNotIn("not a literal word-for-word", blob)   # old anti-faithful wording gone

    def test_non_chinese_translation_triggers_one_retry(self):
        calls = {"n": 0}
        def audience(_self):
            calls["n"] += 1
            return _audience("X", "no chinese here") if calls["n"] == 1 else _audience("炖", "汤在慢慢地炖着。")
        ai = self._standard_ai(audience=audience)
        T.get_ai_response = ai
        res = T.transcreate(self.spec, {"midjourney_prompt": _SCENE}, verbose=False)
        self.assertEqual(ai.count("audience"), 2)
        self.assertTrue(T._has_han(res["audience_sentence"]))

    # ── length degradation never truncates the Chinese ─────────────────────────

    def test_over_length_degrades_without_truncating_chinese(self):
        long_zh = "这是一句非常非常非常非常非常非常非常非常非常长的中文句子。" * 4
        ai = self._standard_ai(audience=_audience("忙碌", long_zh, "🍳", "🔥"))
        T.get_ai_response = ai
        res = T.transcreate(self.spec, {"midjourney_prompt": "A kitchen"}, verbose=False)
        self.assertIn(long_zh, res["full_tweet"])                 # never sliced
        self.assertNotIn("🍳", res["full_tweet"])                 # emoji dropped
        self.assertEqual(res["full_tweet"].count("#"), 1)        # hashtags shortened

    def test_in_budget_tweet_keeps_emoji_and_hashtags(self):
        ai = self._standard_ai()
        T.get_ai_response = ai
        res = T.transcreate(self.spec, {"midjourney_prompt": _SCENE}, verbose=False)
        self.assertLessEqual(T.x_weighted_len(res["full_tweet"]), self.spec.max_tweet_length)
        self.assertEqual(res["full_tweet"].count("#"), 2)

    # ── contract ───────────────────────────────────────────────────────────────

    def test_return_contract(self):
        ai = self._standard_ai()
        T.get_ai_response = ai
        res = T.transcreate(self.spec, {"midjourney_prompt": _SCENE}, verbose=False)
        self.assertEqual(set(res), {"full_tweet", "source_word", "source_sentence",
                                    "audience_word", "audience_sentence", "cefr"})
        self.assertIn(res["source_sentence"], res["full_tweet"])
        self.assertIn(res["source_word"], res["full_tweet"])
        self.assertIn(res["audience_sentence"], res["full_tweet"])

    def test_candidate_count_follows_config(self):
        config.TRANSCREATION_CANDIDATES = 2
        ai = self._standard_ai()
        T.get_ai_response = ai
        T.transcreate(self.spec, {"midjourney_prompt": _SCENE}, verbose=False)
        self.assertEqual(ai.count("_src_"), 2)


@unittest.skipIf(T is None, f"transcreate import unavailable: {_IMPORT_ERR}")
class PureHelperTests(unittest.TestCase):
    def test_clean_scene_strips_style_suffix(self):
        self.assertEqual(T._clean_scene(_SCENE + _STYLE_SUFFIX), _SCENE)

    def test_clean_scene_empty(self):
        self.assertEqual(T._clean_scene(""), "")
        self.assertEqual(T._clean_scene(None), "")

    def test_situation_takes_first_sentence(self):
        lush = ("cozy living room at dusk, a shy teenage boy offers a tiny gift box to an "
                "elderly woman. Warm lamplight conveys quiet kindness." + _STYLE_SUFFIX)
        self.assertEqual(
            T._situation(lush),
            "cozy living room at dusk, a shy teenage boy offers a tiny gift box to an elderly woman")

    def test_situation_falls_back_when_no_sentence_break(self):
        long_no_period = "a beer garden at dusk with three graduates celebrating " * 6
        out = T._situation(long_no_period)
        self.assertTrue(out)
        self.assertLessEqual(len(out), 220)

    def test_situation_empty(self):
        self.assertEqual(T._situation(""), "")

    def test_first_emoji_trims_doubles(self):
        self.assertEqual(T._first_emoji("🖋️ 📓"), "🖋️")
        self.assertEqual(T._first_emoji("📓"), "📓")
        self.assertEqual(T._first_emoji(""), "")
        self.assertEqual(T._first_emoji(None), "")

    def test_has_han(self):
        self.assertTrue(T._has_han("拖延"))
        self.assertFalse(T._has_han("procrastinate"))

    def test_recent_words_dedup_and_window(self):
        recs = [{"source_word": w} for w in ["a", "b", "a", "c"]]
        self.assertEqual(T._recent_words(recs, 30), ["a", "b", "c"])
        self.assertEqual(T._recent_words(recs, 2), ["a", "c"])   # last-2 window, deduped

    def test_fix_quotes_only_for_non_german(self):
        tmpl = "„[SHORT_FUNNY_SOURCE_SENTENCE]“"
        out = T._fix_quotes(self._spec("English"), tmpl)
        self.assertEqual(out, "“[SHORT_FUNNY_SOURCE_SENTENCE]”")
        self.assertEqual(T._fix_quotes(self._spec("German"), tmpl), tmpl)

    @staticmethod
    def _spec(source_language):
        return _spec_from_dict({"id": "x", "source_language": source_language,
                                "target_language": "Chinese"})


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    unittest.main()
