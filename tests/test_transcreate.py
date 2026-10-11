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
                       "emoji1": e1, "emoji2": e2, "quality_ok": True, "image_fit": True,
                       "audience_word_pinyin": "tuōyán",
                       "audience_sentence_pinyin": "Tā yìzhí tuōyán, chíchí bù kěn kāishǐ.", "pinyin_ok": True})


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
            "_review": json.dumps({"approved_indexes": [0]}),
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

    def test_word_picker_sees_older_history_too(self):
        self._write_history([{"source_word": "oldestword"}] +
                            [{"source_word": f"recentword{i}"} for i in range(35)])
        ai = self._standard_ai()
        T.get_ai_response = ai
        T.transcreate(self.spec, {}, verbose=False)
        self.assertIn("oldestword", ai.prompts_for("word")[0][0])

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

    def test_word_prompt_asks_for_life_concepts_not_bare_objects(self):
        """Concrete objects (tab, splinter) only support flat anecdotes — "He
        spills his beer." The funny words are life concepts you can write a truth
        about, which is what the German path's words are (Sonntagsruhe, Abschluss)."""
        ai = self._standard_ai()
        T.get_ai_response = ai
        T.transcreate(self.spec, {"midjourney_prompt": _SCENE}, verbose=False)
        user, _ = ai.prompts_for("word")[0]
        self.assertIn("LIFE CONCEPT", user)
        self.assertIn("witty one-line truth", user)
        self.assertIn("bare physical objects with no social meaning", user)

    def test_stage1_requires_concrete_meaning_and_comic_turn(self):
        """Meaning clues and a visible detail must support the comic turn."""
        ai = self._standard_ai()
        T.get_ai_response = ai
        T.transcreate(self.spec, {"midjourney_prompt": _SCENE}, verbose=False)
        user, _ = ai.prompts_for("_src_")[0]
        self.assertIn("show the word's meaning through a concrete everyday example", user)
        self.assertNotIn("Put a turn in it", user)
        self.assertIn("Add a small comic turn using basic words", user)

    def test_image_grounding_applies_on_normal_cycles_too(self):
        config.resolve_tweet_style = lambda cycle: "normal"
        ai = self._standard_ai()
        T.get_ai_response = ai
        T.transcreate(self.spec, {"midjourney_prompt": _SCENE}, verbose=False)
        user, _ = ai.prompts_for("_src_")[0]
        self.assertIn("Do not invent a different scene", user)

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

    # ── example-sentence vocabulary must stay readable for learners ────────────

    def test_stage1_restricts_surrounding_vocabulary(self):
        """The taught word is the only hard word allowed; without this the model
        wrote "Conquering bureaucracy earned her this gleaming certificate"."""
        config.CEFR_ROTATION = True
        self._write_history([{"source_word": "notebook", "cefr_level": "B2"}])
        ai = self._standard_ai()
        T.get_ai_response = ai
        T.transcreate(self.spec, {"midjourney_prompt": _SCENE}, verbose=False)
        user, _system = ai.prompts_for("_src_")[0]
        self.assertIn("ONLY word allowed to be difficult", user)
        self.assertIn("basic English a beginner already knows", user)
        # The level that governs the sentence is the one the word pick assigned to
        # the chosen word (B1 here) — the rotation only steers which word is picked.
        self.assertIn("learners at B1 level", user)

    def test_stage1_vocab_rule_without_cefr(self):
        """No level anywhere → still restrict vocabulary, just without a level."""
        config.CEFR_ROTATION = False
        ai = self._standard_ai(word=json.dumps({"words": [{"word": "procrastinate"}]}))
        T.get_ai_response = ai
        T.transcreate(self.spec, {"midjourney_prompt": _SCENE}, verbose=False)
        user, _ = ai.prompts_for("_src_")[0]
        self.assertIn("ONLY word allowed to be difficult", user)
        self.assertIn("this is a lesson for learners:", user)

    # ── "normal" cycles keep light humor (they used to have none) ──────────────

    def test_normal_style_still_asks_for_humor(self):
        """Every joke-less live tweet traced to a 'normal' cycle, where all humor
        guidance used to be stripped. Normal now means lighter, never none."""
        config.resolve_tweet_style = lambda cycle: "normal"
        ai = self._standard_ai()
        T.get_ai_response = ai
        T.transcreate(self.spec, {"midjourney_prompt": _SCENE}, verbose=False)
        user, system = ai.prompts_for("_src_")[0]
        blob = user + " " + system
        self.assertIn("raise a smile", blob)
        self.assertIn("A plain caption is not enough", blob)
        self.assertIn("Try this comic angle using easy words:", user)      # angles apply on normal too
        self.assertIn("clear, simple and natural", user)

    def test_funny_style_keeps_the_strong_tone(self):
        config.resolve_tweet_style = lambda cycle: "funny"
        ai = self._standard_ai()
        T.get_ai_response = ai
        T.transcreate(self.spec, {"midjourney_prompt": _SCENE}, verbose=False)
        user, system = ai.prompts_for("_src_")[0]
        self.assertIn("Include a clear, gentle everyday joke using easy words", user + " " + system)
        self.assertIn("rewrite the joke more simply instead of removing it", user)

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

    def test_stage2_gets_scene_for_word_sense_disambiguation(self):
        """Without the situation, "she might spill everything" was rendered 泄露
        ("divulge secrets") instead of spilling a drink."""
        ai = self._standard_ai()
        T.get_ai_response = ai
        T.transcreate(self.spec, {"midjourney_prompt": _SCENE + _STYLE_SUFFIX}, verbose=False)
        user, _ = ai.prompts_for("audience")[0]
        self.assertIn("daydreams at a cluttered desk", user)
        self.assertIn("right sense of an ambiguous word", user)
        self.assertIn("never translate the picture", user)
        self.assertIn("sense the word actually carries", user)

    def test_stage2_without_scene_has_no_context_line(self):
        ai = self._standard_ai()
        T.get_ai_response = ai
        T.transcreate(self.spec, {"midjourney_prompt": ""}, verbose=False)
        user, _ = ai.prompts_for("audience")[0]
        self.assertNotIn("post's picture shows", user)

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
        ai = self._standard_ai(audience=_audience("忙碌", long_zh, "🍳", "🔥"),
                               _src_=_sentence("I procrastinate every day."))
        T.get_ai_response = ai
        with self.assertRaisesRegex(ValueError, "within 3 attempts"):
            T.transcreate(self.spec, {"midjourney_prompt": "A kitchen"}, verbose=False)
        self.assertEqual(ai.count("_review"), 3)
        self.assertEqual(ai.count("audience"), 6)  # each attempt tries a shorter translation

    def test_in_budget_tweet_keeps_emoji_and_hashtags(self):
        ai = self._standard_ai()
        T.get_ai_response = ai
        res = T.transcreate(self.spec, {"midjourney_prompt": _SCENE}, verbose=False)
        self.assertLessEqual(T.x_weighted_len(res["full_tweet"]), self.spec.max_tweet_length)
        self.assertEqual(res["full_tweet"].count("#"), 2)

    # ── audience-language hashtags ─────────────────────────────────────────────

    def test_chinese_hashtags_replace_the_scaffold_tags(self):
        """The scaffolds tag the TAUGHT language (#LearnEnglish), which is
        undiscoverable for the Chinese speakers this account teaches — measured
        0 CJK hashtags across 40 live posts at ~2 impressions each."""
        spec_tags = _spec_from_dict({
            "id": "zh", "source_language": "English", "target_language": "Chinese",
            "source_flag": "\U0001F1EC\U0001F1E7", "target_flag": "\U0001F1E8\U0001F1F3",
            "script": "simplified", "max_tweet_length": 280,
            "history_file": self._tmp.name,
            "hashtag_sets": ["#\u82f1\u8bed\u5b66\u4e60 #\u6bcf\u65e5\u82f1\u8bed",
                             "#\u82f1\u8bed\u5355\u8bcd #\u5b66\u82f1\u8bed"],
        })
        ai = self._standard_ai()
        T.get_ai_response = ai
        res = T.transcreate(spec_tags, {"midjourney_prompt": _SCENE}, verbose=False)
        tweet = res["full_tweet"]
        tags = [l for l in tweet.split("\n") if l.lstrip().startswith("#")]
        self.assertEqual(len(tags), 1, tweet)
        self.assertTrue(any(0x4E00 <= ord(c) <= 0x9FFF for c in tags[0]), tags)
        self.assertNotIn("#LearnEnglish", tweet)
        self.assertLessEqual(T.x_weighted_len(tweet), spec_tags.max_tweet_length)

    def test_hashtag_sets_rotate_with_history_length(self):
        sets = ["#AAA", "#BBB", "#CCC"]
        spec_tags = _spec_from_dict({
            "id": "zh", "source_language": "English", "target_language": "Chinese",
            "script": "simplified", "max_tweet_length": 280,
            "history_file": self._tmp.name, "hashtag_sets": sets,
        })
        seen = []
        for n in range(4):
            self._write_history([{"source_word": f"w{i}", "cefr_level": "A2"} for i in range(n)])
            ai = self._standard_ai()
            T.get_ai_response = ai
            tweet = T.transcreate(spec_tags, {"midjourney_prompt": _SCENE}, verbose=False)["full_tweet"]
            seen.append(next(l for l in tweet.split("\n") if l.lstrip().startswith("#")))
        self.assertEqual(seen, ["#AAA", "#BBB", "#CCC", "#AAA"])

    def test_no_hashtag_sets_keeps_scaffold_tags(self):
        """The German path defines none and must be untouched."""
        ai = self._standard_ai()
        T.get_ai_response = ai
        tweet = T.transcreate(self.spec, {"midjourney_prompt": _SCENE}, verbose=False)["full_tweet"]
        self.assertIn("#LearnEnglish", tweet)

    # ── contract ───────────────────────────────────────────────────────────────

    def test_return_contract(self):
        ai = self._standard_ai()
        T.get_ai_response = ai
        res = T.transcreate(self.spec, {"midjourney_prompt": _SCENE}, verbose=False)
        self.assertEqual(set(res), {"full_tweet", "source_word", "source_sentence",
                                    "audience_word", "audience_sentence", "cefr", "cefr_target",
                                    "audience_word_pinyin", "audience_sentence_pinyin"})
        self.assertIn(res["source_sentence"], res["full_tweet"])
        self.assertIn(res["source_word"], res["full_tweet"])
        self.assertIn(res["audience_sentence"], res["full_tweet"])

    def test_candidate_count_follows_config(self):
        config.TRANSCREATION_CANDIDATES = 2
        ai = self._standard_ai()
        T.get_ai_response = ai
        T.transcreate(self.spec, {"midjourney_prompt": _SCENE}, verbose=False)
        self.assertEqual(ai.count("_src_"), 2)

    def test_difficult_candidate_never_reaches_picker(self):
        cands = [{"word": "generous", "sentence": "Her generous benefaction alleviates destitution."},
                 {"word": "generous", "sentence": "She is generous and gives us food."}]
        ai = self._standard_ai(_review=json.dumps({"approved_indexes": [1]}))
        T.get_ai_response = ai
        approved = T._review_candidates(self.spec, "generous", cands)
        self.assertEqual(approved, [cands[1]])
        self.assertEqual(T._pick_funniest(self.spec, approved, True, False), cands[1])
        self.assertEqual(ai.count("pick"), 0)

    def test_failed_review_regenerates_and_reviews_again(self):
        def review(ai):
            return json.dumps({"approved_indexes": [] if ai.count("_review") == 1 else [0],
                               "feedback": "Use a concrete place to explain the meaning"})
        ai = self._standard_ai(_review=review)
        T.get_ai_response = ai
        T.transcreate(self.spec, {"midjourney_prompt": _SCENE}, verbose=False)
        self.assertEqual(ai.count("_review"), 2)
        self.assertIn("Use a concrete place", ai.prompts_for("_src_")[3][0])

    def test_review_failure_never_falls_back_to_unreviewed_sentence(self):
        for response in ('{}', '{"approved_indexes": [true]}',
                         '{"approved_indexes": [99]}', '{"approved_indexes": []}', 'not JSON'):
            with self.subTest(response=response):
                ai = self._standard_ai(_review=response)
                T.get_ai_response = ai
                with self.assertRaisesRegex(ValueError, "within 3 attempts"):
                    T.transcreate(self.spec, {}, verbose=False)
                self.assertEqual(ai.count("audience"), 0)
                self.assertEqual(ai.count("_review"), 3)

    def test_pinyin_layout_in_every_scaffold(self):
        from scaffolds import _load_pool
        s2 = json.loads(_audience("拖延", "她一直拖延，迟迟不肯开始。"))
        for name, template in _load_pool():
            with self.subTest(scaffold=name):
                tweet = T._assemble(self.spec, template, "procrastinate", "I procrastinate.", s2, "")
                self.assertIn("拖延 (tuōyán)", tweet)
                lines = tweet.splitlines()
                chinese_line = next(i for i, line in enumerate(lines) if s2["audience_sentence"] in line)
                self.assertEqual(lines[chinese_line + 1], s2["audience_sentence_pinyin"])

    def test_missing_unapproved_or_numbered_pinyin_rejected(self):
        for field, value in (("audience_word_pinyin", ""), ("pinyin_ok", False),
                             ("audience_sentence_pinyin", "ta1 tuo1 yan2"),
                             ("audience_sentence_pinyin", "她拖延")):
            with self.subTest(field=field, value=value):
                data = json.loads(_audience("拖延", "她一直拖延，迟迟不肯开始。"))
                data[field] = value
                T.get_ai_response = self._standard_ai(audience=json.dumps(data))
                with self.assertRaises(ValueError):
                    T._stage2_audience(self.spec, "procrastinate", "I procrastinate.")

    def test_shorter_translation_replaces_pinyin_together(self):
        long = json.loads(_audience("拖延", "她一直拖延，迟迟不肯开始。" * 20))
        short = json.loads(_audience("拖延", "我拖延。"))
        short["audience_sentence_pinyin"] = "Wǒ tuōyán."
        T.get_ai_response = self._standard_ai(audience=json.dumps(short))
        from scaffolds import scaffold_at
        tweet = T._shrink_to_fit(self.spec, scaffold_at(0)[1], "procrastinate",
                                "I procrastinate.", long, "")
        self.assertEqual(long["audience_sentence"], short["audience_sentence"])
        self.assertEqual(long["audience_sentence_pinyin"], short["audience_sentence_pinyin"])
        self.assertIn("我拖延。", tweet)
        self.assertIn("Wǒ tuōyán.", tweet)
        self.assertLessEqual(T.x_weighted_len(tweet), 280)

    def test_website_metadata_preserves_pinyin_and_supports_legacy_records(self):
        from services.website_publish import _meta_for
        record = json.loads(_audience("拖延", "她一直拖延，迟迟不肯开始。"))
        meta = json.loads(_meta_for(record))
        self.assertEqual(meta["audience_word_pinyin"], record["audience_word_pinyin"])
        self.assertEqual(meta["audience_sentence_pinyin"], record["audience_sentence_pinyin"])
        self.assertEqual(json.loads(_meta_for({}))["audience_sentence_pinyin"], "")

    def test_history_saves_pronunciation_with_chinese(self):
        # Load the node directly: nodes/__init__.py imports optional media/X clients.
        import runpy
        from pathlib import Path
        _record_history = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                             "nodes/fanout_targets.py"))["_record_history"]
        T.get_ai_response = self._standard_ai()
        result = T.transcreate(self.spec, {}, verbose=False)
        record = _record_history(self.spec, {"cycle": 1}, result, "test-id", "", "", "")
        with open(self.spec.history_file, encoding="utf-8") as f:
            saved = json.load(f)[0]
        for field in ("audience_word", "audience_sentence", "audience_word_pinyin", "audience_sentence_pinyin"):
            self.assertEqual(saved[field], result[field])
            self.assertEqual(record[field], result[field])

    def test_overlong_example_is_regenerated_and_reviewed(self):
        def sentence(ai):
            text = "I procrastinate every day." if ai.count("_review") == 0 else "I procrastinate."
            return json.dumps({"sentence": text})
        def audience(ai):
            # Both the first translation and its shorter re-render are too long.
            if ai.count("audience") <= 2:
                return _audience("拖延", "她一直拖延，迟迟不肯开始。" * 20)
            data = json.loads(_audience("拖延", "我拖延。"))
            data["audience_sentence_pinyin"] = "Wǒ tuōyán."
            return json.dumps(data)
        ai = self._standard_ai(_src_=sentence, audience=audience)
        T.get_ai_response = ai
        result = T.transcreate(self.spec, {}, verbose=False)
        self.assertEqual(result["source_sentence"], "I procrastinate.")
        self.assertEqual(ai.count("_review"), 2)
        self.assertLessEqual(T.x_weighted_len(result["full_tweet"]), 280)

    def test_scene_and_humor_rejections_retry_before_translation(self):
        for problem in ("No humor: plain caption", "Picture shows a man holding beer, not people gathering"):
            with self.subTest(problem=problem):
                ai = self._standard_ai(_review=json.dumps({"approved_indexes": [], "feedback": problem}))
                T.get_ai_response = ai
                with self.assertRaisesRegex(ValueError, "within 3 attempts"):
                    T.transcreate(self.spec, {"midjourney_prompt": _SCENE}, verbose=False)
                self.assertEqual(ai.count("_review"), 3)
                self.assertEqual(ai.count("pick"), 0)
                self.assertEqual(ai.count("audience"), 0)
                self.assertIn(problem, ai.prompts_for("_src_")[-1][0])
                review, _ = ai.prompts_for("_review")[0]
                self.assertIn(_SCENE, review)
                self.assertIn("recognizable gentle joke", review)
                self.assertIn("Reject generic setting-only", review)

    def test_polyphonic_pinyin_preserved_and_context_requested(self):
        data = json.loads(_audience("银行", "我去银行。"))
        data.update(audience_word_pinyin="yínháng", audience_sentence_pinyin="Wǒ qù yínháng.")
        ai = self._standard_ai(audience=json.dumps(data))
        T.get_ai_response = ai
        result = T._stage2_audience(self.spec, "bank", "I go to the bank.")
        self.assertEqual(result["audience_word_pinyin"], "yínháng")
        self.assertIn("polyphonic", ai.prompts_for("audience")[0][0])


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
