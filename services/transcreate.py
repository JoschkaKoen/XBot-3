"""
services/transcreate — generate a natural, FUNNY tweet for a secondary language
pair (e.g. English→Chinese) that fits the SAME reused image as the base cycle.

This is the quality-critical step. It mirrors the German→English generator's
*architecture* (nodes/generate_content) rather than translating the base tweet —
German and English humor differ, so a fresh joke lands better than a ported one.
The one shared artifact is the image; everything else is generated new:

  Step 0 — inputs from THIS target's own history (data/post_history.<id>.json):
    an avoid-list of recently taught words + (optional) a rotated CEFR level.
  Step 1 — WORD: pick a fresh taught-language word (e.g. English) the way the
    German bot does — frequent, practical, everyday — with the picture's situation
    as a topical anchor, NOT as something to describe. Two kinds of word are
    rejected because neither can carry a joke: ones naming the mood or the visible
    action (→ image captions), and bare objects with no social meaning (→ flat
    anecdotes). What is wanted is an everyday LIFE CONCEPT you could write a witty
    one-line truth about. This is the biggest single lever on funniness —
    A/B tested on real scenes, see _word_pick_call.
    Deterministically enforced against the avoid-list.
  Step 2 — SENTENCE: generate N diverse funny candidates for that fixed word
    (each pushed to a different comedic angle so best-of-N has real variety), then
    pick the funniest via TWEET_PICKER_MODEL — exactly like the German bot.
    The taught word is the ONLY hard word allowed (_VOCAB_RULE): everything around
    it must be basic vocabulary, or the learner can't read the example. "normal"
    style cycles use _LIGHT_TONE — lighter humor, never none, matching the German
    path, whose scaffold keeps asking for a [SHORT_FUNNY_SOURCE_SENTENCE] on those
    cycles too.
    Candidates are written FOR the audience (e.g. Chinese speakers learning
    English: universally relatable humor, no culture-locked references, must stay
    funny in a faithful translation) and, when the target declares a
    content_policy, steered around exactly what the downstream compliance gate
    would block — and nothing more (_POLICY_GUIDANCE).
  Step 3 — AUDIENCE: render a close, FAITHFUL audience-language (e.g. Simplified
    Chinese) translation a learner can map back to the English word-for-word, and
    assemble the tweet DETERMINISTICALLY from a rotating scaffold (shared pool),
    so layout/hashtags never drift.

Scaffold rotation is state-free (index = len(this target's history) % pool size),
so it never disturbs the primary path's persisted scaffold_state.json.

Reuses services.ai_client.get_ai_response (token usage auto-recorded → cost
report) and config.{TRANSCREATION_MODEL, TRANSCREATION_CANDIDATES, WORD_PICK_MODEL,
TWEET_PICKER_MODEL, CEFR_ROTATION, resolve_tweet_style, MAX_EXAMPLE_WORDS}.
"""

from __future__ import annotations

import json
import logging
import re

import config
from scaffolds import fill_scaffold, scaffold_at
from services.ai_client import get_ai_response
from services.history import next_cefr_level
from utils.io import safe_json_read
from utils.ui import info as ui_info, ok as ui_ok, tweet_box

logger = logging.getLogger("xbot.transcreate")

# Mirrors the German bot's funny-tone instruction (nodes/generate_content.py).
_FUNNY_TONE = (
    "The sentence MUST be genuinely very funny — it should make the reader laugh and smirk, "
    "and stay positive and uplifting, not cynical. CRITICAL: it must be REALISTIC and make "
    "logical sense — the humor comes from a relatable everyday situation, an ironic twist, or "
    "a witty observation, NEVER from surreal nonsense. A native speaker should read it and "
    "think 'ha, that's so true'. Use a SPECIFIC, concrete, vivid detail — never a bland, "
    "generic description."
)

# Per-content-policy steering for JOKE GENERATION, applied only when the target
# declares a content_policy (data/secondary_targets.json) — the same key that
# arms the downstream compliance gate (services/content_safety.py). Deliberately
# narrow: steer around exactly what that gate would block and NOTHING more, so
# the humor is not watered down (user steer 2026-07-21: adjust only as much as
# necessary for compliance — and not more).
_POLICY_GUIDANCE = {
    "china": (
        "The account's readers are in mainland China, so the post must be publishable there: no "
        "political jokes about China, its government or leaders, and none about topics that are "
        "very sensitive in China (territorial or historical-political issues, protests, "
        "censorship). That is the ONLY restriction — everyday-life humor stays sharp; do not "
        "water the joke down beyond avoiding those topics."
    ),
}

# Light-humor tone for "normal" cycles. The German path keeps asking for a
# *funny* sentence even on those (its scaffold slot is literally
# [SHORT_FUNNY_SOURCE_SENTENCE]), so dropping humor entirely here made half of
# every zh post a flat statement — "She can drop the jar into the bin." Every
# joke-less live tweet traced to a normal cycle, so normal now means *lighter*
# humor, not none.
_LIGHT_TONE = (
    "The sentence should still raise a smile — a warm, witty little observation from everyday "
    "life. It does not need a hard punchline, but it must never be a flat statement of the "
    "obvious. Keep it realistic, positive and specific."
)

# What actually separates a wooden line from a funny one here. The German bot's
# jokes are little epigrams — a general truth with a turn ("Sonntagsruhe sorgt für
# leere Regale und volle Sofas") — while the fan-out kept reporting single events
# ("He smiles after he spills his beer."), which reads flat no matter how good the
# tone instructions are. Applies on funny AND light cycles: a wry truth is the
# house voice, only its strength varies.
_JOKE_SHAPE = (
    "SHAPE — write a little TRUTH about life, not a report of one event.\n"
    "- Prefer a general observation (\"always\", \"every time\", \"until\", \"never\") over "
    "narrating what one person did.\n"
    "- Put a turn in it: a contrast, an unexpected pairing, or a reversal at the very end. Two "
    "things that clash in one short line is the strongest form.\n"
    "- Merely describing the scene, or reporting one action, is the wooden failure to avoid.\n"
    "- Vary how you open — do NOT start every sentence with \"Every\".\n"
    "- It is a teaching example, so write a complete sentence and end it with punctuation.\n"
)

# The taught word is the lesson; everything around it must be easy or the learner
# cannot read the example. Without this the model wrote "Conquering bureaucracy
# earned her this gleaming certificate today" for a C2 word — the taught word was
# the least of the reader's problems. Measured: learner-simplicity 4.0 → 5.0/5.
_VOCAB_RULE = (
    'VOCABULARY — this is a lesson{level}:\n'
    '- "{word}" is the ONLY word allowed to be difficult — it is the word being taught.\n'
    '- EVERY other word must be very common, basic {src} a beginner already knows.\n'
    '- No literary or showy words (no "savor", "gleaming", "conquering", "amidst") and no rare nouns.\n'
    '- Keep the grammar simple: one short clause, everyday phrasing.\n'
)

# Distinct comedic angles handed one-per-candidate so the N takes actually diverge
# (same word + same scene otherwise collapse into paraphrases of one line).
_ANGLES = (
    "an ironic twist — the outcome is the opposite of what you'd expect",
    "playful exaggeration / hyperbole, pushed just far enough to stay believable",
    "a relatable everyday struggle or small failure the reader has definitely lived",
    "a witty, deadpan observation with an understated punch",
    "a surprise punchline saved for the very last few words",
)


# ── X "weighted length" (CJK / emoji count ~2) ────────────────────────────────

def x_weighted_len(text: str) -> int:
    """Approximate X's weighted character count: CJK/Kana/Hangul/emoji ≈ 2, else 1."""
    total = 0
    for ch in text:
        o = ord(ch)
        wide = (
            0x1100 <= o <= 0x11FF or 0x2E80 <= o <= 0x303E or 0x3041 <= o <= 0x33FF or
            0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xA000 <= o <= 0xA4CF or
            0xAC00 <= o <= 0xD7A3 or 0xF900 <= o <= 0xFAFF or 0xFE30 <= o <= 0xFE4F or
            0xFF00 <= o <= 0xFF60 or 0xFFE0 <= o <= 0xFFE6 or 0x1F000 <= o <= 0x1FAFF or
            0x1F1E6 <= o <= 0x1F1FF or 0x20000 <= o <= 0x2FA1F
        )
        total += 2 if wide else 1
    return total


# ── JSON helpers (mirror generate_content's robust parsing) ────────────────────

def _strip_fence(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return raw.strip()


def _parse_json(raw: str) -> dict:
    data = json.loads(_strip_fence(raw))
    if isinstance(data, list):
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError("transcreation did not return a JSON object")
    return data


# ── small pure helpers ─────────────────────────────────────────────────────────

def _has_han(text: str) -> bool:
    """True if *text* contains at least one CJK ideograph (0x4E00–0x9FFF)."""
    return any(0x4E00 <= ord(c) <= 0x9FFF for c in text)


def _first_emoji(val) -> str:
    """First whitespace-delimited token of *val* — trims the model's habit of
    returning two emoji ('🖋️ 📓') where one is asked for."""
    toks = str(val or "").strip().split()
    return toks[0] if toks else ""


def _recent_words(records: list, k: int = 30) -> list:
    """Last *k* records' taught words, deduped newest-window, order preserved."""
    words = [(r.get("source_word") or "").strip() for r in records[-k:]]
    return list(dict.fromkeys(w for w in words if w))


def _clean_scene(scene: str) -> str:
    """Strip the constant image-style suffix from a stored midjourney_prompt.

    generate_image appends a fixed photographic suffix (", shot on Canon EOS R5,
    … natural warm smiles") and/or config.Z_IMAGE_PROMPT_SUFFIX to every prompt.
    Left in, that vocabulary biases every word pick toward the same cozy/camera
    cluster — fatal now that the scene DRIVES word choice. Cut at the first known
    suffix; degrade to the raw scene if the style registry is unavailable.
    """
    scene = (scene or "").strip()
    if not scene:
        return ""
    suffixes: list[str] = []
    try:
        from styles import get_style, known_styles
        for name in known_styles():
            suf = getattr(get_style(name), "midjourney_suffix", "") or ""
            if suf.strip():
                suffixes.append(suf)
    except Exception as exc:  # registry unavailable → belt-and-braces prompt line still helps
        logger.debug("transcreate: could not load style suffixes (%s)", exc)
    zsuf = (getattr(config, "Z_IMAGE_PROMPT_SUFFIX", "") or "").strip()
    if zsuf:
        suffixes.append(zsuf)
    for suf in suffixes:
        idx = scene.find(suf)
        if idx != -1:
            scene = scene[:idx]
            break
    return scene.strip(" .,\n")


def _situation(scene: str) -> str:
    """Compress a cleaned image prompt to its first sentence — the bare situation.

    The stored prompt is a lush *visual* description ("warm lamplight … gentle
    smile … quiet kindness"); feeding all of it to the word pick soaked that call
    in descriptive register and produced mood words (peaceful/convivial/repose)
    that no joke can be built on. The first sentence carries setting + subjects +
    action and drops the atmospheric tail. Measured on 6 real scenes: mood-word
    picks → concrete comic words (unwrap, spill, mosquito, wind).
    """
    scene = _clean_scene(scene)
    if not scene:
        return ""
    # ≥25 chars so a lead fragment ("Wide shot.") is skipped for the real first
    # sentence, but a genuinely short one ("A woman daydreams at a desk.") is kept.
    m = re.search(r"^(.{25,220}?)\.(?:\s|$)", scene)
    return (m.group(1) if m else scene[:220]).strip()


def _fix_quotes(spec, template: str) -> str:
    """Swap the Quotes scaffold's German typography (U+201E … U+201C) around the
    sentence for English (U+201C … U+201D) when the taught language isn't German.
    Done on the template, not post-fill — German's close quote U+201C IS English's
    open quote, so a blind character replace would corrupt it."""
    if spec.source_language == "German":
        return template
    return template.replace(
        "„[SHORT_FUNNY_SOURCE_SENTENCE]“",
        "“[SHORT_FUNNY_SOURCE_SENTENCE]”",
    )


# ── Step 1: pick a fresh, JOKE-CAPABLE taught-language word for the situation ───

def _word_pick_call(spec, scene: str, avoid_prompt: list, cefr_hint: str) -> list:
    """Ask for a shortlist of everyday words the situation *supports* — the way the
    German bot picks (frequent, practical, everyday), with the scene as a topical
    anchor rather than something to describe.

    This is the funniness bottleneck. Asking for words "evoked by the mood" made
    the model name the picture's atmosphere (peaceful, convivial, repose, quiet)
    or its visible action (give), and no joke can be built on those — the sentence
    could only restate the image. A/B tested on 6 real scenes: this wording won
    4/6 blind head-to-heads and produced unwrap / spill / mosquito / wind instead.
    """
    src, tgt = spec.source_language, spec.target_language
    if scene:
        scene_clause = (
            f"Today's post is set in this everyday situation:\n{_situation(scene)}\n\n"
        )
        anchor_clause = (
            "- Name an everyday LIFE CONCEPT people have opinions about — a habit, a small ritual, "
            "a routine, a social situation, a recurring annoyance or pleasure. The test: could you "
            "write a witty one-line truth about it?\n"
            "- It must be something people DO or RUN INTO, never a feeling or an abstract state. "
            "Even at high CEFR levels, choose the plain everyday word (holiday, break, leftovers, "
            "chore) over a formal abstract noun ('anticipation', 'respite', 'indulgence', "
            "'nostalgia', 'serenity') — abstractions produce solemn, humourless lines.\n"
            "- Belong to the world of that situation: something a person there would actually "
            "talk about, do, want, or complain about\n\n"
            "CRITICAL — do NOT pick a word that merely DESCRIBES the picture. Reject words that "
            "name what is visibly happening, and reject mood/atmosphere adjectives (e.g. "
            "'peaceful', 'quiet', 'convivial', 'cozy') — those produce caption-like sentences.\n"
            "Equally, avoid bare physical objects with no social meaning ('jar', 'board', 'lamp', "
            "'tab', 'splinter') — those only support flat anecdotes ('He spills his beer.'). The "
            "words that carry real humor hold a little everyday human drama: holiday, deadline, "
            "leftovers, small talk, shortcut, excuse, bargain, nap, chore, milestone, habit.\n"
        )
    else:
        scene_clause = anchor_clause = ""
    if cefr_hint in ("A1", "A2"):
        level_clause = (
            f"Target CEFR level: {cefr_hint} — the words MUST fit this level. At this level a "
            "common concrete word is fine; prefer the less-obvious option only when it still fits.\n"
        )
    elif cefr_hint:
        level_clause = f"Target CEFR level: {cefr_hint} — the words MUST fit this level.\n"
    else:
        level_clause = "Choose an appropriate CEFR level (A1–C2) for each word.\n"
    avoid_clause = (
        f"Do NOT reuse any of these already-taught words: {', '.join(avoid_prompt)}\n"
        if avoid_prompt else ""
    )
    system = (
        f"You are a language teacher choosing a fresh, useful {src} word to teach {tgt} speakers. "
        "You always respond with valid JSON only."
    )
    user = (
        scene_clause
        + f"Pick 3 {src} words to teach, ranked best first. Each word must:\n"
        "- Be frequently used and widespread in EVERYDAY LIFE (no jargon, no rare or literary "
        "words, not a proper noun)\n"
        "- Be practical and useful — the kind of word a learner is glad to know\n"
        + anchor_clause
        + level_clause
        + avoid_clause
        + '\nReturn ONLY this JSON object:\n'
        '{"words": [{"word": "<word>", "cefr": "<A1|A2|B1|B2|C1|C2>"}, ...]}'
    )
    raw = get_ai_response(
        config.WORD_PICK_MODEL, user, system,
        max_tokens=200, temperature=0.9, retry_label=f"transcreate_{spec.id}_word",
    )
    data = _parse_json(raw)
    opts = data.get("words")
    return opts if isinstance(opts, list) else []


def _pick_word(spec, scene: str, avoid_prompt: list, avoid_all: set,
               cefr_hint: str, verbose: bool) -> dict:
    """Pick one word not in *avoid_all*, from a ranked shortlist; one re-pick if
    every suggestion collides, then take the best anyway."""
    seen_reject: list = []
    for attempt in range(2):
        try:
            opts = _word_pick_call(spec, scene, avoid_prompt + seen_reject, cefr_hint)
        except Exception as exc:
            logger.warning("transcreate[%s] word pick failed: %s", spec.id, exc)
            opts = []
        for o in opts:
            w = (o.get("word") or "").strip()
            if w and w.lower() not in avoid_all:
                cefr = (o.get("cefr") or cefr_hint or "").strip().upper()
                if verbose:
                    ui_info(f"word: {w}" + (f"  [{cefr}]" if cefr else ""))
                return {"word": w, "cefr": cefr}
        seen_reject += [(o.get("word") or "").strip() for o in opts if o.get("word")]
        if opts:
            logger.info("transcreate[%s]: all word suggestions were repeats — re-picking.", spec.id)
    # Fallback: take the best of whatever we last saw, even if a repeat.
    if seen_reject:
        w = seen_reject[0]
        if verbose:
            ui_info(f"word: {w}  (repeat — avoid-list exhausted)")
        return {"word": w, "cefr": (cefr_hint or "").upper()}
    raise ValueError("word pick produced no usable word")


# ── Step 2: N diverse funny sentences for the fixed word ───────────────────────

def _stage1_candidates(spec, word: str, scene: str, funny: bool, n: int, verbose: bool,
                       cefr: str = "") -> list:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    src, tgt = spec.source_language, spec.target_language
    tone = _FUNNY_TONE if funny else _LIGHT_TONE
    vocab_rule = _VOCAB_RULE.format(
        level=f" for learners at {cefr} level" if cefr else " for learners",
        word=word, src=src,
    )
    scene_line = (
        f"- It should suit this picture (loose guardrail, don't contradict it): {scene}\n"
        if scene else ""
    )
    # The joke is written in the TAUGHT language but read by the AUDIENCE: it must
    # land for them, and (faithful-translation design) survive the {tgt} render.
    audience_line = (
        f"Your readers are {tgt} speakers learning {src} — the joke must land for THEM: build it "
        f"on universally relatable everyday situations rather than {src}-language wordplay or "
        f"Western pop-culture references they may not know, and prefer humor that stays funny in "
        f"a faithful {tgt} translation. "
    )
    policy_line = _POLICY_GUIDANCE.get(getattr(spec, "content_policy", "") or "", "")
    system = (
        f"You are a stand-up comedy writer and {src} teacher running a hugely popular X account "
        f"that teaches {src} to {tgt} speakers. "
        + audience_line
        + (policy_line + " " if policy_line else "")
        + tone + " "
        + "You always respond with valid JSON only."
    )

    def _one(i: int):
        # Angles drive candidate diversity, so they apply on light cycles too —
        # only the strength of the humor differs between the two tones.
        angle = _ANGLES[i % len(_ANGLES)]
        user = (
            f'Write ONE natural, idiomatic{" and genuinely funny" if funny else " and lightly witty"} '
            f'{src} example sentence (≤ {config.MAX_EXAMPLE_WORDS} words) that teaches the word '
            f'"{word}".\n'
            f'- The sentence MUST contain the word "{word}".\n'
            + scene_line
            + f"- Build the humor on: {angle}\n"
            + "\n" + _JOKE_SHAPE
            + "\n" + vocab_rule
            + f"\n{tone}\n"
            + '\nReturn ONLY this JSON object:\n'
            '{"sentence": "<the example sentence>"}'
        )
        try:
            raw = get_ai_response(
                config.TRANSCREATION_MODEL, user, system,
                max_tokens=300, temperature=0.95,
                retry_label=f"transcreate_{spec.id}_src_{i + 1}",
            )
            sentence = (_parse_json(raw).get("sentence") or "").strip()
        except Exception as exc:
            logger.warning("transcreate[%s] candidate %d failed: %s", spec.id, i + 1, exc)
            return None
        return {"word": word, "sentence": sentence} if sentence else None

    lock = threading.Lock()
    arrived = [0]

    def _run(i: int):
        c = _one(i)
        if c and verbose:
            with lock:
                arrived[0] += 1
                ui_info(f"cand {arrived[0]}/{n}: {c['sentence']}")
        return c

    with ThreadPoolExecutor(max_workers=min(n, 4)) as pool:
        cands = [c for c in pool.map(_run, range(n)) if c]

    if not cands:
        raise ValueError("stage 1 produced no usable candidates")
    # Soft-filter: keep candidates that actually contain the taught word, if any do.
    containing = [c for c in cands if word.lower() in c["sentence"].lower()]
    return containing or cands


def _pick_funniest(spec, cands: list, funny: bool, verbose: bool) -> dict:
    if len(cands) == 1:
        return cands[0]
    src = spec.source_language
    numbered = "\n".join(f"{i + 1}. {c['sentence']}" for i, c in enumerate(cands))
    if funny:
        prompt = (
            f"Choose the FUNNIEST of these {len(cands)} {src} vocabulary sentences for "
            f"{spec.target_language} learners:\n\n{numbered}\n\n"
            "Pick the one with the sharpest punchline / best ironic twist / most relatable everyday "
            "humor (warm beats cynical). Reply with ONLY the number, then a short reason — e.g. "
            "'2 — sharp, relatable punchline'."
        )
        system = (
            "You are a comedy editor picking the funniest vocabulary sentence. "
            "Reply with only the number followed by a short reason — nothing before the number."
        )
    else:
        prompt = (
            f"Choose the BEST of these {len(cands)} {src} vocabulary sentences for "
            f"{spec.target_language} learners:\n\n{numbered}\n\n"
            "Pick the one that is most natural, charming and witty — it should raise a smile. A "
            "flat statement of the obvious must lose. Reply with ONLY the number, then a short "
            "reason — e.g. '2 — warm and vivid, with a smile'."
        )
        system = (
            "You are an editor picking the most charming, quietly witty vocabulary sentence. "
            "Reply with only the number followed by a short reason — nothing before the number."
        )
    try:
        raw = get_ai_response(
            config.TWEET_PICKER_MODEL, prompt, system,
            max_tokens=60, temperature=0.0, retry_label=f"transcreate_{spec.id}_pick",
        ).strip()
        m = re.search(r"\b([1-9])\b", raw)
        idx = int(m.group(1)) - 1 if m else 0
        reason = raw[m.end():].strip(" —-") if (m and m.end() < len(raw)) else ""
        if not (0 <= idx < len(cands)):
            idx = 0
    except Exception as exc:
        logger.warning("transcreate[%s] picker failed (%s) — using first candidate.", spec.id, exc)
        idx, reason = 0, ""
    if verbose:
        ui_ok(f"selected candidate {idx + 1}{('  — ' + reason) if reason else ''}")
    return cands[idx]


# ── Step 3: faithful audience-language render (content pieces only) ─────────────

def _stage2_audience(spec, word: str, sentence: str, *, scene: str = "", extra: str = "") -> dict:
    src, tgt = spec.source_language, spec.target_language
    system = (
        f"You are a language teacher creating {src} vocabulary posts for {tgt} speakers. You produce a "
        f"close, FAITHFUL {tgt} ({spec.script}) translation a learner can map back to the {src} "
        "word-for-word — natural and colloquial, never robotic, but faithful: do NOT reinvent the "
        "joke, swap references, or add meaning. You always respond with valid JSON only."
    )
    # The situation disambiguates word senses: "she might spill everything" in a
    # beer garden means spilling a drink, not divulging secrets — without this
    # context stage 2 rendered exactly that as 泄露 ("divulge").
    context_line = (
        f"The post's picture shows: {_situation(scene)}\n"
        f"Use it ONLY to pick the right sense of an ambiguous word — never translate the picture.\n"
        if scene else ""
    )
    user = (
        f"{src} word: {word}\n"
        f"{src} sentence: {sentence}\n"
        + context_line
        + f"\nProduce, for {tgt} learners:\n"
        f'1. "audience_word": the {tgt} meaning of the word — concise, {spec.script}. It must be '
        f"the sense the word actually carries in the sentence above.\n"
        f'2. "audience_sentence": a close, faithful {tgt} ({spec.script}) translation of the '
        "sentence above — natural and idiomatic, but it MUST preserve the exact meaning so a "
        "learner can map it back to the original. Do NOT rewrite the joke or localize it.\n"
        '3. "emoji1": ONE emoji that fits the word (not a laughing face).\n'
        '4. "emoji2": ONE emoji that fits the sentence (not a laughing face).\n'
        f"- All {tgt} text must use {spec.script} characters.\n"
        + (f"{extra}\n" if extra else "")
        + '\nReturn ONLY this JSON object:\n'
        '{"audience_word": "...", "audience_sentence": "...", "emoji1": "...", "emoji2": "..."}'
    )
    raw = get_ai_response(
        config.TRANSCREATION_MODEL, user, system,
        max_tokens=300, temperature=0.7, retry_label=f"transcreate_{spec.id}_audience",
    )
    data = _parse_json(raw)
    aw = (data.get("audience_word") or "").strip()
    as_ = (data.get("audience_sentence") or "").strip()
    if not aw or not as_:
        raise ValueError("stage 2 returned empty audience fields")
    return {
        "audience_word": aw,
        "audience_sentence": as_,
        "emoji1": _first_emoji(data.get("emoji1")),
        "emoji2": _first_emoji(data.get("emoji2")),
    }


# ── assembly + length ladder ───────────────────────────────────────────────────

def _assemble(spec, template: str, word: str, sentence: str, s2: dict, cefr: str) -> str:
    return fill_scaffold(template, {
        "SOURCE_FLAG": spec.source_flag,
        "TARGET_FLAG": spec.target_flag,
        "SOURCE_LANGUAGE": spec.source_language.replace(" ", ""),
        "TARGET_LANGUAGE": spec.target_language.replace(" ", ""),
        "ARTICLE": "",                       # taught language (English) has no article slot
        "SOURCE_WORD": word,
        "TARGET_TRANSLATION": s2["audience_word"],
        "SHORT_FUNNY_SOURCE_SENTENCE": sentence,
        "TARGET_TRANSLATION_OF_SENTENCE": s2["audience_sentence"],
        "EMOJI1": s2["emoji1"],
        "EMOJI2": s2["emoji2"],
        "LEVEL": cefr,
    })


def _shorten_hashtags(full: str) -> str:
    """Reduce the tweet's hashtag line to its first tag (~−19 weighted chars)."""
    lines = full.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].lstrip().startswith("#"):
            toks = lines[i].split()
            lines[i] = toks[0] if toks else lines[i]
            break
    return "\n".join(lines)


def _shrink_to_fit(spec, template: str, word: str, sentence: str, s2: dict, cefr: str,
                   scene: str = "") -> str:
    """Bring the tweet under the weighted cap WITHOUT ever truncating the Chinese:
    (a) one shorter faithful re-render, (b) drop emoji, (c) shorten hashtags,
    (d) post as-is (X may reject, but the website mirror — primary China channel,
    no 280 limit — records it regardless; a hard CJK slice would be worse)."""
    cap = spec.max_tweet_length
    full = _assemble(spec, template, word, sentence, s2, cefr)
    if x_weighted_len(full) <= cap:
        return full

    # (a) shorter, still-faithful re-render with a stated weighted-char floor.
    overhead = x_weighted_len(_assemble(
        spec, template, word, sentence,
        {**s2, "audience_word": "", "audience_sentence": ""}, cefr))
    budget = max(10, cap - overhead)
    try:
        s2b = _stage2_audience(spec, word, sentence, scene=scene, extra=(
            f"⚠ Keep the {spec.target_language} meaning and sentence SHORT — together well under "
            f"{budget} weighted characters (each {spec.target_language} character counts ~2 on X). "
            "Stay a faithful translation."))
        cand = _assemble(spec, template, word, sentence, s2b, cefr)
        if x_weighted_len(cand) < x_weighted_len(full):
            full, s2 = cand, s2b
        if x_weighted_len(full) <= cap:
            return full
    except Exception as exc:
        logger.warning("transcreate[%s]: shorter re-render failed: %s", spec.id, exc)

    # (b) drop emoji2, then both emoji.
    for e1, e2 in ((s2["emoji1"], ""), ("", "")):
        cand = _assemble(spec, template, word, sentence, {**s2, "emoji1": e1, "emoji2": e2}, cefr)
        full = cand
        if x_weighted_len(cand) <= cap:
            return cand

    # (c) shorten the hashtag line.
    full = _shorten_hashtags(full)
    if x_weighted_len(full) <= cap:
        return full

    logger.warning("transcreate[%s]: still %d weighted chars (cap %d) — posting anyway.",
                   spec.id, x_weighted_len(full), cap)
    return full


# ── Public entry point ────────────────────────────────────────────────────────

def transcreate(spec, base: dict, cycle: int = 0, verbose: bool = True) -> dict:
    """
    Generate a fresh, funny tweet for *spec*'s language pair that fits the base
    cycle's reused image. Returns:
      full_tweet, source_word, source_sentence (spoken + KTV-subtitled),
      audience_word, audience_sentence, cefr.
    Raises on unrecoverable failure (the fan-out node isolates this per target).
    """
    funny = config.resolve_tweet_style(cycle) == "funny"
    n = max(1, int(getattr(config, "TRANSCREATION_CANDIDATES", 3)))

    records = safe_json_read(spec.history_file, default=[])
    if not isinstance(records, list):
        records = []
    avoid_all = {(r.get("source_word") or "").strip().lower()
                 for r in records if (r.get("source_word") or "").strip()}
    avoid_prompt = _recent_words(records, 30)
    cefr_hint = next_cefr_level(records) if getattr(config, "CEFR_ROTATION", False) else ""
    scene = _clean_scene(base.get("midjourney_prompt") or "")

    if verbose:
        ui_info(f"Step 1/3 — picking a fresh {spec.source_language} word evoked by the picture …")
    picked = _pick_word(spec, scene, avoid_prompt, avoid_all, cefr_hint, verbose)
    word = picked["word"]
    cefr = picked["cefr"] or cefr_hint

    if verbose:
        ui_info(f"Step 2/3 — {spec.source_language}: {n} candidate(s) for '{word}' …")
    cands = _stage1_candidates(spec, word, scene, funny, n, verbose, cefr=cefr)
    best = _pick_funniest(spec, cands, funny, verbose)
    sentence = best["sentence"]

    if verbose:
        ui_info(f"Step 3/3 — {spec.target_language} ({spec.script}): faithful translation + assemble …")
    s2 = _stage2_audience(spec, word, sentence, scene=scene)
    if not _has_han(s2["audience_sentence"]):
        logger.info("transcreate[%s]: translation had no CJK — retrying.", spec.id)
        s2 = _stage2_audience(spec, word, sentence, scene=scene,
                              extra=f"The audience_sentence MUST be written in {spec.script} characters.")

    _name, template = scaffold_at(len(records))
    template = _fix_quotes(spec, template)
    full_tweet = _shrink_to_fit(spec, template, word, sentence, s2, cefr, scene=scene)

    result = {
        "full_tweet": full_tweet,
        "source_word": word,
        "source_sentence": sentence,
        "audience_word": s2["audience_word"],
        "audience_sentence": s2["audience_sentence"],
        "cefr": cefr,
    }
    if verbose:
        ui_info(f"{spec.source_language}: {word} — {sentence}")
        ui_info(f"{spec.target_language}: {s2['audience_word']} — {s2['audience_sentence']}")
        tweet_box(full_tweet)
    return result
