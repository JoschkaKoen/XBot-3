"""
services/transcreate — turn the base cycle's content into a natural tweet for a
secondary language pair (e.g. English→Chinese), consistent with the SAME image.

This is the quality-critical step. It is a *transcreation*, not a literal
translation: the only hard constraint is that the output stays consistent with
the already-generated image (anchored on ``midjourney_prompt``); naturalness wins
over word-for-word fidelity (per product spec).

Two stages, mirroring the base bot's house style/persona/tone:
  1. Nail the TAUGHT language (e.g. English): refine a natural example sentence
     that matches the scene + pick a genuinely teachable headword.
  2. Nail the AUDIENCE language (e.g. Simplified Chinese): idiomatic translation +
     assemble the final tweet in the house multi-line scaffold.

Reuses ``services.ai_client.get_ai_response`` (token usage is auto-recorded, so
the extra cost shows up in the per-cycle cost report).
"""

from __future__ import annotations

import json
import logging

import config
from services.ai_client import get_ai_response

logger = logging.getLogger("xbot.transcreate")


# ── X "weighted length" (CJK / emoji count ~2) ────────────────────────────────

def x_weighted_len(text: str) -> int:
    """Approximate X's weighted character count: CJK/Kana/Hangul/emoji ≈ 2, else 1.

    Used to keep the assembled tweet under a free account's 280-weighted limit
    (a fresh Chinese account is not Premium, and Hanzi count double on X)."""
    total = 0
    for ch in text:
        o = ord(ch)
        wide = (
            0x1100 <= o <= 0x11FF or   # Hangul Jamo
            0x2E80 <= o <= 0x303E or   # CJK radicals / Kangxi / CJK punctuation
            0x3041 <= o <= 0x33FF or   # Hiragana, Katakana, CJK symbols
            0x3400 <= o <= 0x4DBF or   # CJK Ext A
            0x4E00 <= o <= 0x9FFF or   # CJK Unified Ideographs
            0xA000 <= o <= 0xA4CF or   # Yi
            0xAC00 <= o <= 0xD7A3 or   # Hangul syllables
            0xF900 <= o <= 0xFAFF or   # CJK compatibility
            0xFE30 <= o <= 0xFE4F or   # CJK compat forms
            0xFF00 <= o <= 0xFF60 or   # Fullwidth forms
            0xFFE0 <= o <= 0xFFE6 or
            0x1F000 <= o <= 0x1FAFF or # emoji & symbols
            0x1F1E6 <= o <= 0x1F1FF or # regional indicators (flags)
            0x20000 <= o <= 0x2FA1F    # CJK Ext B+
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


# ── Stage 1: taught language ──────────────────────────────────────────────────

def _stage1_source(spec, base: dict, funny: bool) -> dict:
    src = spec.source_language
    scene = base.get("midjourney_prompt", "") or ""
    seed = base.get("example_sentence_target", "") or ""   # base TARGET == our taught language
    ref_src = base.get("example_sentence_source", "") or ""
    ref_word = base.get("source_word", "") or ""

    tone = ""
    if funny:
        tone = (
            "\nTone: the sentence must be genuinely funny in a warm, relatable, realistic way "
            "(an everyday situation or witty twist — never surreal nonsense; it must describe "
            "something that could actually happen).\n"
        )

    system = (
        f"You are a {src} language teacher and comedy writer running a popular X account that "
        f"teaches {src} to {spec.target_language} speakers. You always respond with valid JSON only."
    )
    user = (
        f"We have an illustrated scene and need polished {src} teaching content for it.\n\n"
        f"Scene (image description — your output MUST stay consistent with this):\n{scene}\n\n"
        f"A {src} sentence describing the scene (use as a basis, improve it):\n{seed}\n"
        f"(For reference only, do not reuse verbatim — original: {ref_src} [{ref_word}])\n\n"
        "Tasks:\n"
        f"1. Write a natural, idiomatic {src} example sentence that clearly matches the scene, "
        f"at most {config.MAX_EXAMPLE_WORDS} words.\n"
        f"2. Choose ONE genuinely useful {src} headword that APPEARS in your sentence and is worth "
        "teaching to learners (a common noun/verb/adjective/phrase). Avoid function words "
        "(the, a, is, and) and proper nouns.\n"
        "3. Give the headword a CEFR level (A1–C2) appropriate for learners of "
        f"{src}.\n"
        f"{tone}\n"
        "Return ONLY this JSON object:\n"
        '{"word": "<headword>", "sentence": "<the example sentence>", "cefr": "<A1|A2|B1|B2|C1|C2>"}'
    )
    raw = get_ai_response(
        config.TRANSCREATION_MODEL, user, system,
        max_tokens=400, temperature=0.8, retry_label=f"transcreate_{spec.id}_src",
    )
    data = _parse_json(raw)
    word = (data.get("word") or "").strip()
    sentence = (data.get("sentence") or "").strip()
    if not word or not sentence:
        raise ValueError("stage 1 returned empty word/sentence")
    return {"word": word, "sentence": sentence, "cefr": (data.get("cefr") or "").strip().upper()}


# ── Stage 2: audience language + assemble ─────────────────────────────────────

def _stage2_audience(spec, s1: dict, base: dict, *, extra: str = "") -> dict:
    src = spec.source_language
    tgt = spec.target_language
    scene = base.get("midjourney_prompt", "") or ""
    fmt = (
        f"{spec.source_flag} {s1['word']}\n"
        f"{spec.target_flag} <{tgt} meaning> <emoji>\n\n"
        f"{spec.source_flag} {s1['sentence']}\n"
        f"{spec.target_flag} <{tgt} translation> <emoji>\n\n"
        f"{spec.hashtags()}"
    )
    system = (
        f"You are a {src} teacher creating viral vocabulary posts for {tgt} speakers. "
        f"You write natural, idiomatic {tgt} ({spec.script} script) — never stiff, literal "
        "translationese. You always respond with valid JSON only."
    )
    user = (
        f"Source ({src}) word: {s1['word']}\n"
        f"Source ({src}) sentence: {s1['sentence']}\n"
        f"Scene (stay consistent): {scene}\n\n"
        "Produce a tweet teaching this word to "
        f"{tgt} speakers:\n"
        f"1. \"audience_word\": the natural {tgt} meaning of the word (concise, {spec.script}).\n"
        f"2. \"audience_sentence\": a natural, idiomatic {tgt} rendering of the sentence "
        f"({spec.script}) — NOT word-for-word; it must read naturally to a native speaker and "
        "match the scene.\n"
        "3. \"full_tweet\": assemble EXACTLY this layout (keep the flags and the two blank lines):\n\n"
        f"{fmt}\n\n"
        "Rules:\n"
        f"- All {tgt} text must use {spec.script} characters.\n"
        "- Replace each <emoji> with ONE emoji that aids understanding (not a laughing face).\n"
        f"- Keep the whole tweet concise — under {spec.max_tweet_length} weighted characters "
        f"(each {tgt} character counts as ~2 on X).\n"
        f"{extra}\n"
        "Return ONLY this JSON object:\n"
        '{"audience_word": "...", "audience_sentence": "...", "full_tweet": "..."}'
    )
    raw = get_ai_response(
        config.TRANSCREATION_MODEL, user, system,
        max_tokens=800, temperature=0.7, retry_label=f"transcreate_{spec.id}_audience",
    )
    data = _parse_json(raw)
    full_tweet = (data.get("full_tweet") or "").strip()
    if not full_tweet:
        raise ValueError("stage 2 returned empty full_tweet")
    return {
        "audience_word": (data.get("audience_word") or "").strip(),
        "audience_sentence": (data.get("audience_sentence") or "").strip(),
        "full_tweet": full_tweet,
    }


# ── Public entry point ────────────────────────────────────────────────────────

def transcreate(spec, base: dict, cycle: int = 0) -> dict:
    """
    Transcreate *base* (the German→English cycle's content) into *spec*'s pair.

    Returns a dict:
      full_tweet       — assembled tweet text for spec's account (length-checked)
      source_word      — taught-language headword
      source_sentence  — taught-language sentence (this is what gets SPOKEN + KTV-subtitled)
      audience_word    — audience-language meaning
      audience_sentence— audience-language sentence
      cefr             — CEFR of the headword
    Raises on unrecoverable failure (the fan-out node isolates this per target).
    """
    funny = config.resolve_tweet_style(cycle) == "funny"

    s1 = _stage1_source(spec, base, funny)
    s2 = _stage2_audience(spec, s1, base)

    # Length guard: one retry with a stricter instruction if over the weighted cap.
    if x_weighted_len(s2["full_tweet"]) > spec.max_tweet_length:
        logger.info(
            "transcreate[%s]: tweet weighted-len %d > %d — retrying shorter.",
            spec.id, x_weighted_len(s2["full_tweet"]), spec.max_tweet_length,
        )
        s2 = _stage2_audience(
            spec, s1, base,
            extra=(
                f"⚠ The previous version was too long. Make BOTH the {spec.target_language} "
                f"meaning and sentence noticeably shorter so the whole tweet is well under "
                f"{spec.max_tweet_length} weighted characters."
            ),
        )
        if x_weighted_len(s2["full_tweet"]) > spec.max_tweet_length:
            logger.warning(
                "transcreate[%s]: still %d weighted chars after retry — posting may be rejected.",
                spec.id, x_weighted_len(s2["full_tweet"]),
            )

    return {
        "full_tweet": s2["full_tweet"],
        "source_word": s1["word"],
        "source_sentence": s1["sentence"],
        "audience_word": s2["audience_word"],
        "audience_sentence": s2["audience_sentence"],
        "cefr": s1["cefr"],
    }
