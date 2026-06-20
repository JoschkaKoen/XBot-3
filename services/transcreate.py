"""
services/transcreate — turn the base cycle's content into a natural, FUNNY tweet
for a secondary language pair (e.g. English→Chinese), consistent with the SAME image.

This is the quality-critical step. It is a *transcreation*, not a literal
translation, and — crucially — it must keep the humor/grit of the base German→
English tweet, not flatten it into an image description.

How it stays funny (mirrors the German bot, which is funny *because* it generates
several candidates and picks the funniest):
  Stage 1 — TAUGHT language (e.g. English): generate N funny candidate sentences
    that PRESERVE the original joke (driven by the base funny tweet, NOT the image
    scene), then pick the funniest via TWEET_PICKER_MODEL.
  Stage 2 — AUDIENCE language (e.g. Simplified Chinese): render it into natural,
    colloquial, still-funny text and assemble the tweet.

The image scene (midjourney_prompt) is only a *guardrail* ("don't contradict the
picture"), never the creative driver.

Reuses services.ai_client.get_ai_response (token usage auto-recorded → shows up in
the cost report) and config.{TRANSCREATION_MODEL, TRANSCREATION_CANDIDATES,
TWEET_PICKER_MODEL, resolve_tweet_style, MAX_EXAMPLE_WORDS}.
"""

from __future__ import annotations

import json
import logging
import re

import config
from services.ai_client import get_ai_response
from utils.ui import info as ui_info, ok as ui_ok, tweet_box

logger = logging.getLogger("xbot.transcreate")

# Mirrors the German bot's funny-tone instruction (nodes/generate_content.py).
_FUNNY_TONE = (
    "The sentence MUST be genuinely very funny — it should make the reader laugh and smirk, "
    "and stay positive and uplifting, not cynical. CRITICAL: it must be REALISTIC and make "
    "logical sense — the humor comes from a relatable everyday situation, an ironic twist, or "
    "a witty observation, NEVER from surreal nonsense. A native speaker should read it and "
    "think 'ha, that's so true'. Keep the SPECIFIC, concrete funny detail of the original — "
    "do not generalise it into a bland description."
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


# ── Stage 1: taught language — N funny candidates, preserve the joke ───────────

def _stage1_candidates(spec, base: dict, funny: bool, n: int, verbose: bool) -> list:
    src = spec.source_language
    scene = (base.get("midjourney_prompt") or "").strip()
    funny_src = (base.get("example_sentence_target") or "").strip()  # base TARGET == our taught lang
    base_tweet = (base.get("full_tweet") or "").strip()

    system = (
        f"You are a {src} language teacher and comedy writer running a very popular X account "
        f"that teaches {src} to {spec.target_language} speakers. "
        + (_FUNNY_TONE + " " if funny else "")
        + "You always respond with valid JSON only."
    )
    scene_line = (
        f"\nThe post has an image (do NOT just describe it — only avoid contradicting it):\n{scene}\n"
        if scene else ""
    )
    user = (
        f"Here is a popular, FUNNY language-learning post. Carry its humor across into a {src} "
        f"vocabulary post for {spec.target_language} speakers — keep the joke, the irony and the "
        f"punch.\n\n"
        f"Original funny post (for its voice + the exact joke to preserve):\n{base_tweet}\n\n"
        f"The {src} line of that post — already funny, and already matches the image, so use it "
        f"as your starting point:\n{funny_src}\n"
        f"{scene_line}\n"
        f"Task:\n"
        f"1. Write a natural, idiomatic{' and genuinely funny' if funny else ''} {src} example "
        f"sentence (≤ {config.MAX_EXAMPLE_WORDS} words) that KEEPS the original joke's specific, "
        f"concrete punch. Do not turn it into a generic description.\n"
        f"2. Pick ONE useful {src} headword that APPEARS in your sentence and is worth teaching "
        f"(common noun/verb/adjective/phrase; not 'the/a/is/and', not a proper noun).\n"
        + (f"\n{_FUNNY_TONE}\n" if funny else "")
        + '\nReturn ONLY this JSON object:\n'
        '{"word": "<headword>", "sentence": "<the example sentence>", "cefr": "<A1|A2|B1|B2|C1|C2>"}'
    )

    cands: list = []
    for i in range(n):
        try:
            raw = get_ai_response(
                config.TRANSCREATION_MODEL, user, system,
                max_tokens=400, temperature=0.9,
                retry_label=f"transcreate_{spec.id}_src_{i + 1}",
            )
            d = _parse_json(raw)
        except Exception as exc:
            logger.warning("transcreate[%s] candidate %d failed: %s", spec.id, i + 1, exc)
            continue
        word = (d.get("word") or "").strip()
        sentence = (d.get("sentence") or "").strip()
        if not word or not sentence:
            continue
        cands.append({"word": word, "sentence": sentence, "cefr": (d.get("cefr") or "").strip().upper()})
        if verbose:
            ui_info(f"cand {i + 1}/{n}: {sentence}  [{word}]")
    if not cands:
        raise ValueError("stage 1 produced no usable candidates")
    return cands


def _pick_funniest(spec, cands: list, verbose: bool) -> dict:
    if len(cands) == 1:
        return cands[0]
    src = spec.source_language
    numbered = "\n".join(f"{i + 1}. {c['sentence']}   (teaches: {c['word']})" for i, c in enumerate(cands))
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


# ── Stage 2: audience language + assemble (keep it funny) ──────────────────────

def _stage2_audience(spec, s1: dict, base: dict, *, extra: str = "") -> dict:
    src, tgt = spec.source_language, spec.target_language
    scene = (base.get("midjourney_prompt") or "").strip()
    fmt = (
        f"{spec.source_flag} {s1['word']}\n"
        f"{spec.target_flag} <{tgt} meaning> <emoji>\n\n"
        f"{spec.source_flag} {s1['sentence']}\n"
        f"{spec.target_flag} <{tgt} translation> <emoji>\n\n"
        f"{spec.hashtags()}"
    )
    system = (
        f"You are a {src} teacher creating viral, funny vocabulary posts for {tgt} speakers. "
        f"You write natural, idiomatic, colloquial {tgt} ({spec.script}) that KEEPS the humor — "
        "never stiff, literal translationese. You always respond with valid JSON only."
    )
    user = (
        f"{src} word: {s1['word']}\n"
        f"{src} sentence (funny — keep it funny): {s1['sentence']}\n"
        + (f"Image guardrail (don't contradict): {scene}\n" if scene else "")
        + "\nProduce a tweet teaching this word to "
        f"{tgt} speakers:\n"
        f'1. "audience_word": the natural {tgt} meaning of the word (concise, {spec.script}).\n'
        f'2. "audience_sentence": render the sentence in natural, colloquial, FUNNY {tgt} '
        f"({spec.script}) that keeps the joke landing — a native {tgt} speaker should find it "
        "funny. NOT a literal word-for-word translation.\n"
        '3. "full_tweet": assemble EXACTLY this layout (keep the flags and the blank lines):\n\n'
        f"{fmt}\n\n"
        "Rules:\n"
        f"- All {tgt} text must use {spec.script} characters.\n"
        "- Replace each <emoji> with ONE emoji that aids understanding (not a laughing face).\n"
        f"- Keep the whole tweet under {spec.max_tweet_length} weighted characters "
        f"(each {tgt} character counts as ~2 on X).\n"
        f"{extra}\n"
        'Return ONLY this JSON object:\n'
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

def transcreate(spec, base: dict, cycle: int = 0, verbose: bool = True) -> dict:
    """
    Transcreate *base* (the German→English cycle's content) into *spec*'s pair,
    preserving the humor. Returns:
      full_tweet, source_word, source_sentence (spoken + KTV-subtitled),
      audience_word, audience_sentence, cefr.
    Raises on unrecoverable failure (the fan-out node isolates this per target).
    """
    funny = config.resolve_tweet_style(cycle) == "funny"
    n = max(1, int(getattr(config, "TRANSCREATION_CANDIDATES", 3)))

    if verbose:
        ui_info(f"Stage 1/2 — {spec.source_language}: {n} candidate(s), preserving the joke …")
    cands = _stage1_candidates(spec, base, funny, n, verbose)
    s1 = _pick_funniest(spec, cands, verbose)

    if verbose:
        ui_info(f"Stage 2/2 — {spec.target_language} ({spec.script}): translating + assembling …")
    s2 = _stage2_audience(spec, s1, base)

    # Length guard: one retry shorter if over the weighted cap.
    if x_weighted_len(s2["full_tweet"]) > spec.max_tweet_length:
        logger.info("transcreate[%s]: tweet too long (%d) — retrying shorter.",
                    spec.id, x_weighted_len(s2["full_tweet"]))
        s2 = _stage2_audience(spec, s1, base, extra=(
            f"⚠ The previous version was too long. Make both the {spec.target_language} meaning "
            f"and sentence noticeably shorter, well under {spec.max_tweet_length} weighted chars."
        ))
        if x_weighted_len(s2["full_tweet"]) > spec.max_tweet_length:
            logger.warning("transcreate[%s]: still %d weighted chars — posting may be rejected.",
                           spec.id, x_weighted_len(s2["full_tweet"]))

    result = {
        "full_tweet": s2["full_tweet"],
        "source_word": s1["word"],
        "source_sentence": s1["sentence"],
        "audience_word": s2["audience_word"],
        "audience_sentence": s2["audience_sentence"],
        "cefr": s1["cefr"],
    }
    if verbose:
        ui_info(f"{spec.source_language}: {result['source_word']} — {result['source_sentence']}")
        ui_info(f"{spec.target_language}: {result['audience_word']} — {result['audience_sentence']}")
        tweet_box(result["full_tweet"])
    return result
