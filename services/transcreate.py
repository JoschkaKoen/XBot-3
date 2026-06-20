"""
services/transcreate — turn the base cycle's content into a natural, FUNNY tweet
for a secondary language pair (e.g. English→Chinese), consistent with the SAME image.

This is the quality-critical step. It is a *transcreation*, not a literal
translation, and — crucially — it must keep the humor/grit of the base German→
English tweet, not flatten it into an image description.

How it stays funny (mirrors the German bot, which is funny *because* it generates
several DIVERSE candidates and picks the funniest):
  Stage 1 — TAUGHT language (e.g. English): generate N candidates that are genuinely
    DIFFERENT from each other — each driven by the image SCENE (the only hard
    constraint, since the image is shared) and pushed toward a distinct comedic angle,
    with the base joke as loose inspiration ("don't translate it"). Pick the funniest
    via TWEET_PICKER_MODEL.
  Stage 2 — AUDIENCE language (e.g. Simplified Chinese): render it into natural,
    colloquial, still-funny text and assemble the tweet.

Earlier versions seeded every candidate with the base English sentence ("keep the
exact joke"), which collapsed them into near-identical paraphrases — so best-of-N
had nothing to choose from. The image scene is now the creative DRIVER (it still
must not be contradicted), which restores real candidate diversity.

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
    "think 'ha, that's so true'. Use a SPECIFIC, concrete, vivid detail — never a bland, "
    "generic description."
)

# Distinct comedic angles handed one-per-candidate so the N takes actually diverge.
# (Unlike the German bot — which gets variety for free from an open-ended "write a
# joke for this WORD" prompt — our scene is fixed by the shared image, so we must
# actively push each candidate toward a different KIND of joke or they collapse into
# paraphrases of the same line.)
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


# ── Stage 1: taught language — N funny candidates, preserve the joke ───────────

def _stage1_candidates(spec, base: dict, funny: bool, n: int, verbose: bool) -> list:
    """Generate N DIVERSE funny candidates, the way the German bot does.

    The variety comes from generating fresh takes — NOT paraphrasing the base
    sentence. The shared image is the only hard constraint, so each candidate is
    driven by the image SCENE and pushed toward a different comedic angle; the
    base joke is loose inspiration, explicitly "don't translate it". Calls run in
    parallel (like nodes/generate_content), each independent at high temperature.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    src, tgt = spec.source_language, spec.target_language
    scene = (base.get("midjourney_prompt") or "").strip()
    base_en = (base.get("example_sentence_target") or "").strip()  # base TARGET == our taught lang
    base_tweet = (base.get("full_tweet") or "").strip()
    picture = scene or base_en          # what the shared image shows (must fit)
    inspiration = base_tweet or base_en  # the original take (loose inspiration only)

    system = (
        f"You are a {src} language teacher and stand-up comedy writer running a hugely popular X "
        f"account that teaches {src} to {tgt} speakers. "
        + (_FUNNY_TONE + " " if funny else "")
        + "You always respond with valid JSON only."
    )

    def _one(i: int):
        angle = _ANGLES[i % len(_ANGLES)] if funny else ""
        picture_line = (
            "The post reuses an existing image, so your sentence MUST fit that picture (don't "
            "contradict it) — but a single picture supports many different jokes, so find a FRESH "
            f"one.\nWhat the picture shows:\n{picture}\n\n" if picture else ""
        )
        inspiration_line = (
            "For inspiration ONLY — the original post's angle. Do NOT translate or paraphrase it; "
            f"invent your OWN, different joke that fits the same picture:\n{inspiration}\n\n"
            if inspiration else ""
        )
        user = (
            picture_line
            + inspiration_line
            + f"Write ONE natural, idiomatic{' and genuinely funny' if funny else ''} {src} example "
            f"sentence (≤ {config.MAX_EXAMPLE_WORDS} words) that fits the picture above.\n"
            + (f"- Build the humor on: {angle}\n" if angle else "")
            + f"- It must contain ONE useful {src} headword worth teaching (common noun/verb/"
            "adjective/phrase; not 'the/a/is/and', not a proper noun).\n"
            + (f"\n{_FUNNY_TONE}\n" if funny else "")
            + '\nReturn ONLY this JSON object:\n'
            '{"word": "<headword>", "sentence": "<the example sentence>", "cefr": "<A1|A2|B1|B2|C1|C2>"}'
        )
        try:
            raw = get_ai_response(
                config.TRANSCREATION_MODEL, user, system,
                max_tokens=400, temperature=0.95,
                retry_label=f"transcreate_{spec.id}_src_{i + 1}",
            )
            d = _parse_json(raw)
        except Exception as exc:
            logger.warning("transcreate[%s] candidate %d failed: %s", spec.id, i + 1, exc)
            return None
        word = (d.get("word") or "").strip()
        sentence = (d.get("sentence") or "").strip()
        if not word or not sentence:
            return None
        return {"word": word, "sentence": sentence, "cefr": (d.get("cefr") or "").strip().upper()}

    cands: list = []
    lock = threading.Lock()
    arrived = [0]

    def _run(i: int):
        c = _one(i)
        if c and verbose:
            with lock:
                arrived[0] += 1
                ui_info(f"cand {arrived[0]}/{n}: {c['sentence']}  [{c['word']}]")
        return c

    with ThreadPoolExecutor(max_workers=min(n, 4)) as pool:
        cands = [c for c in pool.map(_run, range(n)) if c]

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
