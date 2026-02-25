"""
Node: generate_content

Picks a German word, generates an example sentence, translates both,
looks up the article, determines CEFR level, picks emojis, and assembles
the full tweet text.
"""

import json
import logging
import string
from typing import Any, Optional

from config import USE_TRENDS, SENTENCE_MODEL, AI_PROVIDER
from services.ai_client import get_ai_response


def _get_sentence_ai() -> callable:
    """Return the AI function to use for German sentence generation."""
    if AI_PROVIDER == "grok":
        if SENTENCE_MODEL == "flagship":
            from services.grok_ai import get_grok_flagship_response
            return get_grok_flagship_response
        if SENTENCE_MODEL == "reasoning":
            from services.grok_ai import get_grok_reasoning_response
            return get_grok_reasoning_response
    return get_ai_response
from services.deepl import translate_with_deepl
from services.get_article import get_article
from services.x_trends import get_germany_trends
from utils.retry import retry_call
from utils.ui import stage_banner, ok, tweet_box, info, warn as ui_warn

# Import lazily to avoid circular imports at module load time
def _load_strategy_from_file() -> dict:
    from nodes.analyze import load_strategy
    return load_strategy()

logger = logging.getLogger("german_bot.generate_content")


# ── helpers ───────────────────────────────────────────────────────────────────

def _strip_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] in ('"', '"', '„') and text[-1] in ('"', '"', '"'):
        text = text[1:-1]
    return text.strip()


def _strip_punctuation(text: str) -> str:
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator).strip()


def _build_word_prompt(strategy: dict) -> str:
    focus = strategy.get("focus", "")
    cefr_hint = strategy.get("preferred_cefr", "A1, A2, B1, B2, C1, C2")
    theme_hint = strategy.get("preferred_themes", "food, travel, daily life, emotions")
    avoid = strategy.get("avoid_words", [])
    avoid_str = ", ".join(avoid[-20:]) if avoid else "none"

    return (
        "You are a German language teacher creating content for an X (Twitter) account "
        "that teaches German vocabulary to English speakers.\n\n"
        "Pick ONE German word (noun, verb, adjective, or common phrase) that:\n"
        f"- Is practical and useful for everyday life\n"
        f"- Fits one of these CEFR levels: {cefr_hint}\n"
        f"- Relates to one of these themes: {theme_hint}\n"
        f"- Has NOT been recently used (avoid: {avoid_str})\n"
        f"{'- Additional focus: ' + focus if focus else ''}\n\n"
        "Reply with ONLY the German word or phrase — no explanation, no article, no punctuation."
    )


# ── trend-based word selection ────────────────────────────────────────────────

def _pick_word_from_trends(avoid_words: list) -> Optional[str]:
    """
    Fetch German trends, log them, and ask the AI to choose the best one
    that works as a German vocabulary word.

    Returns the chosen German word/phrase, or None if no suitable trend found
    or if trend fetching fails (caller should fall back to AI-free selection).
    """
    trends = get_germany_trends(max_trends=20)

    if not trends:
        # x_trends.py already logged why (timeout / parse error); one clean UI line is enough
        ui_warn("No trends available — falling back to AI word selection.")
        return None

    # Log all trends clearly to the terminal
    trend_names = [t["name"] for t in trends]
    logger.info("Fetched %d German trends:", len(trends))
    lines = "\n".join(f"  {i+1:2d}. {t['name']}" for i, t in enumerate(trends))
    print(f"\n  📈  Current German Trends ({len(trends)}):\n{lines}\n", flush=True)
    logger.debug("All trends: %s", trend_names)

    # Ask the AI for a ranked list of up to 5 suitable candidates (one API call).
    # We then walk the list and pick the first word not in avoid_words.
    trend_list_str = "\n".join(f"{i+1}. {name}" for i, name in enumerate(trend_names))
    avoid_set = {w.lower() for w in avoid_words}
    pick_req = (
        "You are a German language teacher selecting words for a vocabulary post on X (Twitter).\n\n"
        f"Here are the current trending topics in Germany:\n{trend_list_str}\n\n"
        "Your task: pick up to 5 trends that work as German vocabulary words or short phrases "
        "that an English speaker learning German (A1–C2 level) would find useful and interesting. "
        "Rank them best-first.\n\n"
        "Rules:\n"
        "- Must be an actual German word or common German phrase (not an English word, brand name, "
        "person's name, or pure hashtag with no German meaning).\n"
        "- Prefer everyday, relatable words over political or niche terms.\n"
        "- Strip any leading # or @ from each word.\n\n"
        "Respond ONLY with a valid JSON array, nothing else:\n"
        '[{"word": "<German word>", "reason": "<one sentence why>"}, ...]\n'
        'If no trend is suitable at all, respond with an empty array: []'
    )

    try:
        raw = retry_call(
            get_ai_response,
            pick_req,
            "You are a German language teacher and content strategist. You select the most teachable vocabulary from trending topics and respond only in valid JSON.",
            max_tokens=300,
            temperature=0.3,
            label="trend_pick",
        )
        raw = raw.strip().strip("```json").strip("```").strip()
        candidates: list = json.loads(raw)

        if not isinstance(candidates, list) or not candidates:
            logger.warning("AI found no suitable trend candidates.")
            ui_warn("No suitable trend words found — falling back to AI word selection.")
            return None

        # Walk ranked candidates; skip any word already used in a past post
        for entry in candidates:
            word = (entry.get("word") or "").lstrip("#@").strip()
            reason = entry.get("reason", "")
            if not word:
                continue
            if word.lower() in avoid_set:
                logger.info("Trend candidate '%s' already used — trying next.", word)
                continue
            logger.info("Trend word chosen: '%s' — %s", word, reason)
            ok(f"Trend word: '{word}' — {reason}")
            return word

        # All candidates were previously used
        logger.warning("All %d trend candidates already used — falling back to AI word selection.", len(candidates))
        ui_warn(f"All suitable trend words already used ({len(candidates)} checked) — falling back to AI word selection.")
        return None

    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Failed to parse trend-pick response (%s) — falling back.", exc)
        ui_warn("Trend word selection failed — falling back to AI word selection.")
        return None


# ── main node ─────────────────────────────────────────────────────────────────

def generate_content(state: dict) -> dict:
    stage_banner(3)
    logger.info("Node: generate_content")

    # Use strategy from state (set by analyze_and_improve earlier this cycle),
    # or fall back to the persisted file, or finally to bare defaults.
    strategy: dict = state.get("strategy") or _load_strategy_from_file()
    cycle: int = state.get("cycle", 0)

    # 1. Pick a German word (trend-based or AI-free depending on config)
    german_word: Optional[str] = None

    if USE_TRENDS:
        info("USE_TRENDS=true — fetching German trends …")
        # Build avoid list from BOTH the strategy AND the full post history so that
        # words from live tweets that aren't in the strategy yet are still excluded.
        from nodes.score import _load_history
        history_words = [r.get("german_word", "") for r in _load_history() if r.get("german_word")]
        strategy_words = strategy.get("avoid_words", [])
        # Merge, deduplicate, preserve order (history is ground truth)
        avoid_words = list(dict.fromkeys(history_words + strategy_words))
        avoid_preview = ", ".join(avoid_words[-5:]) if avoid_words else "none"
        logger.debug("Trend word selection: avoiding %d word(s): %s", len(avoid_words), avoid_words[-20:])
        logger.info("Avoiding %d previously used word(s) (e.g. %s)", len(avoid_words), avoid_preview)
        german_word = _pick_word_from_trends(avoid_words)   # None = no suitable trend found

    if not german_word:
        # Either USE_TRENDS=false, or trend fetch/selection failed → AI picks freely
        # Enrich strategy's avoid_words with full history before building the prompt
        if not USE_TRENDS:  # (already built above for the trends path)
            from nodes.score import _load_history
            history_words = [r.get("german_word", "") for r in _load_history() if r.get("german_word")]
            strategy_words = strategy.get("avoid_words", [])
            strategy = {**strategy, "avoid_words": list(dict.fromkeys(history_words + strategy_words))}
        word_prompt = _build_word_prompt(strategy)
        german_word = retry_call(
            get_ai_response,
            word_prompt,
            "You are a German language teacher creating vocabulary content for social media. You suggest single German words that are practical, interesting, and appropriate for learners.",
            max_tokens=30,
            temperature=0.9,
            label="pick_word",
        ).strip()

    # Remove any leading article the LLM might have sneaked in
    for art in ("der ", "die ", "das ", "Der ", "Die ", "Das "):
        if german_word.startswith(art):
            german_word = german_word[len(art):]
            break
    ok(f"Word selected: {german_word}")
    logger.info("Picked word: %s", german_word)

    # 2. Look up article
    article: str = get_article(german_word)
    logger.info("Article: %s", article)

    # 3. Translate word
    translated_word: str = retry_call(
        translate_with_deepl,
        german_word, "DE", "EN",
        label="DeepL word",
    )
    # Lowercase first letter for English (matches existing bot style)
    if translated_word and translated_word[0].isupper():
        translated_word = translated_word[0].lower() + translated_word[1:]
    logger.info("Translation: %s", translated_word)

    # 4. Generate emoji for the word
    word_emoji_req = (
        f'Only return one emoji which can help understand the meaning of this word: '
        f'"{translated_word}". Do not return an explanation of the emoji.'
    )
    word_emoji: str = retry_call(
        get_ai_response,
        word_emoji_req,
        "You are very good at finding the most relevant emoji for a given word or sentence.",
        max_tokens=10,
        temperature=0.8,
        label="word_emoji",
    ).strip()

    # 5. Generate example sentence in German — use the configured sentence model
    sentence_req = (
        f'Schreibe einen kurzen, einfachen und witzigen deutschen Beispielsatz, '
        f'der das Wort "{german_word}" korrekt verwendet.\n\n'
        f'Wichtige Regeln:\n'
        f'- Der Satz muss grammatikalisch korrekt sein.\n'
        f'- Das Wort "{german_word}" muss sinnvoll und natürlich eingesetzt sein.\n'
        f'- Der Satz muss realistisch sein und im echten Leben Sinn ergeben.\n'
        f'- Der Satz muss positiv, warm und motivierend sein — keine Verneinungen, keine negativen Gefühle, kein Stress, kein Unglück.\n'
        f'- Kein unsinniger oder surrealer Humor.\n'
        f'- Kurz und knapp (max. 12 Wörter).\n'
        f'- Keine Emojis. Nur den Satz zurückgeben, nichts anderes.'
    )
    sentence_ai = _get_sentence_ai()
    example_de: str = retry_call(
        sentence_ai,
        sentence_req,
        "Sie sind Deutschlehrer und schreiben klare, korrekte, natürliche und stets positive Beispielsätze.",
        max_tokens=200,
        temperature=0.7,
        label="example_sentence",
    )
    example_de = _strip_quotes(example_de)
    logger.info("Example DE: %s", example_de)

    # 6. Translate example sentence
    example_en: str = retry_call(
        translate_with_deepl,
        example_de, "DE", "EN",
        label="DeepL sentence",
    )
    example_en = _strip_quotes(example_en)
    logger.info("Example EN: %s", example_en)

    # 7. Determine CEFR level — classify the WORD, not the sentence.
    # The sentence is intentionally kept simple regardless of word difficulty,
    # so classifying the sentence always yields B1. The vocabulary item itself
    # is what the post is about, so that is what we assess.
    article_prefix = f"{article} " if article not in ("no known noun", "") else ""
    level_req = (
        f'Du bist ein erfahrener Deutschlehrer und GER-Prüfer.\n'
        f'Auf welchem GER-Niveau (A1, A2, B1, B2, C1 oder C2) wird das deutsche Wort '
        f'„{article_prefix}{german_word}" typischerweise gelehrt oder erstmals von Lernenden angetroffen?\n\n'
        f'Orientierung:\n'
        f'- A1/A2: sehr grundlegende Alltagswörter (Haus, essen, gut, Mutter)\n'
        f'- B1/B2: mittelschwerer Wortschatz, seltener, aber nützlich (Gewohnheit, erklären, überzeugend)\n'
        f'- C1/C2: fortgeschrittene, seltene oder fachsprachliche Wörter (Weltanschauung, Menschenverstand, unberechenbar)\n\n'
        f'Antworte NUR mit dem GER-Niveau, z. B. „A2". Keine Erklärung.'
    )
    cefr_level: str = retry_call(
        get_ai_response,
        level_req,
        "Du bist ein präziser GER-Vokabelklassifikator. Antworte nur mit dem Niveaukürzel.",
        max_tokens=5,
        temperature=0.1,
        label="cefr_level",
    )
    cefr_level = _strip_punctuation(cefr_level).upper()
    logger.info("CEFR level: %s", cefr_level)

    # 8. Generate emoji for the sentence — must differ from word_emoji
    sent_emoji_req = (
        f'Only return one emoji which can help understand the meaning of this sentence: '
        f'"{example_en}". '
        f'Do NOT use this emoji: {word_emoji} — pick a different one. '
        f'Do not return an explanation of the emoji.'
    )
    sentence_emoji: str = retry_call(
        get_ai_response,
        sent_emoji_req,
        "You are very good at finding the most relevant emoji for a given word or sentence.",
        max_tokens=10,
        temperature=0.8,
        label="sentence_emoji",
    ).strip()
    # Fallback: if the LLM still returned the same emoji, use a generic alternative
    if sentence_emoji == word_emoji:
        sentence_emoji = "✨"

    # 9. Assemble tweet text
    if article != "no known noun":
        word_line_de = f"🇩🇪  {article} {german_word}"
    else:
        word_line_de = f"🇩🇪  {german_word}"

    word_line_en = f"🇬🇧  {translated_word}  {word_emoji * 2}"

    full_tweet = (
        f"#DeutschLernen {cefr_level}\n\n"
        f"{word_line_de}\n"
        f"{word_line_en}\n\n"
        f"🇩🇪  {example_de}\n"
        f"🇬🇧  {example_en}  {sentence_emoji * 2}"
    )

    tweet_box(full_tweet)
    logger.info("Full tweet assembled (%d chars):\n%s", len(full_tweet), full_tweet)

    return {
        **state,
        "german_word": german_word,
        "article": article,
        "cefr_level": cefr_level,
        "word_emoji": word_emoji,
        "translated_word": translated_word,
        "example_sentence_de": example_de,
        "example_sentence_en": example_en,
        "sentence_emoji": sentence_emoji,
        "full_tweet": full_tweet,
        "cycle": cycle,
        "error": None,
    }
