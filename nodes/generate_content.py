"""
Node: generate_content

Picks a German word (trend-based or AI-free), then generates the complete
tweet in a single Grok call that returns JSON.  Local post-processing handles
tweet-length enforcement.
"""

import json
import logging
import string
from typing import Any, Optional

from config import (
    USE_TRENDS, TWEET_MODEL, TWEET_PICKER_MODEL, AI_PROVIDER, FUNNY_MODE,
    TREND_FILTER_MODEL, WORD_PICK_MODEL, SIMILARITY_MODEL, MAX_TWEET_LENGTH,
)
from services.ai_client import get_ai_response
from services.x_trends import get_germany_trends
from utils.retry import retry_call
from utils.ui import stage_banner, ok, tweet_box, info, warn as ui_warn


def _get_tweet_ai() -> callable:
    """Return the AI function to use for tweet generation."""
    if AI_PROVIDER == "grok":
        if TWEET_MODEL == "flagship":
            from services.grok_ai import get_grok_flagship_response
            return get_grok_flagship_response
        if TWEET_MODEL == "reasoning":
            from services.grok_ai import get_grok_reasoning_response
            return get_grok_reasoning_response
    return get_ai_response


def _get_tweet_picker_ai() -> callable:
    """Return the AI function to use for picking the best tweet candidate (TWEET_PICKER_MODEL)."""
    return _model_to_ai_fn(TWEET_PICKER_MODEL)


def _model_to_ai_fn(model: str) -> callable:
    """Resolve a model name string to the corresponding Grok AI function."""
    if AI_PROVIDER == "grok":
        if model == "flagship":
            from services.grok_ai import get_grok_flagship_response
            return get_grok_flagship_response
        if model == "reasoning":
            from services.grok_ai import get_grok_reasoning_response
            return get_grok_reasoning_response
        from services.grok_ai import get_grok_response
        return get_grok_response
    return get_ai_response


def _get_trend_filter_ai() -> callable:
    """AI function for filtering trend keywords (TREND_FILTER_MODEL)."""
    return _model_to_ai_fn(TREND_FILTER_MODEL)


def _get_word_pick_ai() -> callable:
    """AI function for free-form word selection (WORD_PICK_MODEL)."""
    return _model_to_ai_fn(WORD_PICK_MODEL)


def _get_similarity_ai() -> callable:
    """AI function for semantic duplicate / similarity check (SIMILARITY_MODEL)."""
    return _model_to_ai_fn(SIMILARITY_MODEL)


def _load_strategy_from_file() -> dict:
    from nodes.analyze import load_strategy
    return load_strategy()


logger = logging.getLogger("german_bot.generate_content")


# ── helpers ───────────────────────────────────────────────────────────────────

def _strip_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] in ('"', '\u201c', '\u201e') and text[-1] in ('"', '\u201d', '\u201c'):
        text = text[1:-1]
    return text.strip()


def _strip_punctuation(text: str) -> str:
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator).strip()


def _truncate_emoji_pairs(tweet: str) -> str:
    """Replace doubled trailing emoji pairs (🚗🚗 → 🚗) to shorten the tweet."""
    lines = tweet.split("\n")
    result = []
    for line in lines:
        if "  " in line:
            idx = line.rfind("  ")
            prefix = line[: idx + 2]
            suffix = line[idx + 2 :].strip()
            n = len(suffix)
            if n % 2 == 0 and n > 0 and suffix[: n // 2] == suffix[n // 2 :]:
                line = prefix + suffix[: n // 2]
        result.append(line)
    return "\n".join(result)


def _build_word_prompt(strategy: dict) -> str:
    style = strategy.get("style", "")
    next_topic = strategy.get("next_topic", "")
    cefr_hint = strategy.get("preferred_cefr", "A1, A2, B1, B2, C1, C2")
    avoid = strategy.get("avoid_words", [])
    avoid_str = ", ".join(avoid[-20:]) if avoid else "none"

    topic_line = f"- Thema / Fokus für diesen Tweet: {next_topic}\n" if next_topic else ""
    style_line  = f"- Stilhinweis: {style}\n" if style else ""

    return (
        "Du bist ein Deutschlehrer und erstellst Inhalte für einen X-Account, "
        "der englischsprachigen Nutzern deutsche Vokabeln beibringt.\n\n"
        "Wähle GENAU EIN deutsches Wort (Nomen, Verb, Adjektiv oder gebräuchliche Phrase), das:\n"
        f"- Im Alltag praktisch und nützlich ist\n"
        f"- Zu einem dieser GER-Niveaus passt: {cefr_hint}\n"
        f"{topic_line}"
        f"- Nicht kürzlich verwendet wurde (vermeiden: {avoid_str})\n"
        f"{style_line}\n"
        "Antworte NUR mit einem gültigen JSON-Objekt – keine Erklärung, kein Artikel außer im JSON:\n"
        '{"word": "<das deutsche Wort ohne Artikel>", "cefr": "<A1|A2|B1|B2|C1|C2>"}'
    )


def _build_tweet_prompt(
    trending_word: str,
    scaffold: str,
    strategy: dict,
    top_tweets: list,
    cefr_level: str = "",
    extra_instruction: str = "",
    word_from_trends: bool = False,
) -> str:
    preferred_cefr = strategy.get("preferred_cefr", "A1, A2, B1, B2, C1, C2")
    # next_topic and style only make sense when the word was chosen freely (not from trends).
    # When word_from_trends=True the topic is already fixed by the trend; injecting a
    # separate next_topic would contradict the chosen word.
    next_topic = "" if word_from_trends else strategy.get("next_topic", "")
    style      = "" if word_from_trends else strategy.get("style", "")

    examples_section = ""
    for i, tweet in enumerate(top_tweets[:3], 1):
        score = tweet.get("engagement_score", 0.0)
        text = tweet.get("full_tweet", "")
        examples_section += f"\n### Tweet {i} (score: {score}):\n{text}\n"
    if not examples_section:
        examples_section = "\n(No past tweets available yet)\n"

    funny_tone_section = ""
    if FUNNY_MODE:
        funny_tone_section = (
            "## Tone\n"
            "The example sentence MUST be genuinely very funny — it should make the reader laugh and smirk. "
            # "A sentence that is merely pleasant or informative is not acceptable.\n\n"
        )

    cefr_rule = (
        f"- The CEFR level for this word is **{cefr_level}** — use this level in the tweet\n"
        f"- The grammar and vocabulary of the example sentence should match {cefr_level}: "
        # + (
        #     "very simple present/past tense, everyday words, short sentences\n"
        #     if cefr_level in ("A1", "A2") else
        #     "moderate complexity, some subordinate clauses allowed\n"
        #     if cefr_level in ("B1", "B2") else
        #     "idiomatic expressions and complex structures are welcome\n"
        # )
        if cefr_level else
        "- Pick an appropriate CEFR level (A1-C2) for this word\n"
        "- The grammar and vocabulary of the example sentence should match the CEFR level: "
        "A1/A2 → simple; B1/B2 → moderate; C1/C2 → complex/idiomatic\n"
    )

    prompt = (
        "You are a German language teacher and comedian running a very popular X (Twitter) account that teaches "
        "German to English speakers.\n\n"
        f"## Your task\n"
        f'Create a complete tweet about this German word: "{trending_word}"\n\n'
        f"## Tweet format (follow this scaffold exactly)\n"
        f"{scaffold}\n\n"
        f"{funny_tone_section}"
        "## Rules\n"
        "- The German word must include the correct article (der/die/das) if it's a noun\n"
        "- If the word has no grammatical article (adjective, phrase, verb, etc.), omit the [ARTICLE] slot entirely\n"
        f"{cefr_rule}"
        #"- The example sentence must be short, warm, and authentic — something a German would actually say\n"
        #"- Vary the sentence ending: most sentences should end with a period (.) — only use an exclamation mark (!) when it is genuinely funny or surprising, and occasionally use a question (?)\n"
        "- The example sentence MUST contain the exact word\n"
        "- The English translations must be natural, not robotic\n"
        "- The emojis should help the reader visually understand the meaning. Don't just use laughing emojis\n"
       
        # "- Each [EMOJI1] or [EMOJI2] placeholder stands for exactly ONE emoji character. "
        # "The scaffold writes them twice (e.g. [EMOJI1][EMOJI1]) to show you should repeat the same emoji. "
        # "Replace the entire [EMOJI1][EMOJI1] with exactly 2 identical emojis (e.g. 🚗🚗). "
        # "Do NOT add any emojis beyond what the scaffold specifies.\n"
        # "- Preserve the exact spacing from the scaffold: if the scaffold shows two spaces between an emoji and the text "
        # "(e.g. '🇩🇪  [GERMAN_WORD]'), use exactly two spaces in the output. Do not collapse spaces or add more.\n"
       
        #"- Each line gets a pair of 2 identical emojis that visually represent the meaning\n"
        # "- The two emoji PAIRS must be DIFFERENT from each other "
        # "(e.g. 🚗🚗 for word line, 🎉🎉 for sentence line — NOT the same pair twice)\n"
        # "- Do NOT wrap the output in quotes or markdown\n\n"
        "## Strategy guidance\n"
        f"- Preferred CEFR levels: {preferred_cefr}\n"
        + (f"- Topic / angle for this tweet: {next_topic}\n" if next_topic else "")
        + (f"- Style instruction: {style}\n" if style else "")
        + "\n"
        "## Output\n"
        "Return ONLY a single valid JSON object with these fields:\n\n"
        "{\n"
        '  "tweet": "<the complete tweet text, exactly matching the scaffold>",\n'
        '  "german_word": "<the bare German word without article, e.g. Führerschein>",\n'
        '  "article": "<der / die / das / no known noun>",\n'
        '  "cefr_level": "<A1 / A2 / B1 / B2 / C1 / C2>",\n'
        '  "example_sentence_de": "<just the German example sentence>",\n'
        '  "example_sentence_en": "<just the English translation of the sentence>"\n'
        "}\n\n"
        "Just the JSON object."
    )

    if extra_instruction:
        prompt += f"\n\n⚠️ IMPORTANT: {extra_instruction}"

    return prompt


def _call_tweet_ai(
    trending_word: str,
    scaffold: str,
    strategy: dict,
    top_tweets: list,
    tweet_ai: callable,
    cefr_level: str = "",
    extra_instruction: str = "",
    word_from_trends: bool = False,
) -> list:
    """Fire 3 parallel API calls, each generating one tweet candidate. Returns list of dicts."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    prompt = _build_tweet_prompt(trending_word, scaffold, strategy, top_tweets, cefr_level, extra_instruction, word_from_trends=word_from_trends)
    if FUNNY_MODE:
        system_prompt = (
            "You are a German language teacher and comedy writer creating funny tweets for X (Twitter). "
            "Every example sentence must be genuinely very funny — it must make the reader laugh."
            # "It should be positive and uplifting not negative and cynical."
            "You always respond with valid JSON only."
        )
    else:
        system_prompt = (
            "You are a German language teacher and comedian creating funny tweets for "
            "X (Twitter). You always respond with valid JSON only."
        )

    _R = "\033[0m"
    _BOLD = "\033[1m"
    _CYAN = "\033[96m"
    _GRAY = "\033[90m"
    _GREEN = "\033[92m"

    arrived = [0]
    lock = threading.Lock()

    def _single_call(idx: int) -> dict:
        raw = retry_call(
            tweet_ai,
            prompt,
            system_prompt,
            max_tokens=700,
            temperature=0.9,
            label=f"generate_tweet_{idx}",
        )
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        result = json.loads(raw)
        if isinstance(result, list):
            result = result[0]  # graceful fallback if AI returns array
        with lock:
            arrived[0] += 1
            tweet_text = result.get("tweet", "")
            print(f"\n  {_GRAY}── Candidate {idx} ({arrived[0]}/3) ────────────────────────────{_R}", flush=True)
            for line in tweet_text.splitlines():
                print(f"  {_GRAY}│{_R}  {line}", flush=True)
        return result

    print(f"\n  {_CYAN}{_BOLD}⚡  Generating 3 tweet candidates in parallel…{_R} (printing each as it arrives)\n", flush=True)

    candidates_map: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_single_call, i + 1): i + 1 for i in range(3)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                candidates_map[idx] = future.result()
            except Exception as exc:
                logger.warning("Candidate %d failed: %s", idx, exc)

    # Return in original order (1, 2, 3); drop any that failed
    return [candidates_map[i] for i in sorted(candidates_map)]


def _select_best_tweet(candidates: list, german_word: str, cefr_level: str) -> dict:
    """
    Print all candidates to the terminal, use the configured TWEET_PICKER_MODEL to pick
    the best one, and highlight the winner. Falls back to the first candidate if selection fails.
    """

    _R    = "\033[0m"
    _BOLD = "\033[1m"
    _CYAN = "\033[96m"
    _GRAY = "\033[90m"
    _GREEN = "\033[92m"

    if len(candidates) == 1:
        print(f"\n  {_GREEN}✔  Only one candidate — selected automatically.{_R}\n", flush=True)
        return candidates[0]

    numbered = "\n\n".join(
        f"Candidate {i + 1}:\n{c.get('tweet', '')}"
        for i, c in enumerate(candidates)
    )

    _fmt_instruction = (
        "Your entire response must be exactly this format and nothing else:\n"
        "<N> — <reason up to 12 words>\n"
        "where <N> is the candidate number (1, 2 or 3). "
        "Do NOT write 'Candidate', do NOT add any text before the number. "
        "Example: 2 — sharpest punchline, warm and instantly relatable"
    )

    if FUNNY_MODE:
        selection_prompt = (
            f"You are evaluating {len(candidates)} German vocabulary tweet candidates for the word "
            f"'{german_word}' (CEFR: {cefr_level or 'unknown'}).\n\n"
            f"{numbered}\n\n"
            "Pick the BEST tweet using these criteria:\n"
            "1. FUNNINESS (primary): Which sentence has the sharpest punchline, the best ironic twist, "
            "or the most absurd contrast? Would a real person laugh or smirk? "
            "A genuinely funny tweet always beats a merely pleasant one.\n"
            "2. POSITIVITY (tiebreaker): If two candidates are equally funny, pick the one that feels "
            "warm, uplifting, and good-natured. Avoid anything that feels mean, cynical, or negative.\n\n"
            f"{_fmt_instruction}\n\n"
            "Your answer:"
        )
        system_prompt_selector = (
            "You are a comedy editor selecting the funniest German vocabulary tweet. "
            "Your two criteria are: (1) funniness — always pick the sharpest joke, "
            "(2) positivity as tiebreaker — warm beats cynical. "
            "Reply with ONLY a number (1, 2 or 3) followed by a short reason — nothing else before the number."
        )
    else:
        selection_prompt = (
            f"You are evaluating {len(candidates)} German vocabulary tweet candidates for the word "
            f"'{german_word}' (CEFR: {cefr_level or 'unknown'}).\n\n"
            f"{numbered}\n\n"
            "Score each candidate on these criteria and pick the BEST overall:\n"
            "- Warmth & authenticity: Is the example sentence natural and relatable?\n"
            "- Engagement potential: Would a native English speaker stop scrolling and interact with this tweet?\n"
            "- Shareability: Is it quotable, relatable, or surprising enough to share?\n"
            "- Learning effect: Does the example sentence make the German word memorable and easy to understand?\n"
            "- Sentence quality: Is the German sentence natural, correctly matched to the CEFR level, and short enough to read at a glance?\n\n"
            f"{_fmt_instruction}\n\n"
            "Your answer:"
        )
        system_prompt_selector = (
            "You are a social media expert and German teacher evaluating tweet quality. "
            "Reply with ONLY a number (1, 2 or 3) followed by a short reason — nothing else before the number."
        )

    chosen_idx = 0
    reason = ""
    try:
        import re as _re
        raw = retry_call(
            _get_tweet_picker_ai(),
            selection_prompt,
            system_prompt_selector,
            max_tokens=60,
            temperature=0.0,
            label="select_tweet",
        )
        raw = raw.strip()
        m = _re.search(r'\b([123])\b', raw)
        if m:
            idx = int(m.group(1)) - 1
            reason = raw[m.end():].strip(" —-") if m.end() < len(raw) else ""
            if 0 <= idx < len(candidates):
                chosen_idx = idx
                logger.info("Tweet selection: picked candidate %d — %s", idx + 1, reason)
            else:
                logger.warning("Tweet selection: parsed index %d out of range — using first candidate.", idx + 1)
        else:
            logger.warning("Tweet selection: could not parse a candidate number from %r — using first candidate.", raw)
    except Exception as exc:
        logger.warning("Tweet selection failed (%s) — using first candidate.", exc)

    reason_str = f"  {_GRAY}{reason}{_R}" if reason else ""
    print(f"\n  {_GREEN}{_BOLD}✔  Selected: Candidate {chosen_idx + 1}{_R}{reason_str}\n", flush=True)
    return candidates[chosen_idx]


# ── trend-based word selection ─────────────────────────────────────────────────

def _pick_word_from_trends(avoid_words: list) -> Optional[tuple[str, str]]:
    """
    Fetch German trends, log them, and ask the AI to choose the best one
    that works as a German vocabulary word.

    Returns (word, cefr_level) tuple, or None if no suitable trend found
    or if trend fetching fails (caller should fall back to AI-free selection).
    """
    trends = get_germany_trends(max_trends=20)

    if not trends:
        ui_warn("No trends available — falling back to AI word selection.")
        return None

    trend_names = [t["name"] for t in trends]
    logger.info("Fetched %d German trends:", len(trends))
    lines = "\n".join(f"  {i+1:2d}. {t['name']}" for i, t in enumerate(trends))
    print(f"\n  📈  Current German Trends ({len(trends)}):\n{lines}\n", flush=True)
    logger.debug("All trends: %s", trend_names)

    trend_list_str = "\n".join(f"{i+1}. {name}" for i, name in enumerate(trend_names))
    avoid_set = {w.lower() for w in avoid_words}
    pick_req = (
        "Du bist ein Deutschlehrer und wählst Wörter für einen Vokabelbeitrag auf X (Twitter).\n\n"
        f"Hier sind die aktuellen Trendthemen in Deutschland:\n{trend_list_str}\n\n"
        "Deine Aufgabe: Extrahiere aus diesen Trends bis zu 15 deutsche Wörter oder kurze Phrasen, "
        "die für einen englischsprachigen Deutschlernenden (A1–C2) wertvoll sind.\n\n"
        "WICHTIG — Sortierkriterium: Sortiere ausschließlich nach dem Nutzen für Deutschlernende, "
        "NICHT nach der Relevanz für das Trendthema. Das Wort 'Heizung' ist wertvoller als 'Pipeline', "
        "auch wenn 'Pipeline' näher am Trendthema liegt.\n\n"
        "Rangfolge von oben nach unten:\n"
        "  1. Alltagswörter, die jeder Deutschlernende kennen muss (Substantive, Verben, Adjektive des täglichen Lebens)\n"
        "  2. Kulturell typisch deutsche Wörter mit hohem Wiedererkennungswert\n"
        "  3. Thematisch passende Wörter mit mittlerem Lernwert\n"
        "  4. Fachbegriffe oder weniger gebräuchliche Wörter — nur wenn kein besseres Wort vorhanden\n\n"
        "Ausschlusskriterien (diese Wörter NICHT aufnehmen):\n"
        "- Englische Lehnwörter oder Internationalismen ohne deutschen Lernwert (Pipeline, Internet, App, Livestream …)\n"
        "- Eigennamen, Markennamen, Städte, Personennamen\n"
        "- Seltene oder veraltete Wörter, die im echten Alltag kaum vorkommen\n"
        "- Politische Fachbegriffe, die nur im Nachrichtenkontext auftauchen\n"
        "- Entferne führende # oder @ von jedem Wort\n\n"
        "Antworte NUR mit einem gültigen JSON-Array, nichts anderes:\n"
        '[{"word": "<deutsches Wort>", "cefr": "<A1|A2|B1|B2|C1|C2>"}, ...]\n'
        "Falls kein Trend ein geeignetes Wort liefert, antworte mit einem leeren Array: []"
    )

    try:
        raw = retry_call(
            _get_trend_filter_ai(),
            pick_req,
            "Du bist ein erfahrener Deutschlehrer. Dein einziges Ziel ist es, Wörter auszuwählen, die für englischsprachige Deutschlernende im echten Alltag möglichst nützlich sind. Trendrelevanz ist zweitrangig — Lernwert hat absolute Priorität. Du antwortest ausschließlich mit gültigem JSON.",
            max_tokens=1200,
            temperature=0.3,
            label="trend_pick",
        )
        text = raw.strip()
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidates: list = json.loads("\n".join(lines))

        if not isinstance(candidates, list) or not candidates:
            logger.warning("AI found no suitable trend candidates.")
            ui_warn("No suitable trend words found — falling back to AI word selection.")
            return None

        # Print full ranked list in original order, marking used/skipped words inline
        _R     = "\033[0m"
        _BOLD  = "\033[1m"
        _GREEN = "\033[92m"
        _RED   = "\033[91m"
        _GRAY  = "\033[90m"
        _CYAN  = "\033[96m"

        chosen_word = None
        chosen_cefr = ""
        free_count  = 0

        _VALID_CEFR = {"A1", "A2", "B1", "B2", "C1", "C2"}

        # Pre-scan to find chosen word and count free slots
        clean_entries = []
        for entry in candidates:
            word = (entry.get("word") or "").lstrip("#@").strip()
            if not word:
                continue
            cefr = (entry.get("cefr") or "").strip().upper()
            if cefr not in _VALID_CEFR:
                cefr = "?"
            word_tokens = set(word.lower().split())
            used = word.lower() in avoid_set or bool(word_tokens & avoid_set)
            clean_entries.append((word, cefr, used))
            if not used:
                free_count += 1
                if chosen_word is None:
                    chosen_word = word
                    chosen_cefr = cefr

        print(f"\n  {_CYAN}{_BOLD}🏆  Trend word shortlist  ({free_count} free):{_R}", flush=True)
        for i, (word, cefr, used) in enumerate(clean_entries, 1):
            cefr_fmt = f"{_CYAN}{cefr:<3}{_R}"
            if used:
                status   = f"{_RED}✖ already used{_R}"
                word_fmt = f"{_GRAY}{word:<20}{_R}"
            elif word == chosen_word:
                status   = f"{_GREEN}✔ selected{_R}"
                word_fmt = f"{_BOLD}{word:<20}{_R}"
            else:
                status   = f"{_GRAY}○ free{_R}"
                word_fmt = f"{word:<20}"
            print(f"  {_GRAY}{i}.{_R} {word_fmt} {cefr_fmt} {status}", flush=True)

        print(flush=True)

        if chosen_word:
            logger.info("Trend word chosen: '%s' (CEFR: %s)", chosen_word, chosen_cefr)
            ok(f"Trend word: '{chosen_word}' [{chosen_cefr}]")
            return chosen_word, chosen_cefr

        logger.warning("All %d trend candidates already used — falling back to AI word selection.", len(candidates))
        ui_warn(f"All suitable trend words already used ({len(candidates)} checked) — falling back to AI word selection.")
        return None

    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Failed to parse trend-pick response (%s) — falling back.", exc)
        ui_warn("Trend word selection failed — falling back to AI word selection.")
        return None


_VALID_CEFR = {"A1", "A2", "B1", "B2", "C1", "C2"}


def _is_word_too_similar(word: str, avoid_words: list) -> tuple[bool, str]:
    """
    Use Grok fast non-reasoning to check whether *word* is morphologically or
    semantically too close to any word in *avoid_words* (e.g. deutsch/deutsche,
    Freund/Freundschaft).  Returns (is_similar, matched_word).
    """
    if not avoid_words:
        return False, ""

    recent = avoid_words[-50:]
    word_list_str = ", ".join(recent)

    prompt = (
        f"Bereits verwendete deutsche Wörter: [{word_list_str}]\n\n"
        f"Neues vorgeschlagenes Wort: \"{word}\"\n\n"
        "Ist dieses Wort zu ähnlich zu einem der bereits verwendeten Wörter? "
        "Zu ähnlich bedeutet: gleicher Wortstamm, Deklination, Konjugation, "
        "Kompositum oder eng verwandte Ableitung "
        "(z.B. deutsch/deutsche, Haus/Häuser, laufen/Lauf, Freund/Freundschaft).\n\n"
        "Antworte NUR mit einem JSON-Objekt:\n"
        '{"similar": true, "matched": "<das ähnliche Wort>"}\n'
        "oder\n"
        '{"similar": false, "matched": ""}'
    )

    try:
        raw = retry_call(
            _get_similarity_ai(),
            prompt,
            "Du prüfst, ob ein deutsches Wort zu ähnlich zu einer Liste bereits verwendeter Wörter ist. "
            "Antworte ausschließlich mit gültigem JSON.",
            max_tokens=60,
            temperature=0.0,
            label="similarity_check",
        ).strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        data = json.loads(raw)
        is_similar = bool(data.get("similar", False))
        matched = data.get("matched", "")
        return is_similar, matched
    except Exception as exc:
        logger.warning("Similarity check failed (%s) — allowing word.", exc)
        return False, ""


# ── main node ──────────────────────────────────────────────────────────────────

def generate_content(state: dict) -> dict:
    stage_banner(3)
    logger.info("Node: generate_content")

    strategy: dict = state.get("strategy") or _load_strategy_from_file()
    cycle: int = state.get("cycle", 0)

    # ── 1. Pick a German word + determine its CEFR level ──────────────────────
    german_word: Optional[str] = None
    word_cefr: str = ""
    avoid_words: list = []
    word_from_trends: bool = False

    if USE_TRENDS:
        info("USE_TRENDS=true — fetching German trends …")
        from nodes.score import _load_history
        history_words = [r.get("german_word", "") for r in _load_history() if r.get("german_word")]
        strategy_words = strategy.get("avoid_words", [])
        avoid_words = list(dict.fromkeys(history_words + strategy_words))
        avoid_preview = ", ".join(avoid_words[-5:]) if avoid_words else "none"
        logger.debug("Trend word selection: avoiding %d word(s): %s", len(avoid_words), avoid_words[-20:])
        logger.info("Avoiding %d previously used word(s) (e.g. %s)", len(avoid_words), avoid_preview)
        trend_result = _pick_word_from_trends(avoid_words)
        if trend_result:
            german_word, word_cefr = trend_result
            word_from_trends = True

    if not german_word:
        if not USE_TRENDS:
            from nodes.score import _load_history
            history_words = [r.get("german_word", "") for r in _load_history() if r.get("german_word")]
        # Always enrich avoid_words with the full history before building the word prompt.
        # When USE_TRENDS=True but trend selection failed, history_words was already built above.
        avoid_words = list(dict.fromkeys(history_words + strategy.get("avoid_words", [])))
        strategy = {**strategy, "avoid_words": avoid_words}
        word_prompt = _build_word_prompt(strategy)
        raw_word = retry_call(
            _get_word_pick_ai(),
            word_prompt,
            "Du bist ein Deutschlehrer und erstellst Vokabelinhalte für Social Media. Du antwortest ausschließlich mit gültigem JSON.",
            max_tokens=40,
            temperature=0.9,
            label="pick_word",
        ).strip()
        try:
            # Strip markdown fences if present
            lines = raw_word.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            word_data = json.loads("\n".join(lines))
            german_word = (word_data.get("word") or "").strip()
            word_cefr = (word_data.get("cefr") or "").strip().upper()
            if word_cefr not in _VALID_CEFR:
                word_cefr = ""
        except (json.JSONDecodeError, AttributeError):
            # Fallback: treat the whole response as just a word (old behaviour)
            german_word = raw_word
            word_cefr = ""

    for art in ("der ", "die ", "das ", "Der ", "Die ", "Das "):
        if german_word.startswith(art):
            german_word = german_word[len(art):]
            break

    # ── 1b. Semantic similarity gate (catches deutsch/deutsche etc.) ──────────
    if not avoid_words:
        avoid_words = strategy.get("avoid_words", [])

    _MAX_SIMILARITY_RETRIES = 3
    for _sim_attempt in range(_MAX_SIMILARITY_RETRIES):
        is_similar, matched = _is_word_too_similar(german_word, avoid_words)
        if not is_similar:
            break
        ui_warn(
            f"'{german_word}' is too similar to previously used '{matched}' "
            f"— picking a new word (attempt {_sim_attempt + 1}/{_MAX_SIMILARITY_RETRIES})"
        )
        logger.info(
            "Similarity gate rejected '%s' (too close to '%s') — re-picking.",
            german_word, matched,
        )
        avoid_words.append(german_word)
        strategy = {**strategy, "avoid_words": list(dict.fromkeys(
            avoid_words + strategy.get("avoid_words", [])
        ))}
        word_prompt = _build_word_prompt(strategy)
        raw_word = retry_call(
            _get_word_pick_ai(),
            word_prompt,
            "Du bist ein Deutschlehrer und erstellst Vokabelinhalte für Social Media. "
            "Du antwortest ausschließlich mit gültigem JSON.",
            max_tokens=40,
            temperature=0.9,
            label="pick_word_retry",
        ).strip()
        try:
            lines = raw_word.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            word_data = json.loads("\n".join(lines))
            german_word = (word_data.get("word") or "").strip()
            word_cefr = (word_data.get("cefr") or "").strip().upper()
            if word_cefr not in _VALID_CEFR:
                word_cefr = ""
        except (json.JSONDecodeError, AttributeError):
            german_word = raw_word
            word_cefr = ""
        for art in ("der ", "die ", "das ", "Der ", "Die ", "Das "):
            if german_word.startswith(art):
                german_word = german_word[len(art):]
                break

    cefr_display = f" [{word_cefr}]" if word_cefr else ""
    ok(f"Word selected: {german_word}{cefr_display}")
    logger.info("Picked word: %s  CEFR: %s", german_word, word_cefr or "unknown")

    # ── 2. Gather context for the single AI call ───────────────────────────────
    from nodes.score import _load_history, get_top_tweets
    from scaffolds import next_scaffold
    scaffold_name, scaffold = next_scaffold()
    info(f"Scaffold: {scaffold_name}")
    logger.info("Selected scaffold: %s", scaffold_name)
    top_tweets: list = []  # disabled — past examples caused topic echo-chamber (food/pizza bias)
    logger.info("Past tweet examples disabled — no in-context examples passed to tweet AI.")

    tweet_ai = _get_tweet_ai()

    # ── 3. Generate 3 candidates → pick best ──────────────────────────────────
    candidates = _call_tweet_ai(german_word, scaffold, strategy, top_tweets, tweet_ai, cefr_level=word_cefr, word_from_trends=word_from_trends)
    logger.info("Generated %d tweet candidate(s).", len(candidates))
    result = _select_best_tweet(candidates, german_word, word_cefr)

    # ── 4. Use AI-generated article directly ───────────────────────────────────
    result["full_tweet"] = result.get("tweet", "")

    # Safety-net: remove any literal "no known noun" the AI may have written in the tweet body
    result["full_tweet"] = result["full_tweet"].replace("no known noun ", "").replace("no known noun", "")

    # Ensure cefr_level in result uses the pre-determined level if AI omitted it
    if word_cefr and result.get("cefr_level", "") not in _VALID_CEFR:
        result["cefr_level"] = word_cefr

    # ── 5. Length check with retry ─────────────────────────────────────────────
    max_retries = 2
    for attempt in range(max_retries + 1):
        tweet_len = len(result["full_tweet"])
        if tweet_len <= MAX_TWEET_LENGTH:
            break
        if attempt < max_retries:
            logger.warning(
                "Tweet too long (%d chars) on attempt %d — retrying with length constraint.",
                tweet_len, attempt + 1,
            )
            ui_warn(f"Tweet too long ({tweet_len} chars) — retrying with length constraint …")
            retry_candidates = _call_tweet_ai(
                german_word, scaffold, strategy, top_tweets, tweet_ai,
                cefr_level=word_cefr,
                extra_instruction=f"Keep total length under {MAX_TWEET_LENGTH - 20} characters",
                word_from_trends=word_from_trends,
            )
            result = _select_best_tweet(retry_candidates, german_word, word_cefr)
            result["full_tweet"] = result.get("tweet", "")
        else:
            logger.warning(
                "Tweet still too long (%d chars) after %d retries — truncating emoji pairs.",
                tweet_len, max_retries,
            )
            ui_warn(f"Tweet still too long ({tweet_len} chars) — truncating emoji pairs.")
            result["full_tweet"] = _truncate_emoji_pairs(result["full_tweet"])

    full_tweet = result["full_tweet"]
    tweet_box(full_tweet)
    logger.info("Full tweet assembled (%d chars):\n%s", len(full_tweet), full_tweet)

    # Start loading ImageReward in the background now — it will be ready by the
    # time Midjourney image generation finishes and pick_best_image() is called.
    from services.image_ranker import warmup as _warmup_image_ranker
    _warmup_image_ranker()

    return {
        **state,
        "german_word":         result.get("german_word", german_word),
        "article":             result.get("article", "no known noun"),
        "cefr_level":          result.get("cefr_level", ""),
        "example_sentence_de": result.get("example_sentence_de", ""),
        "example_sentence_en": result.get("example_sentence_en", ""),
        "full_tweet":          full_tweet,
        "cycle":               cycle,
        "error":               None,
    }
