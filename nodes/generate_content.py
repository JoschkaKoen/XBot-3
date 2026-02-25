"""
Node: generate_content

Picks a German word (trend-based or AI-free), then generates the complete
tweet in a single Grok call that returns JSON.  Local post-processing handles
article verification and tweet-length enforcement.
"""

import json
import logging
import string
from typing import Any, Optional

from config import USE_TRENDS, SENTENCE_MODEL, AI_PROVIDER, CONTROVERSIAL_MODE
from services.ai_client import get_ai_response
from services.get_article import get_article
from services.x_trends import get_germany_trends
from utils.retry import retry_call
from utils.ui import stage_banner, ok, tweet_box, info, warn as ui_warn


def _get_tweet_ai() -> callable:
    """Return the AI function to use for tweet generation."""
    if AI_PROVIDER == "grok":
        if SENTENCE_MODEL == "flagship":
            from services.grok_ai import get_grok_flagship_response
            return get_grok_flagship_response
        if SENTENCE_MODEL == "reasoning":
            from services.grok_ai import get_grok_reasoning_response
            return get_grok_reasoning_response
    return get_ai_response


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
    focus = strategy.get("focus", "")
    cefr_hint = strategy.get("preferred_cefr", "A1, A2, B1, B2, C1, C2")
    theme_hint = strategy.get("preferred_themes", "food, travel, daily life, emotions")
    avoid = strategy.get("avoid_words", [])
    avoid_str = ", ".join(avoid[-20:]) if avoid else "none"

    return (
        "Du bist ein Deutschlehrer und erstellst Inhalte für einen X-Account, "
        "der englischsprachigen Nutzern deutsche Vokabeln beibringt.\n\n"
        "Wähle GENAU EIN deutsches Wort (Nomen, Verb, Adjektiv oder gebräuchliche Phrase), das:\n"
        f"- Im Alltag praktisch und nützlich ist\n"
        f"- Zu einem dieser GER-Niveaus passt: {cefr_hint}\n"
        f"- Einem dieser Themen entspricht: {theme_hint}\n"
        f"- Nicht kürzlich verwendet wurde (vermeiden: {avoid_str})\n"
        f"{'- Zusätzlicher Fokus: ' + focus if focus else ''}\n\n"
        "Antworte NUR mit dem deutschen Wort oder der Phrase – keine Erklärung, kein Artikel, keine Satzzeichen."
    )


def _build_tweet_prompt(
    trending_word: str,
    scaffold: str,
    strategy: dict,
    top_tweets: list,
    extra_instruction: str = "",
) -> str:
    preferred_cefr = strategy.get("preferred_cefr", "A1, A2, B1, B2, C1, C2")
    preferred_themes = strategy.get("preferred_themes", "food, travel, daily life, emotions")
    focus = strategy.get("focus", "")

    examples_section = ""
    for i, tweet in enumerate(top_tweets[:3], 1):
        score = tweet.get("engagement_score", 0.0)
        text = tweet.get("full_tweet", "")
        examples_section += f"\n### Tweet {i} (score: {score}):\n{text}\n"
    if not examples_section:
        examples_section = "\n(No past tweets available yet)\n"

    prompt = (
        "You are a German language teacher running a popular X (Twitter) account that teaches "
        "German to English speakers.\n\n"
        f"## Your task\n"
        f'Create a complete tweet about this German word: "{trending_word}"\n\n'
        f"## Tweet format (follow this scaffold exactly)\n"
        f"{scaffold}\n\n"
        "## Rules\n"
        "- The German word must include the correct article (der/die/das) if it's a noun\n"
        "- Pick an appropriate CEFR level (A1-C2) for this word\n"
        "- The example sentence must be short, funny, warm, and authentic — something a German would actually say\n"
        "- Vary the sentence ending: most sentences should end with a period (.) — only use an exclamation mark (!) when it is genuinely funny or surprising, and occasionally use a question (?)\n"
        "- The example sentence MUST contain the exact word\n"
        "- The English translations must be natural, not robotic\n"
        "- Each line gets a pair of 2 identical emojis that visually represent the meaning\n"
        "- The two emoji PAIRS must be DIFFERENT from each other "
        "(e.g. 🚗🚗 for word line, 🎉🎉 for sentence line — NOT the same pair twice)\n"
        "- The CEFR level must honestly match the vocabulary difficulty of the word\n"
        "- Do NOT wrap the output in quotes or markdown\n\n"
        "## Strategy guidance\n"
        f"- Preferred CEFR levels: {preferred_cefr}\n"
        f"- Preferred themes: {preferred_themes}\n"
        f"- Focus: {focus}\n\n"
        "## Here are the best-performing past tweets (imitate their style and quality):\n"
        f"{examples_section}\n"
        "## Output\n"
        "Return ONLY a valid JSON object with these fields:\n\n"
        "{\n"
        '  "tweet": "<the complete tweet text, exactly matching the scaffold>",\n'
        '  "german_word": "<the bare German word without article, e.g. Führerschein>",\n'
        '  "article": "<der / die / das / no known noun>",\n'
        '  "cefr_level": "<A1 / A2 / B1 / B2 / C1 / C2>",\n'
        '  "example_sentence_de": "<just the German example sentence>",\n'
        '  "example_sentence_en": "<just the English translation of the sentence>"\n'
        "}\n\n"
        "No markdown. No explanation. Just the JSON."
    )

    if CONTROVERSIAL_MODE:
        prompt += (
            "\n\n## Tone (overrides the focus instruction above)\n"
            "Give the example sentence a funny, ironic, or self-aware twist"
            "Think dry humour, gentle sarcasm, or a relatable everyday observation said with a wink. "
            "The sentence must still be natural German and contain the exact word. "
            "IMPORTANT: Keep it light and warm — NOT negative, cynical, or pessimistic. The reader should smile or laugh, not cringe.\n"
            "Examples of the right tone: 'Ich brauche keine Motivation — ich brauche Kaffee.' / "
            "'Mit genug Ausdauer schafft man alles. Außer vielleicht den frühen Bus.' / "
            "'Mein Wortschatz ist beeindruckend. Nur mein Mut, ihn zu benutzen, fehlt noch.'"
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
    extra_instruction: str = "",
) -> dict:
    prompt = _build_tweet_prompt(trending_word, scaffold, strategy, top_tweets, extra_instruction)
    raw = retry_call(
        tweet_ai,
        prompt,
        (
            "You are a German language teacher creating engaging vocabulary content for "
            "X (Twitter). You always respond with valid JSON only."
        ),
        max_tokens=600,
        temperature=0.85,
        label="generate_tweet",
    )
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(raw)


# ── trend-based word selection ─────────────────────────────────────────────────

def _pick_word_from_trends(avoid_words: list) -> Optional[str]:
    """
    Fetch German trends, log them, and ask the AI to choose the best one
    that works as a German vocabulary word.

    Returns the chosen German word/phrase, or None if no suitable trend found
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
        '[{"word": "<deutsches Wort>", "reason": "<ein Satz: warum ist dieses Wort nützlich für Lernende?>"}, ...]\n'
        "Falls kein Trend ein geeignetes Wort liefert, antworte mit einem leeren Array: []"
    )

    try:
        from services.grok_ai import get_grok_reasoning_response
        raw = retry_call(
            get_grok_reasoning_response,
            pick_req,
            "Du bist ein erfahrener Deutschlehrer. Dein einziges Ziel ist es, Wörter auszuwählen, die für englischsprachige Deutschlernende im echten Alltag möglichst nützlich sind. Trendrelevanz ist zweitrangig — Lernwert hat absolute Priorität. Du antwortest ausschließlich mit gültigem JSON.",
            max_tokens=1200,
            temperature=0.3,
            label="trend_pick",
        )
        raw = raw.strip().strip("```json").strip("```").strip()
        candidates: list = json.loads(raw)

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

        chosen_word   = None
        chosen_reason = ""
        free_count    = 0

        # Pre-scan to find chosen word and count free slots
        clean_entries = []
        for entry in candidates:
            word = (entry.get("word") or "").lstrip("#@").strip()
            if not word:
                continue
            used = word.lower() in avoid_set
            clean_entries.append((word, entry.get("reason", ""), used))
            if not used:
                free_count += 1
                if chosen_word is None:
                    chosen_word   = word
                    chosen_reason = entry.get("reason", "")

        print(f"\n  {_CYAN}{_BOLD}🏆  Trend word shortlist  ({free_count} free):{_R}", flush=True)
        for i, (word, reason, used) in enumerate(clean_entries, 1):
            if used:
                status     = f"{_RED}✖ already used{_R}"
                word_fmt   = f"{_GRAY}{word:<20}{_R}"
            elif word == chosen_word:
                status     = f"{_GREEN}✔ selected{_R}"
                word_fmt   = f"{_BOLD}{word:<20}{_R}"
            else:
                status     = f"{_GRAY}○ free{_R}"
                word_fmt   = f"{word:<20}"
            print(f"  {_GRAY}{i}.{_R} {word_fmt} {status}  {_GRAY}{reason}{_R}", flush=True)

        print(flush=True)

        if chosen_word:
            logger.info("Trend word chosen: '%s' — %s", chosen_word, chosen_reason)
            ok(f"Trend word: '{chosen_word}' — {chosen_reason}")
            return chosen_word

        logger.warning("All %d trend candidates already used — falling back to AI word selection.", len(candidates))
        ui_warn(f"All suitable trend words already used ({len(candidates)} checked) — falling back to AI word selection.")
        return None

    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Failed to parse trend-pick response (%s) — falling back.", exc)
        ui_warn("Trend word selection failed — falling back to AI word selection.")
        return None


# ── main node ──────────────────────────────────────────────────────────────────

def generate_content(state: dict) -> dict:
    stage_banner(3)
    logger.info("Node: generate_content")

    strategy: dict = state.get("strategy") or _load_strategy_from_file()
    cycle: int = state.get("cycle", 0)

    # ── 1. Pick a German word ──────────────────────────────────────────────────
    german_word: Optional[str] = None

    if USE_TRENDS:
        info("USE_TRENDS=true — fetching German trends …")
        from nodes.score import _load_history
        history_words = [r.get("german_word", "") for r in _load_history() if r.get("german_word")]
        strategy_words = strategy.get("avoid_words", [])
        avoid_words = list(dict.fromkeys(history_words + strategy_words))
        avoid_preview = ", ".join(avoid_words[-5:]) if avoid_words else "none"
        logger.debug("Trend word selection: avoiding %d word(s): %s", len(avoid_words), avoid_words[-20:])
        logger.info("Avoiding %d previously used word(s) (e.g. %s)", len(avoid_words), avoid_preview)
        german_word = _pick_word_from_trends(avoid_words)

    if not german_word:
        if not USE_TRENDS:
            from nodes.score import _load_history
            history_words = [r.get("german_word", "") for r in _load_history() if r.get("german_word")]
        # Always enrich avoid_words with the full history before building the word prompt.
        # When USE_TRENDS=True but trend selection failed, history_words was already built above.
        strategy = {**strategy, "avoid_words": list(dict.fromkeys(history_words + strategy.get("avoid_words", [])))}
        word_prompt = _build_word_prompt(strategy)
        german_word = retry_call(
            get_ai_response,
            word_prompt,
            "Du bist ein Deutschlehrer und erstellst Vokabelinhalte für Social Media. Du schlägst einzelne deutsche Wörter vor, die praktisch, interessant und für Lernende geeignet sind.",
            max_tokens=30,
            temperature=0.9,
            label="pick_word",
        ).strip()

    for art in ("der ", "die ", "das ", "Der ", "Die ", "Das "):
        if german_word.startswith(art):
            german_word = german_word[len(art):]
            break
    ok(f"Word selected: {german_word}")
    logger.info("Picked word: %s", german_word)

    # ── 2. Gather context for the single AI call ───────────────────────────────
    from nodes.score import _load_history, get_top_tweets
    scaffold: str = strategy.get("scaffold") or _load_strategy_from_file()["scaffold"]
    top_tweets = get_top_tweets(_load_history())
    logger.info("Using %d top tweet(s) as in-context examples.", len(top_tweets))

    tweet_ai = _get_tweet_ai()

    # ── 3. Single AI call → JSON ───────────────────────────────────────────────
    result = _call_tweet_ai(german_word, scaffold, strategy, top_tweets, tweet_ai)

    # ── 4. Article verification (local, no extra API call) ─────────────────────
    verified_article = get_article(result.get("german_word", german_word))
    ai_article = result.get("article", "no known noun")
    if (
        verified_article != ai_article
        and verified_article not in ("no known noun", "")
        and ai_article not in ("no known noun", "")
    ):
        logger.warning(
            "Fixed article: %s → %s for \"%s\"", ai_article, verified_article, result.get("german_word")
        )
        print(
            f'  ⚠️  Fixed article: {ai_article} → {verified_article} for "{result.get("german_word")}"',
            flush=True,
        )
        result["full_tweet"] = result.get("tweet", "").replace(
            f"{ai_article} {result.get('german_word', german_word)}",
            f"{verified_article} {result.get('german_word', german_word)}",
            1,
        )
        result["article"] = verified_article
    else:
        result["full_tweet"] = result.get("tweet", "")

    # ── 5. Length check with retry ─────────────────────────────────────────────
    max_retries = 2
    for attempt in range(max_retries + 1):
        tweet_len = len(result["full_tweet"])
        if tweet_len <= 280:
            break
        if attempt < max_retries:
            logger.warning(
                "Tweet too long (%d chars) on attempt %d — retrying with length constraint.",
                tweet_len, attempt + 1,
            )
            ui_warn(f"Tweet too long ({tweet_len} chars) — retrying with length constraint …")
            result = _call_tweet_ai(
                german_word, scaffold, strategy, top_tweets, tweet_ai,
                extra_instruction="Keep total length under 260 characters",
            )
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
