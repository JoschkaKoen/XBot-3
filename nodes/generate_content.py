"""
Node: generate_content

Picks a German word (trend-based or AI-free), then generates the complete
tweet in a single Grok call that returns JSON.  Local post-processing handles
tweet-length enforcement.
"""

import json
import logging
from typing import Optional

import config
from services.ai_client import get_ai_response
from services.x_trends import get_trends
from utils.retry import retry_call
from utils.ui import stage_banner, ok, tweet_box, info, warn as ui_warn


def _get_tweet_ai() -> callable:
    """Return the AI function to use for tweet generation."""
    if config.AI_PROVIDER == "grok":
        if config.TWEET_MODEL == "flagship":
            from services.grok_ai import get_grok_flagship_response
            return get_grok_flagship_response
        if config.TWEET_MODEL == "reasoning":
            from services.grok_ai import get_grok_reasoning_response
            return get_grok_reasoning_response
    return get_ai_response


def _get_tweet_picker_ai() -> callable:
    """Return the AI function to use for picking the best tweet candidate (TWEET_PICKER_MODEL)."""
    return _model_to_ai_fn(config.TWEET_PICKER_MODEL)


def _model_to_ai_fn(model: str) -> callable:
    """Resolve a model name string to the corresponding Grok AI function."""
    if config.AI_PROVIDER == "grok":
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
    return _model_to_ai_fn(config.TREND_FILTER_MODEL)


def _get_word_pick_ai() -> callable:
    """AI function for free-form word selection (WORD_PICK_MODEL)."""
    return _model_to_ai_fn(config.WORD_PICK_MODEL)


def _get_similarity_ai() -> callable:
    """AI function for semantic duplicate / similarity check (SIMILARITY_MODEL)."""
    return _model_to_ai_fn(config.SIMILARITY_MODEL)


def _load_strategy_from_file() -> dict:
    from nodes.analyze import load_strategy
    return load_strategy()


logger = logging.getLogger("german_bot.generate_content")


# ── helpers ───────────────────────────────────────────────────────────────────

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
    src = config.SOURCE_LANGUAGE
    tgt = config.TARGET_LANGUAGE

    topic_line = f"- Topic / angle for this tweet: {next_topic}\n" if next_topic else ""
    style_line  = f"- Style hint: {style}\n" if style else ""

    avoid_block = (
        f"\nCRITICAL — MUST NOT return any of these already-used words: {avoid_str}\n"
        if avoid_str != "none"
        else ""
    )

    return (
        f"You are a {src} teacher creating content for an X account that teaches "
        f"{src} vocabulary to {tgt} speakers.\n\n"
        f"Pick EXACTLY ONE {src} word (noun, verb, adjective, or common phrase) that:\n"
        f"- Is frequently used and widespread in everyday life (no technical jargon, no rare words)\n"
        f"- Has a positive or neutral meaning — avoid words with negative, sad, or depressing connotations\n"
        f"- Is practical and useful in daily life\n"
        f"- Fits one of these CEFR levels: {cefr_hint}\n"
        f"{topic_line}"
        f"{style_line}"
        f"{avoid_block}\n"
        f"Reply with ONLY a valid JSON object — no explanation, no article outside the JSON:\n"
        f'{{"word": "<the {src} word WITHOUT article>", "cefr": "<A1|A2|B1|B2|C1|C2>"}}'
    )


def _expand_scaffold(scaffold: str) -> str:
    """Substitute language-pair config values into scaffold placeholders at runtime."""
    return (
        scaffold
        .replace("[SOURCE_FLAG]",     config.SOURCE_FLAG)
        .replace("[TARGET_FLAG]",     config.TARGET_FLAG)
        .replace("[SOURCE_LANGUAGE]", config.SOURCE_LANGUAGE)
        .replace("[TARGET_LANGUAGE]", config.TARGET_LANGUAGE)
    )


def _build_tweet_prompt(
    trending_word: str,
    scaffold: str,
    strategy: dict,
    top_tweets: list,
    cefr_level: str = "",
    extra_instruction: str = "",
    word_from_trends: bool = False,
    funny: bool = True,
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
    if funny:
        funny_tone_section = (
            "## Tone\n"
            "The example sentence MUST be genuinely very funny — it should make the reader laugh and smirk. "
            "The humor should be positive and uplifting, not negative and cynical."
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

    src = config.SOURCE_LANGUAGE
    tgt = config.TARGET_LANGUAGE
    expanded_scaffold = _expand_scaffold(scaffold)

    prompt = (
        f"You are a {src} language teacher and comedian running a very popular X (Twitter) account that teaches "
        f"{src} to {tgt} speakers.\n\n"
        f"## Your task\n"
        f'Create a complete tweet about this {src} word: "{trending_word}"\n\n'
        f"## Tweet format (follow this scaffold exactly)\n"
        f"{expanded_scaffold}\n\n"
        f"{funny_tone_section}"
        "## Rules\n"
        f"- The {src} word must include the correct grammatical article if it's a noun\n"
        "- If the word has no grammatical article (adjective, phrase, verb, etc.), omit the [ARTICLE] slot entirely\n"
        f"{cefr_rule}"
        "- The example sentence MUST contain the exact word\n"
        f"- The {src} example sentence must be at most {config.MAX_EXAMPLE_WORDS} words long\n"
        f"- Make sure the {src} example sentence is not longer than {config.MAX_EXAMPLE_WORDS} words\n"
        f"- The {tgt} translations must be natural, not robotic\n"
        "- The emojis should help the reader visually understand the meaning. Don't just use laughing emojis\n"
        "## Strategy guidance\n"
        f"- Preferred CEFR levels: {preferred_cefr}\n"
        + (f"- Topic / angle for this tweet: {next_topic}\n" if next_topic else "")
        + (f"- Style instruction: {style}\n" if style else "")
        + "\n"
        "## Output\n"
        "Return ONLY a single valid JSON object with these fields:\n\n"
        "{\n"
        '  "tweet": "<the complete tweet text, exactly matching the scaffold>",\n'
        f'  "source_word": "<the bare {src} word without article>",\n'
        '  "article": "<grammatical article / no known noun>",\n'
        '  "cefr_level": "<A1 / A2 / B1 / B2 / C1 / C2>",\n'
        f'  "example_sentence_source": "<just the {src} example sentence>",\n'
        f'  "example_sentence_target": "<just the {tgt} translation of the sentence>"\n'
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
    funny: bool = True,
) -> list:
    """Fire 3 parallel API calls, each generating one tweet candidate. Returns list of dicts."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    prompt = _build_tweet_prompt(trending_word, scaffold, strategy, top_tweets, cefr_level, extra_instruction, word_from_trends=word_from_trends, funny=funny)
    src = config.SOURCE_LANGUAGE
    tgt = config.TARGET_LANGUAGE
    if funny:
        system_prompt = (
            f"You are a {src} language teacher and comedy writer creating funny tweets for X (Twitter). "
            "Every example sentence must be genuinely very funny — it must make the reader laugh."
            "It should be positive and uplifting, not negative and cynical."
            "You always respond with valid JSON only."
        )
    else:
        system_prompt = (
            f"You are a {src} language teacher and comedian creating funny and uplifting tweets for "
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
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning("Candidate %d: invalid JSON (%s) — skipping.", idx, e)
            raise
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


def _select_best_tweet(candidates: list, source_word: str, cefr_level: str, funny: bool = True) -> dict:
    """
    Print all candidates to the terminal, use the configured TWEET_PICKER_MODEL to pick
    the best one, and highlight the winner. Falls back to the first candidate if selection fails.
    """

    _R    = "\033[0m"
    _BOLD = "\033[1m"
    _CYAN = "\033[96m"
    _GRAY = "\033[90m"
    _GREEN = "\033[92m"

    if not candidates:
        raise ValueError("No tweet candidates succeeded; all 3 generations failed or returned invalid JSON. Check logs and try again.")
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

    src = config.SOURCE_LANGUAGE
    tgt = config.TARGET_LANGUAGE
    if funny:
        selection_prompt = (
            f"You are evaluating {len(candidates)} {src} vocabulary tweet candidates for the word "
            f"'{source_word}' (CEFR: {cefr_level or 'unknown'}).\n\n"
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
            f"You are a comedy editor selecting the funniest {src} vocabulary tweet. "
            "Your two criteria are: (1) funniness — always pick the sharpest joke, "
            "(2) positivity as tiebreaker — warm beats cynical. "
            "Reply with ONLY a number (1, 2 or 3) followed by a short reason — nothing else before the number."
        )
    else:
        selection_prompt = (
            f"You are evaluating {len(candidates)} {src} vocabulary tweet candidates for the word "
            f"'{source_word}' (CEFR: {cefr_level or 'unknown'}).\n\n"
            f"{numbered}\n\n"
            "Score each candidate on these criteria and pick the BEST overall:\n"
            "- Warmth & authenticity: Is the example sentence natural and relatable?\n"
            f"- Engagement potential: Would a native {tgt} speaker stop scrolling and interact with this tweet?\n"
            "- Shareability: Is it quotable, relatable, or surprising enough to share?\n"
            f"- Learning effect: Does the example sentence make the {src} word memorable and easy to understand?\n"
            f"- Sentence quality: Is the {src} sentence natural, correctly matched to the CEFR level, and short enough to read at a glance?\n\n"
            f"{_fmt_instruction}\n\n"
            "Your answer:"
        )
        system_prompt_selector = (
            f"You are a social media expert and {src} teacher evaluating tweet quality. "
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
    trends = get_trends(max_trends=20)

    if not trends:
        ui_warn("No trends available — falling back to AI word selection.")
        return None

    trend_names = [t["name"] for t in trends]
    src = config.SOURCE_LANGUAGE
    tgt = config.TARGET_LANGUAGE
    country = config.TRENDS_COUNTRY.capitalize()
    logger.info("Fetched %d trends (%s):", len(trends), country)
    lines = "\n".join(f"  {i+1:2d}. {t['name']}" for i, t in enumerate(trends))
    print(f"\n  📈  Current {country} Trends ({len(trends)}):\n{lines}\n", flush=True)
    logger.debug("All trends: %s", trend_names)

    trend_list_str = "\n".join(f"{i+1}. {name}" for i, name in enumerate(trend_names))
    avoid_set = {w.lower() for w in avoid_words}
    pick_req = (
        f"You are a {src} teacher selecting vocabulary words for a post on X (Twitter).\n\n"
        f"Here are the current trending topics in {country}:\n{trend_list_str}\n\n"
        f"Your task: extract up to 15 {src} words or short phrases from these trends "
        f"that are valuable for a {tgt}-speaking {src} learner (A1–C2).\n\n"
        "IMPORTANT — ranking criterion: rank exclusively by learning value for the learner, "
        "NOT by relevance to the trend topic. A common everyday word is always more valuable "
        "than a technical or news-specific term.\n\n"
        "Ranking priority (top to bottom):\n"
        f"  1. Everyday words every {src} learner must know (nouns, verbs, adjectives of daily life)\n"
        f"  2. Culturally iconic {src} words with high recognition value\n"
        "  3. Thematically relevant words with moderate learning value\n"
        "  4. Technical or uncommon words — only if no better word is available\n\n"
        "Exclusion criteria (do NOT include these):\n"
        f"- Loanwords or internationalisms with no {src} learning value (e.g. Pipeline, Internet, App, Livestream)\n"
        "- Proper nouns, brand names, cities, personal names\n"
        "- Rare or archaic words that hardly appear in real everyday life\n"
        "- Political jargon that only appears in news contexts\n"
        "- Strip any leading # or @ from each word\n\n"
        "Reply with ONLY a valid JSON array, nothing else:\n"
        '[{"word": "<source language word>", "cefr": "<A1|A2|B1|B2|C1|C2>"}, ...]\n'
        "If no trend yields a suitable word, reply with an empty array: []"
    )

    try:
        raw = retry_call(
            _get_trend_filter_ai(),
            pick_req,
            f"You are an experienced {src} teacher. Your only goal is to select words that are maximally useful for {tgt}-speaking {src} learners in real everyday life. Trend relevance is secondary — learning value has absolute priority. Reply exclusively with valid JSON.",
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

        # Pre-scan to find chosen word and count free slots.
        # Only the top TREND_CANDIDATE_LIMIT entries are considered; the rest are
        # shown in the display but never selected (triggers AI fallback instead).
        _candidate_limit = config.TREND_CANDIDATE_LIMIT
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
            in_window = len(clean_entries) < _candidate_limit
            clean_entries.append((word, cefr, used))
            if in_window and not used:
                free_count += 1
                if chosen_word is None:
                    chosen_word = word
                    chosen_cefr = cefr

        print(f"\n  {_CYAN}{_BOLD}🏆  Trend word shortlist  ({free_count} free, top {_candidate_limit} considered):{_R}", flush=True)
        for i, (word, cefr, used) in enumerate(clean_entries, 1):
            cefr_fmt = f"{_CYAN}{cefr:<3}{_R}"
            in_window = i <= _candidate_limit
            if not in_window:
                status   = f"{_GRAY}– beyond limit{_R}"
                word_fmt = f"{_GRAY}{word:<20}{_R}"
            elif used:
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

        logger.warning(
            "All top-%d trend candidates already used — falling back to AI word selection.",
            _candidate_limit,
        )
        ui_warn(
            f"All top-{_candidate_limit} trend words already used — falling back to AI word selection."
        )
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
    src = config.SOURCE_LANGUAGE

    prompt = (
        f"Already used {src} words: [{word_list_str}]\n\n"
        f"New proposed word: \"{word}\"\n\n"
        f"Is this word too similar to any of the already used words? "
        "Too similar means: same root, inflection, conjugation, compound, "
        "or closely related derivation "
        "(e.g. deutsch/deutsche, Haus/Häuser, laufen/Lauf, Freund/Freundschaft).\n\n"
        "Reply with ONLY a JSON object:\n"
        '{"similar": true, "matched": "<the similar word>"}\n'
        "or\n"
        '{"similar": false, "matched": ""}'
    )

    try:
        raw = retry_call(
            _get_similarity_ai(),
            prompt,
            f"You check whether a {src} word is too similar to a list of already used words. "
            "Reply exclusively with valid JSON.",
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
    tweet_style: str = config.resolve_tweet_style(cycle)

    # ── 1. Pick a source-language word + determine its CEFR level ─────────────
    german_word: Optional[str] = None
    word_cefr: str = ""
    avoid_words: list = []
    word_from_trends: bool = False

    if config.USE_TRENDS:
        info(f"USE_TRENDS=true — fetching trends ({config.TRENDS_COUNTRY}) …")
        from nodes.score import _load_history
        history_words = [r.get("source_word", "") for r in _load_history() if r.get("source_word")]
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
        from nodes.score import _load_history
        history_words = [r.get("source_word", "") for r in _load_history() if r.get("source_word")]
        # Enrich avoid_words with full history before building the word prompt.
        avoid_words = list(dict.fromkeys(history_words + strategy.get("avoid_words", [])))
        strategy = {**strategy, "avoid_words": avoid_words}
        word_prompt = _build_word_prompt(strategy)
        src = config.SOURCE_LANGUAGE
        avoid_sys_str = ", ".join(f'"{w}"' for w in avoid_words[-20:]) if avoid_words else "none"
        raw_word = retry_call(
            _get_word_pick_ai(),
            word_prompt,
            f"You are a {src} teacher creating vocabulary content for social media. "
            "You select exclusively common, widely-used words with positive or neutral meaning — no rare, negative, or depressing words. "
            f"You MUST NOT return any of these already-used words: {avoid_sys_str}. "
            "You reply exclusively with valid JSON.",
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

    # ── 1b. Semantic similarity gate (catches deutsch/deutsche etc.) ──────────
    if not avoid_words:
        avoid_words = strategy.get("avoid_words", [])

    _MAX_SIMILARITY_RETRIES = 3
    rejected_words: list = []
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
        rejected_words.append(german_word)
        avoid_words.append(german_word)
        strategy = {**strategy, "avoid_words": list(dict.fromkeys(
            avoid_words + strategy.get("avoid_words", [])
        ))}
        rejected_str = ", ".join(f'"{w}"' for w in rejected_words)
        word_prompt = _build_word_prompt(strategy)
        word_prompt += (
            f"\n\nCRITICAL: You MUST NOT return any of these words — "
            f"they were all just rejected because they are already used: {rejected_str}. "
            f"You MUST pick a completely different word."
        )
        raw_word = retry_call(
            _get_word_pick_ai(),
            word_prompt,
            f"You are a {src} teacher creating vocabulary content for social media. "
            "You select exclusively common, widely-used words with positive or neutral meaning — no rare, negative, or depressing words. "
            f"You MUST NOT return any of these already-used words: {rejected_str}. "
            "You reply exclusively with valid JSON.",
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

    cefr_display = f" [{word_cefr}]" if word_cefr else ""
    ok(f"Word selected: {german_word}{cefr_display}")
    logger.info("Picked word: %s  CEFR: %s", german_word, word_cefr or "unknown")

    # ── 2. Gather context for the single AI call ───────────────────────────────
    from nodes.score import _load_history
    from scaffolds import next_scaffold
    scaffold_name, scaffold = next_scaffold()
    info(f"Scaffold: {scaffold_name}")
    logger.info("Selected scaffold: %s", scaffold_name)
    top_tweets: list = []  # disabled — past examples caused topic echo-chamber (food/pizza bias)
    logger.info("Past tweet examples disabled — no in-context examples passed to tweet AI.")

    tweet_ai = _get_tweet_ai()

    # ── 3. Generate 3 candidates → pick best ──────────────────────────────────
    funny = tweet_style == "funny"
    candidates = _call_tweet_ai(german_word, scaffold, strategy, top_tweets, tweet_ai, cefr_level=word_cefr, word_from_trends=word_from_trends, funny=funny)
    logger.info("Generated %d tweet candidate(s).", len(candidates))
    result = _select_best_tweet(candidates, german_word, word_cefr, funny=funny)

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
        if tweet_len <= config.MAX_TWEET_LENGTH:
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
                extra_instruction=f"Keep total length under {config.MAX_TWEET_LENGTH - 20} characters",
                word_from_trends=word_from_trends,
                funny=funny,
            )
            result = _select_best_tweet(retry_candidates, german_word, word_cefr, funny=funny)
            result["full_tweet"] = result.get("tweet", "")
        else:
            logger.warning(
                "Tweet still too long (%d chars) after %d retries — truncating emoji pairs, then hard cap.",
                tweet_len, max_retries,
            )
            ui_warn(f"Tweet still too long ({tweet_len} chars) — truncating emoji pairs, then hard cap.")
            result["full_tweet"] = _truncate_emoji_pairs(result["full_tweet"])
            # Hard cap in case emoji truncation didn't get under limit
            if len(result["full_tweet"]) > config.MAX_TWEET_LENGTH:
                result["full_tweet"] = result["full_tweet"][: config.MAX_TWEET_LENGTH - 3].rstrip() + "…"

    full_tweet = result["full_tweet"]
    tweet_box(full_tweet)
    logger.info("Full tweet assembled (%d chars):\n%s", len(full_tweet), full_tweet)

    # Start loading ImageReward in the background now — it will be ready by the
    # time Midjourney image generation finishes and pick_best_image() is called.
    from services.image_ranker import warmup as _warmup_image_ranker
    _warmup_image_ranker()

    return {
        **state,
        "source_word":             result.get("source_word", german_word),
        "article":                 result.get("article", "no known noun"),
        "cefr_level":              result.get("cefr_level", ""),
        "example_sentence_source": result.get("example_sentence_source", ""),
        "example_sentence_target": result.get("example_sentence_target", ""),
        "full_tweet":              full_tweet,
        "cycle":                   cycle,
        "error":                   None,
    }
